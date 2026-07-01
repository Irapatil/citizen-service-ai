from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import AsyncGenerator

import structlog
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

from backend.graph.state import CitizenServiceState
from backend.graph.workflow import get_compiled_graph
from backend.models.schemas import (
    ExecutionMetrics,
    QueryRequest,
    QueryResponse,
)
from backend.services.llm import get_llm_provider
from backend.services.memory_service import MemoryService

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Citizen Services"])

_AGENT_NODES = {
    "orchestrator_node",
    "planner_node",
    "validator_node",
    "application_agent_node",
    "billing_agent_node",
    "knowledge_agent_node",
    "complaint_agent_node",
    "aggregator_node",
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_initial_state(req: QueryRequest) -> CitizenServiceState:
    memory = MemoryService(req.session_id)
    history_text = memory.get_history_as_text(n=5)

    return CitizenServiceState(
        user_query=req.user_query,
        session_id=req.session_id,
        citizen_id=req.citizen_id,
        messages=[HumanMessage(content=req.user_query)],
        # Legacy fields (kept for backwards compat with older agents)
        intent_list=[],
        complexity="simple",
        requires_parallel=False,
        # New enterprise fields
        query_type="",
        intents=[],
        entities={},
        confidence=0.0,
        intent_confidence={},
        selected_agents=[],
        skipped_agents=[],
        agent_explanations={},
        execution_mode="single",
        validation_result=None,
        execution_timeline=[],
        rag_documents=[],
        # Shared
        execution_plan=None,
        application_result=None,
        billing_result=None,
        knowledge_result=None,
        complaint_result=None,
        final_response="",
        agent_logs=[],
        execution_metrics=ExecutionMetrics(),
        metadata={
            **req.metadata,
            "session_history": history_text,
            "request_time": datetime.now(timezone.utc).isoformat(),
        },
    )


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _build_query_response(req: QueryRequest, final_state: CitizenServiceState) -> QueryResponse:
    metrics = final_state.get("execution_metrics") or ExecutionMetrics()
    vr = final_state.get("validation_result")

    return QueryResponse(
        session_id=req.session_id,
        final_response=final_state.get("final_response", ""),
        intents_detected=final_state.get("intents") or final_state.get("intent_list", []),
        agents_used=[log.agent_name.value for log in final_state.get("agent_logs", [])],
        execution_metrics=metrics,
        application_result=final_state.get("application_result"),
        billing_result=final_state.get("billing_result"),
        knowledge_result=final_state.get("knowledge_result"),
        complaint_result=final_state.get("complaint_result"),
        # New enterprise fields
        query_type=final_state.get("query_type", ""),
        entities=final_state.get("entities", {}),
        confidence=final_state.get("confidence", 0.0),
        selected_agents=final_state.get("selected_agents", []),
        execution_mode=final_state.get("execution_mode", "single"),
        execution_timeline=final_state.get("execution_timeline", []),
        validation_result=vr.dict() if vr and hasattr(vr, "dict") else (vr if isinstance(vr, dict) else None),
        success=True,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/query", response_model=QueryResponse)
async def process_query(req: QueryRequest) -> QueryResponse:
    """Main citizen query endpoint — runs the full multi-agent LangGraph workflow."""
    logger.info("query.received", session=req.session_id, query=req.user_query[:100])

    graph = get_compiled_graph()
    initial_state = _build_initial_state(req)

    try:
        final_state: CitizenServiceState = await graph.ainvoke(initial_state)
    except Exception as exc:
        logger.error("query.graph_error", error=str(exc), session=req.session_id)
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {exc}")

    memory = MemoryService(req.session_id)
    memory.add_turn(
        user_query=req.user_query,
        final_response=final_state.get("final_response", ""),
        agent_outputs={
            "application": final_state.get("application_result"),
            "billing":     final_state.get("billing_result"),
            "knowledge":   final_state.get("knowledge_result"),
            "complaint":   final_state.get("complaint_result"),
        },
        intents=final_state.get("intents") or final_state.get("intent_list", []),
    )

    response = _build_query_response(req, final_state)
    logger.info("query.complete", session=req.session_id, agents=len(response.agents_used))
    return response


@router.post("/query/stream")
async def stream_query(req: QueryRequest) -> StreamingResponse:
    # STEP 4 — backend entry
    print(f"\n{'='*60}")
    print(f"[STEP 4] Entered /query/stream")
    print(f"[STEP 5] Incoming request  session={req.session_id}")
    print(f"[STEP 5] Incoming query    {req.user_query!r}")
    print(f"{'='*60}\n", flush=True)

    graph = get_compiled_graph()
    print(f"[STEP 6] Graph compiled: {graph}", flush=True)
    initial_state = _build_initial_state(req)
    print(f"[STEP 6] Initial state built. Keys: {list(initial_state.keys())}", flush=True)

    async def generate() -> AsyncGenerator[str, None]:
        print("[STEP 6] Generator started - invoking graph.astream_events()", flush=True)
        try:
            final_state: CitizenServiceState | None = None
            seen_starts: set[str] = set()
            seen_ends: set[str] = set()

            async for event in graph.astream_events(initial_state, version="v2"):
                kind: str = event["event"]
                name: str = event.get("name", "")

                # STEP 7 — log every node start/end
                if kind == "on_chain_start" and name in _AGENT_NODES:
                    print(f"[STEP 7] NODE START  -> {name}", flush=True)
                elif kind == "on_chain_end" and name in _AGENT_NODES:
                    out = event["data"].get("output") or {}
                    out_keys = list(out.keys()) if isinstance(out, dict) else type(out).__name__
                    print(f"[STEP 7] NODE END    <- {name}  output_keys={out_keys}", flush=True)
                elif kind == "on_chain_end" and name == "LangGraph":
                    print(f"[STEP 7] GRAPH DONE   LangGraph on_chain_end received", flush=True)

                # Agent start
                if kind == "on_chain_start" and name in _AGENT_NODES and name not in seen_starts:
                    seen_starts.add(name)
                    short = name.replace("_node", "")
                    display = short.replace("_", " ").title()
                    yield _sse({"type": "agent_start", "agent": short, "display": display})

                # Agent complete
                elif kind == "on_chain_end" and name in _AGENT_NODES and name not in seen_ends:
                    seen_ends.add(name)
                    short = name.replace("_node", "")
                    display = short.replace("_", " ").title()
                    output = event["data"].get("output", {}) or {}

                    yield _sse({"type": "agent_complete", "agent": short, "display": display})

                    # Emit enriched events after key agents complete
                    if name == "orchestrator_node":
                        yield _sse({
                            "type": "orchestrator_complete",
                            "query_type": output.get("query_type", ""),
                            "intents": output.get("intents") or output.get("intent_list", []),
                            "intent_confidence": output.get("intent_confidence", {}),
                            "entities": output.get("entities", {}),
                            "confidence": output.get("confidence", 0.0),
                        })

                    elif name == "planner_node":
                        sel = output.get("selected_agents", [])
                        skp = output.get("skipped_agents", [])
                        exp = output.get("agent_explanations", {})
                        # Build structured plan for JSON view
                        plan_obj = output.get("execution_plan")
                        plan_tasks = []
                        if plan_obj:
                            for t in getattr(plan_obj, "tasks", []):
                                plan_tasks.append({
                                    "agent": getattr(t, "agent", {}).value if hasattr(getattr(t, "agent", None), "value") else str(getattr(t, "agent", "")),
                                    "task": getattr(t, "task", ""),
                                    "parameters": getattr(t, "parameters", {}),
                                })
                        yield _sse({
                            "type": "plan_ready",
                            "selected_agents": sel,
                            "skipped_agents": skp,
                            "execution_mode": output.get("execution_mode", "single"),
                            "agent_explanations": exp,
                            "plan_json": {
                                "intents": [],          # filled by orchestrator_complete on frontend
                                "selected_agents": sel,
                                "skipped_agents": skp,
                                "execution_mode": output.get("execution_mode", "single"),
                                "tasks": plan_tasks,
                            },
                        })

                    elif name == "knowledge_agent_node":
                        kr = output.get("knowledge_result")
                        if kr:
                            docs = (
                                getattr(kr, "documents", None)
                                or (kr.get("documents") if isinstance(kr, dict) else None)
                                or []
                            )
                            if docs:
                                yield _sse({
                                    "type": "rag_complete",
                                    "documents": docs[:6],
                                    "count": len(docs),
                                })

                    elif name == "validator_node":
                        vr = output.get("validation_result")
                        if vr:
                            def _vget(key, default=None):
                                return getattr(vr, key, None) or (vr.get(key) if isinstance(vr, dict) else None) or default
                            yield _sse({
                                "type": "validation_complete",
                                "verdict":   _vget("verdict", "Validation complete"),
                                "score":     _vget("completeness_score", 0.9),
                                "is_valid":  _vget("is_valid", True),
                                "issues":    _vget("issues", []),
                                "agents_to_retry": _vget("agents_to_retry", []),
                            })

                # Full graph done
                elif kind == "on_chain_end" and name == "LangGraph":
                    final_state = event["data"].get("output")

            # Stream final response word by word
            if final_state:
                final_text: str = final_state.get("final_response", "")
                print(f"[STEP 9] Returning response — {len(final_text)} chars, streaming tokens…", flush=True)
                words = final_text.split(" ")
                for i, word in enumerate(words):
                    token = word + (" " if i < len(words) - 1 else "")
                    yield _sse({"type": "token", "content": token})
                    await asyncio.sleep(0.02)

                memory = MemoryService(req.session_id)
                memory.add_turn(
                    user_query=req.user_query,
                    final_response=final_text,
                    intents=final_state.get("intents") or final_state.get("intent_list", []),
                )

                # Build per-agent performance metrics from timeline
                tl = final_state.get("execution_timeline", [])
                metrics = [
                    {
                        "agent":        e.get("agent", ""),
                        "display_name": e.get("display_name", ""),
                        "status":       e.get("status", ""),
                        "duration_ms":  e.get("duration_ms"),
                        "is_parallel":  e.get("is_parallel", False),
                        "summary":      e.get("summary", ""),
                    }
                    for e in tl
                ]

                # Recent session memory (last 4 turns before this one)
                recent_turns = memory.get_history()
                session_memory = [
                    {
                        "query":    t.user_query,
                        "intents":  t.intents,
                        "preview":  (t.final_response or "")[:120],
                    }
                    for t in recent_turns[-4:]
                ]

                yield _sse({
                    "type": "done",
                    "session_id": req.session_id,
                    "timeline": tl,
                    "metrics": metrics,
                    "session_memory": session_memory,
                    "query_type": final_state.get("query_type", ""),
                    "selected_agents": final_state.get("selected_agents", []),
                    "skipped_agents": final_state.get("skipped_agents", []),
                    "execution_mode": final_state.get("execution_mode", "single"),
                    "rag_documents": final_state.get("rag_documents", []),
                })
            else:
                yield _sse({"type": "done", "session_id": req.session_id, "timeline": []})

        except Exception as exc:
            import traceback
            # STEP 8 — full traceback so nothing is swallowed
            print(f"\n[STEP 8] EXCEPTION IN GENERATOR:", flush=True)
            traceback.print_exc()
            logger.error("stream_query.error", error=str(exc), session=req.session_id)
            yield _sse({"type": "error", "detail": str(exc)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/diagnose")
async def diagnose() -> dict:
    """
    End-to-end pipeline smoke-test. Runs a fixed query through the full
    LangGraph workflow and returns a structured report of every stage.
    """
    from langchain_core.messages import HumanMessage as _HM

    graph = get_compiled_graph()
    test_query = "What is the status of my application and how do I pay my fees?"
    initial_state = _build_initial_state(
        QueryRequest(user_query=test_query, session_id="diagnose-001")
    )

    report: dict = {
        "query": test_query,
        "stages": {},
        "errors": [],
        "final_response_length": 0,
        "status": "unknown",
    }

    try:
        events_seen: list[str] = []
        final_state = None

        async for event in graph.astream_events(initial_state, version="v2"):
            kind = event["event"]
            name = event.get("name", "")
            label = f"{kind}|{name}"
            events_seen.append(label)

            if kind == "on_chain_end" and name in _AGENT_NODES:
                output = event["data"].get("output") or {}
                report["stages"][name] = {
                    "status": "completed",
                    "output_keys": list(output.keys()) if isinstance(output, dict) else str(type(output)),
                }

            if kind == "on_chain_end" and name == "LangGraph":
                final_state = event["data"].get("output")

        report["events_seen"] = events_seen
        report["agents_completed"] = list(report["stages"].keys())

        if final_state:
            fr = final_state.get("final_response", "")
            report["final_response_length"] = len(fr)
            report["final_response_preview"] = fr[:200]
            report["selected_agents"] = final_state.get("selected_agents", [])
            report["execution_mode"] = final_state.get("execution_mode", "")
            report["status"] = "ok" if fr else "empty_response"
        else:
            report["status"] = "no_final_state"
            report["errors"].append("LangGraph on_chain_end event not received")

    except Exception as exc:
        report["status"] = "error"
        report["errors"].append(str(exc))
        logger.error("diagnose.error", error=str(exc))

    return report


@router.get("/model-info")
async def get_model_info() -> dict:
    provider = get_llm_provider()
    return {"provider": provider.provider_name, "model": provider.model_name, "status": "active"}


@router.get("/session/{session_id}/history")
async def get_session_history(session_id: str) -> dict:
    memory = MemoryService(session_id)
    turns = memory.get_history()
    return {
        "session_id": session_id,
        "total_turns": len(turns),
        "turns": [
            {
                "turn_id": t.turn_id,
                "timestamp": t.timestamp.isoformat(),
                "user_query": t.user_query,
                "final_response": t.final_response,
                "intents": t.intents,
            }
            for t in turns
        ],
    }


@router.delete("/session/{session_id}")
async def clear_session(session_id: str) -> dict:
    MemoryService.delete_session(session_id)
    return {"session_id": session_id, "status": "cleared"}


@router.get("/health")
async def health_check() -> dict:
    provider = get_llm_provider()
    return {
        "status": "healthy",
        "service": "citizen-service-ai",
        "provider": provider.provider_name,
        "model": provider.model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
