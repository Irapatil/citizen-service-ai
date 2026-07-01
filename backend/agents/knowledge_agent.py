from __future__ import annotations

import json
from datetime import datetime

import structlog

from backend.graph.state import CitizenServiceState
from backend.models.schemas import (
    AgentLog,
    AgentName,
    KnowledgeResult,
    TaskStatus,
)
from backend.services.llm import get_llm_provider
from backend.tools.knowledge_tools import retrieve_documents, search_policies

logger = structlog.get_logger(__name__)

_TOOLS = [retrieve_documents, search_policies]

_SYSTEM_PROMPT = """You are the Knowledge Agent for a Government Citizen Services Platform.

Your role: Answer citizen questions using official government policy documents and FAQs.
Always use the provided tools to retrieve accurate, up-to-date information.

Available tools:
- retrieve_documents(query, top_k): Search the FAQ knowledge base semantically.
- search_policies(topic): Search government policy documents.

IMPORTANT: Your answers must be grounded in retrieved documents.
Always synthesise information from multiple sources where relevant.
Cite document sources when providing policy information."""


async def knowledge_agent_node(state: CitizenServiceState) -> dict:
    """
    Level 3 — Knowledge Agent.
    Retrieves FAQ and policy documents using RAG, then synthesises an answer.
    """
    log = AgentLog(agent_name=AgentName.KNOWLEDGE, task="knowledge_retrieval")
    t_start = datetime.utcnow()
    logger.info("knowledge_agent.start")
    print("[STEP 7] Knowledge Agent START", flush=True)

    plan = state.get("execution_plan")
    user_query = state.get("user_query", "")

    knowledge_tasks = [t for t in plan.tasks if t.agent == AgentName.KNOWLEDGE] if plan else []
    if not knowledge_tasks:
        log.complete(TaskStatus.SKIPPED)
        return {
            "agent_logs": [log],
            "execution_timeline": [{
                "agent": "knowledge_agent", "display_name": "Knowledge Agent",
                "status": "skipped", "duration_ms": 0, "summary": "Not selected for this query",
                "is_parallel": True,
            }],
        }

    task_names = {t.task for t in knowledge_tasks}

    try:
        provider = get_llm_provider()
        llm_with_tools = provider.bind_tools(_TOOLS, temperature=0.2)

        from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

        tool_map = {t.name: t for t in _TOOLS}
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Citizen Query: {user_query}\n\n"
                f"Tasks required: {', '.join(task_names)}\n\n"
                "Search the knowledge base and provide a comprehensive, sourced answer."
            )),
        ]

        response = None
        for _ in range(3):
            response = await llm_with_tools.ainvoke(messages)
            messages.append(response)
            if not getattr(response, "tool_calls", None):
                break
            for tc in response.tool_calls:
                tool_fn = tool_map.get(tc["name"])
                if tool_fn:
                    result = tool_fn.invoke(tc["args"])
                    messages.append(ToolMessage(content=json.dumps(result), tool_call_id=tc["id"]))

        final_content = str(response.content) if response else ""

        docs_resp = retrieve_documents.invoke({"query": user_query, "top_k": 4})
        documents = docs_resp.get("documents", [])
        sources = [d.get("id", d.get("source", "")) for d in documents]

        policies_resp = (
            search_policies.invoke({"topic": user_query})
            if any(n in task_names for n in ("search_policies", "policy_search", "search_policy_documents"))
            else {"policies": []}
        )

        result = KnowledgeResult(
            documents=documents,
            policies=policies_resp.get("policies", []),
            answer=final_content,
            sources=[s for s in sources if s],
            success=True,
        )

        log.complete(TaskStatus.SUCCESS)
        duration_ms = (datetime.utcnow() - t_start).total_seconds() * 1000
        logger.info("knowledge_agent.done", docs=len(documents))

        return {
            "knowledge_result": result,
            "rag_documents": documents,
            "agent_logs": [log],
            "execution_timeline": [{
                "agent": "knowledge_agent", "display_name": "Knowledge Agent",
                "status": "completed",
                "start_time": t_start.isoformat(),
                "end_time": datetime.utcnow().isoformat(),
                "duration_ms": round(duration_ms, 1),
                "summary": f"Retrieved {len(documents)} document(s) from knowledge base",
                "is_parallel": True,
            }],
        }

    except Exception as exc:
        logger.error("knowledge_agent.error", error=str(exc))
        log.complete(TaskStatus.FAILED, error=str(exc))
        duration_ms = (datetime.utcnow() - t_start).total_seconds() * 1000
        return {
            "knowledge_result": KnowledgeResult(
                answer="Unable to retrieve knowledge base information at this time.",
                success=False, error=str(exc),
            ),
            "agent_logs": [log],
            "execution_timeline": [{
                "agent": "knowledge_agent", "display_name": "Knowledge Agent",
                "status": "failed", "duration_ms": round(duration_ms, 1),
                "summary": f"Error: {str(exc)[:80]}", "is_parallel": True,
            }],
        }
