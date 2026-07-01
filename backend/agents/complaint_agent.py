from __future__ import annotations

import json
import re
from datetime import datetime

import structlog

from backend.graph.state import CitizenServiceState
from backend.models.schemas import (
    AgentLog,
    AgentName,
    ComplaintResult,
    TaskStatus,
)
from backend.services.llm import get_llm_provider
from backend.tools.complaint_tools import create_complaint, check_complaint_status

logger = structlog.get_logger(__name__)

_TOOLS = [create_complaint, check_complaint_status]

_SYSTEM_PROMPT = """You are the Complaint Agent for a Government Citizen Services Platform.

Your role: Help citizens file complaints and track existing ones.
Use the available tools to create or retrieve complaint records.

Available tools:
- create_complaint(subject, description, category, application_id, priority): File a new complaint.
- check_complaint_status(complaint_id): Check status of an existing complaint.

When creating a complaint:
- Extract a concise subject from the query
- Categorise appropriately (Delay, Service Issue, Billing Error, Misconduct, Other)
- Set priority based on urgency (Low, Medium, High, Critical)
- Always confirm the complaint ID and expected resolution date to the citizen."""


def _extract_complaint_id(query: str, parameters: dict, entities: dict) -> str | None:
    cid = parameters.get("complaint_id") or entities.get("complaint_id")
    if not cid:
        m = re.search(r"CMP[-\s]?\w{6,}", query, re.IGNORECASE)
        if m:
            cid = m.group().replace(" ", "-").upper()
    return cid


async def complaint_agent_node(state: CitizenServiceState) -> dict:
    """Level 3 — Complaint Agent."""
    log = AgentLog(agent_name=AgentName.COMPLAINT, task="complaint_handling")
    t_start = datetime.utcnow()
    logger.info("complaint_agent.start")
    print("[STEP 7] Complaint Agent START", flush=True)

    plan = state.get("execution_plan")
    user_query = state.get("user_query", "")
    entities = state.get("entities", {})

    complaint_tasks = [t for t in plan.tasks if t.agent == AgentName.COMPLAINT] if plan else []
    if not complaint_tasks:
        log.complete(TaskStatus.SKIPPED)
        return {
            "agent_logs": [log],
            "execution_timeline": [{
                "agent": "complaint_agent", "display_name": "Complaint Agent",
                "status": "skipped", "duration_ms": 0, "summary": "Not selected for this query",
                "is_parallel": True,
            }],
        }

    parameters = complaint_tasks[0].parameters if complaint_tasks else {}
    task_names = {t.task for t in complaint_tasks}
    complaint_id = _extract_complaint_id(user_query, parameters, entities)

    try:
        provider = get_llm_provider()
        llm_with_tools = provider.bind_tools(_TOOLS, temperature=0.1)

        from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

        tool_map = {t.name: t for t in _TOOLS}
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Citizen Query: {user_query}\n\n"
                f"Complaint ID (if known): {complaint_id or 'Not provided'}\n"
                f"Tasks required: {', '.join(task_names)}\n\n"
                "Use the appropriate tool(s) to handle this complaint request."
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

        # Extract complaint data from tool call results in the message history
        complaint_data: dict = {}
        for msg in messages:
            if hasattr(msg, "content") and msg.content:
                try:
                    parsed = json.loads(str(msg.content))
                    if isinstance(parsed, dict) and parsed.get("success"):
                        raw = parsed.get("data", parsed)
                        if "complaint_id" in raw:
                            complaint_data = raw
                            break
                except (json.JSONDecodeError, TypeError):
                    pass

        if not complaint_data and complaint_id:
            complaint_data = check_complaint_status.invoke(
                {"complaint_id": complaint_id}
            ).get("data", {})

        cid = complaint_data.get("complaint_id", complaint_id)
        result = ComplaintResult(
            complaint_id=cid,
            status=complaint_data.get("status"),
            created_at=complaint_data.get("created_at"),
            escalation_level=complaint_data.get("escalation_level"),
            message=final_content,
            success=True,
        )

        log.complete(TaskStatus.SUCCESS)
        duration_ms = (datetime.utcnow() - t_start).total_seconds() * 1000
        is_new = any(n in task_names for n in ("create_complaint", "complaint_create", "create_new_complaint"))
        summary = f"{'Created' if is_new else 'Checked'}: {cid or 'Unknown'} — {result.status or 'Registered'}"

        return {
            "complaint_result": result,
            "agent_logs": [log],
            "execution_timeline": [{
                "agent": "complaint_agent", "display_name": "Complaint Agent",
                "status": "completed",
                "start_time": t_start.isoformat(),
                "end_time": datetime.utcnow().isoformat(),
                "duration_ms": round(duration_ms, 1),
                "summary": summary, "is_parallel": True,
            }],
        }

    except Exception as exc:
        logger.error("complaint_agent.error", error=str(exc))
        log.complete(TaskStatus.FAILED, error=str(exc))
        duration_ms = (datetime.utcnow() - t_start).total_seconds() * 1000
        return {
            "complaint_result": ComplaintResult(
                message="Unable to process complaint at this time.",
                success=False, error=str(exc),
            ),
            "agent_logs": [log],
            "execution_timeline": [{
                "agent": "complaint_agent", "display_name": "Complaint Agent",
                "status": "failed", "duration_ms": round(duration_ms, 1),
                "summary": f"Error: {str(exc)[:80]}", "is_parallel": True,
            }],
        }
