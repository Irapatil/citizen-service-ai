from __future__ import annotations

import json
import re
from datetime import datetime

import structlog

from backend.graph.state import CitizenServiceState
from backend.models.schemas import (
    AgentLog,
    AgentName,
    ApplicationResult,
    TaskStatus,
)
from backend.services.llm import get_llm_provider
from backend.tools.application_tools import get_application_status, get_application_history

logger = structlog.get_logger(__name__)

_TOOLS = [get_application_status, get_application_history]

_SYSTEM_PROMPT = """You are the Application Agent for a Government Citizen Services Platform.

Your role: Help citizens with permit and application enquiries.
Use the available tools to fetch real data, then compose a clear, empathetic response.

Available tools:
- get_application_status(application_id):  Get current status, assigned officer, and estimated completion.
- get_application_history(application_id): Get the full timeline/audit history of an application.

If no application ID is provided, use "APP-2024-001" as a demonstration default.
Always provide specific IDs, dates, and actionable next steps."""


def _extract_app_id(query: str, parameters: dict, entities: dict) -> str:
    # Priority: explicit task parameters > orchestrator entities > regex > default
    if parameters.get("application_id"):
        return parameters["application_id"]
    if entities.get("application_id"):
        return entities["application_id"]
    match = re.search(r"APP[-\s]?\d{4}[-\s]?\d{3,}", query, re.IGNORECASE)
    if match:
        return match.group().replace(" ", "-").upper()
    return "APP-2024-001"


async def application_agent_node(state: CitizenServiceState) -> dict:
    """
    Level 3 — Application Agent.
    Handles permit status and application history queries using tool calling.
    """
    log = AgentLog(agent_name=AgentName.APPLICATION, task="application_query")
    t_start = datetime.utcnow()
    logger.info("application_agent.start")
    print("[STEP 7] Application Agent START", flush=True)

    plan = state.get("execution_plan")
    user_query = state.get("user_query", "")
    entities = state.get("entities", {})

    app_tasks = [t for t in plan.tasks if t.agent == AgentName.APPLICATION] if plan else []
    if not app_tasks:
        log.complete(TaskStatus.SKIPPED)
        return {
            "agent_logs": [log],
            "execution_timeline": [{
                "agent": "application_agent", "display_name": "Application Agent",
                "status": "skipped", "duration_ms": 0, "summary": "Not selected for this query",
                "is_parallel": True,
            }],
        }

    parameters = app_tasks[0].parameters if app_tasks else {}
    task_names = {t.task for t in app_tasks}
    app_id = _extract_app_id(user_query, parameters, entities)

    try:
        provider = get_llm_provider()
        llm_with_tools = provider.bind_tools(_TOOLS)

        from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

        tool_map = {t.name: t for t in _TOOLS}
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Citizen Query: {user_query}\n\n"
                f"Application ID: {app_id}\n"
                f"Tasks required: {', '.join(task_names)}\n\n"
                "Use the available tools to retrieve information and provide a comprehensive response."
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

        # Always fetch the structured data directly for the result object
        status_data = get_application_status.invoke({"application_id": app_id})
        history_data = (
            get_application_history.invoke({"application_id": app_id})
            if any(n in task_names for n in ("get_application_history", "application_history", "retrieve_application_history"))
            else {"history": []}
        )

        app_data = status_data.get("data", {})
        result = ApplicationResult(
            application_id=app_data.get("application_id"),
            status=app_data.get("status"),
            submitted_date=app_data.get("submitted_date"),
            last_updated=app_data.get("last_updated"),
            history=history_data.get("history", []),
            message=final_content,
            success=True,
        )

        log.complete(TaskStatus.SUCCESS)
        duration_ms = (datetime.utcnow() - t_start).total_seconds() * 1000
        logger.info("application_agent.done", status=result.status)

        return {
            "application_result": result,
            "agent_logs": [log],
            "execution_timeline": [{
                "agent": "application_agent", "display_name": "Application Agent",
                "status": "completed",
                "start_time": t_start.isoformat(),
                "end_time": datetime.utcnow().isoformat(),
                "duration_ms": round(duration_ms, 1),
                "summary": f"Status: {result.status} ({app_id})",
                "is_parallel": True,
            }],
        }

    except Exception as exc:
        logger.error("application_agent.error", error=str(exc))
        log.complete(TaskStatus.FAILED, error=str(exc))
        duration_ms = (datetime.utcnow() - t_start).total_seconds() * 1000
        return {
            "application_result": ApplicationResult(
                message="Unable to retrieve application details at this time.",
                success=False, error=str(exc),
            ),
            "agent_logs": [log],
            "execution_timeline": [{
                "agent": "application_agent", "display_name": "Application Agent",
                "status": "failed", "duration_ms": round(duration_ms, 1),
                "summary": f"Error: {str(exc)[:80]}", "is_parallel": True,
            }],
        }
