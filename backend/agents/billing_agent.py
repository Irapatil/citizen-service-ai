from __future__ import annotations

import json
import re
from datetime import datetime

import structlog

from backend.graph.state import CitizenServiceState
from backend.models.schemas import (
    AgentLog,
    AgentName,
    BillingResult,
    TaskStatus,
)
from backend.services.llm import get_llm_provider
from backend.tools.billing_tools import get_payment_details, generate_receipt_pdf

logger = structlog.get_logger(__name__)

_TOOLS = [get_payment_details, generate_receipt_pdf]

_SYSTEM_PROMPT = """You are the Billing Agent for a Government Citizen Services Platform.

Your role: Help citizens with payment lookups, receipts, and invoices.
Use the available tools to retrieve accurate, specific payment information.

Available tools:
- get_payment_details(payment_id, application_id): Look up payment records by ID.
- generate_receipt_pdf(payment_id): Generate a downloadable receipt PDF link.

Always provide specific payment amounts, dates, invoice numbers, and download instructions."""


def _extract_ids(query: str, parameters: dict, entities: dict) -> tuple[str | None, str | None]:
    pay_id = parameters.get("payment_id") or entities.get("payment_id")
    app_id = parameters.get("application_id") or entities.get("application_id")
    if not pay_id:
        m = re.search(r"PAY[-\s]?\d{4}[-\s]?\w+", query, re.IGNORECASE)
        if m:
            pay_id = m.group().replace(" ", "-").upper()
    if not app_id:
        m = re.search(r"APP[-\s]?\d{4}[-\s]?\d{3,}", query, re.IGNORECASE)
        if m:
            app_id = m.group().replace(" ", "-").upper()
    return pay_id, app_id


async def billing_agent_node(state: CitizenServiceState) -> dict:
    """
    Level 3 — Billing Agent.
    Handles payment lookups and receipt generation using tool calling.
    """
    log = AgentLog(agent_name=AgentName.BILLING, task="billing_query")
    t_start = datetime.utcnow()
    logger.info("billing_agent.start")
    print("[STEP 7] Billing Agent START", flush=True)

    plan = state.get("execution_plan")
    user_query = state.get("user_query", "")
    entities = state.get("entities", {})

    billing_tasks = [t for t in plan.tasks if t.agent == AgentName.BILLING] if plan else []
    if not billing_tasks:
        log.complete(TaskStatus.SKIPPED)
        return {
            "agent_logs": [log],
            "execution_timeline": [{
                "agent": "billing_agent", "display_name": "Billing Agent",
                "status": "skipped", "duration_ms": 0, "summary": "Not selected for this query",
                "is_parallel": True,
            }],
        }

    parameters = billing_tasks[0].parameters if billing_tasks else {}
    task_names = {t.task for t in billing_tasks}
    pay_id, app_id = _extract_ids(user_query, parameters, entities)

    try:
        provider = get_llm_provider()
        llm_with_tools = provider.bind_tools(_TOOLS)

        from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

        tool_map = {t.name: t for t in _TOOLS}
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Citizen Query: {user_query}\n\n"
                f"Payment ID: {pay_id or 'Not provided'}\n"
                f"Application ID: {app_id or 'Not provided'}\n"
                f"Tasks required: {', '.join(task_names)}\n\n"
                "Use tools to retrieve payment info and generate receipt if requested."
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

        pay_data = get_payment_details.invoke(
            {"payment_id": pay_id, "application_id": app_id or "APP-2024-001"}
        ).get("data", {})

        resolved_pay_id = pay_data.get("payment_id", pay_id)
        receipt_url = None
        if any(n in task_names for n in ("generate_receipt_pdf", "receipt_request", "generate_payment_receipt", "generate_receipt")):
            receipt_resp = generate_receipt_pdf.invoke({"payment_id": resolved_pay_id or "PAY-2024-001"})
            receipt_url = receipt_resp.get("receipt_url")

        result = BillingResult(
            payment_id=resolved_pay_id,
            amount=pay_data.get("amount"),
            payment_date=pay_data.get("payment_date"),
            receipt_url=receipt_url,
            invoice_number=pay_data.get("invoice_number"),
            message=final_content,
            success=True,
        )

        log.complete(TaskStatus.SUCCESS)
        duration_ms = (datetime.utcnow() - t_start).total_seconds() * 1000
        logger.info("billing_agent.done", payment_id=result.payment_id, amount=result.amount)

        summary = f"Payment: ${result.amount} ({result.payment_id})"
        if receipt_url:
            summary += " + Receipt generated"

        return {
            "billing_result": result,
            "agent_logs": [log],
            "execution_timeline": [{
                "agent": "billing_agent", "display_name": "Billing Agent",
                "status": "completed",
                "start_time": t_start.isoformat(),
                "end_time": datetime.utcnow().isoformat(),
                "duration_ms": round(duration_ms, 1),
                "summary": summary,
                "is_parallel": True,
            }],
        }

    except Exception as exc:
        logger.error("billing_agent.error", error=str(exc))
        log.complete(TaskStatus.FAILED, error=str(exc))
        duration_ms = (datetime.utcnow() - t_start).total_seconds() * 1000
        return {
            "billing_result": BillingResult(
                message="Unable to retrieve billing information at this time.",
                success=False, error=str(exc),
            ),
            "agent_logs": [log],
            "execution_timeline": [{
                "agent": "billing_agent", "display_name": "Billing Agent",
                "status": "failed", "duration_ms": round(duration_ms, 1),
                "summary": f"Error: {str(exc)[:80]}", "is_parallel": True,
            }],
        }
