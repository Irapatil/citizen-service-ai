from __future__ import annotations
from datetime import datetime, timezone
import structlog
from backend.graph.state import CitizenServiceState
from backend.models.schemas import (
    AgentLog,
    AgentName,
    ExecutionMetrics,
    TaskStatus,
)

logger = structlog.get_logger(__name__)

_SYSTEM_PROMPT = """You are the Response Aggregator Agent for a Government Citizen Services Platform.
Your role: Synthesise the outputs from all specialist agents into a single, clear, citizen-friendly response.

Guidelines:
- Write in plain, accessible language (no government jargon).
- Organise logically: most urgent/actionable information first.
- Remove redundancy — do not repeat the same fact twice.
- Include specific IDs, dates, amounts, and URLs where available.
- End with a clear "Next Steps" section.
- Maintain a warm, professional, empathetic tone.
- Use markdown formatting (bold for key info, tables for structured data, headers for sections).
- Keep the response complete but concise."""


def _build_context(state: CitizenServiceState) -> str:
    sections: list[str] = [
        f"Original Query: {state.get('user_query', '')}",
        f"Detected Intents: {', '.join(state.get('intents') or state.get('intent_list', []))}",
    ]

    app = state.get("application_result")
    if app and app.success:
        sections.append(
            f"\n=== Application Information ===\n"
            f"Application ID: {app.application_id}\n"
            f"Status: {app.status}\n"
            f"Submitted: {app.submitted_date}\n"
            f"Last Updated: {app.last_updated}\n"
            f"Agent Response: {app.message}"
        )
        if app.history:
            timeline = "\n".join(
                f"  - {h.get('event')} ({h.get('date')}): {h.get('description')}"
                for h in app.history[-4:]
            )
            sections.append(f"Recent Timeline:\n{timeline}")

    billing = state.get("billing_result")
    if billing and billing.success:
        sections.append(
            f"\n=== Billing Information ===\n"
            f"Payment ID: {billing.payment_id}\n"
            f"Amount: ${billing.amount}\n"
            f"Payment Date: {billing.payment_date}\n"
            f"Invoice: {billing.invoice_number}\n"
            f"Receipt URL: {billing.receipt_url or 'Not generated'}\n"
            f"Agent Response: {billing.message}"
        )

    knowledge = state.get("knowledge_result")
    if knowledge and knowledge.success:
        sections.append(
            f"\n=== Knowledge Base Information ===\n"
            f"Sources: {', '.join(knowledge.sources)}\n"
            f"Agent Response: {knowledge.answer}"
        )
        if knowledge.policies:
            pol_summary = "\n".join(
                f"  - {p.get('title', '')}: {p.get('summary', '')[:150]}"
                for p in knowledge.policies[:3]
            )
            sections.append(f"Relevant Policies:\n{pol_summary}")

    complaint = state.get("complaint_result")
    if complaint and complaint.success:
        sections.append(
            f"\n=== Complaint Information ===\n"
            f"Complaint ID: {complaint.complaint_id}\n"
            f"Status: {complaint.status}\n"
            f"Escalation Level: {complaint.escalation_level}\n"
            f"Agent Response: {complaint.message}"
        )

    # Note any partial failures
    failed = []
    for name, result in [("Application", app), ("Billing", billing), ("Knowledge", knowledge), ("Complaint", complaint)]:
        if result and not result.success:
            failed.append(f"{name}: {result.error}")
    if failed:
        sections.append("\n=== Partial Failures ===\n" + "\n".join(failed))

    return "\n".join(sections)


def _compute_metrics(state: CitizenServiceState) -> ExecutionMetrics:
    logs = state.get("agent_logs", [])
    return ExecutionMetrics(
        total_agents_invoked=len(logs),
        successful_agents=sum(1 for l in logs if l.status == TaskStatus.SUCCESS),
        failed_agents=sum(1 for l in logs if l.status == TaskStatus.FAILED),
        total_latency_ms=round(sum(l.latency_ms or 0 for l in logs), 2),
        total_tokens=sum(l.token_usage.get("total_tokens", 0) for l in logs),
        parallel_groups=1 if state.get("requires_parallel") else 0,
    )


async def aggregator_node(state: CitizenServiceState) -> dict:
    """
    Level 4 — Response Aggregator Agent.
    Merges all validated agent outputs into a single citizen-friendly response.
    """
    log = AgentLog(agent_name=AgentName.AGGREGATOR, task="response_aggregation")
    t_start = datetime.now(timezone.utc)
    logger.info("aggregator.start")
    print("[STEP 7] Aggregator START", flush=True)

    provider = get_llm_provider()
    context = _build_context(state)

    response = await provider.invoke(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Synthesise the following information into a single, complete "
                    f"response for the citizen:\n\n{context}"
                ),
            },
        ],
        temperature=0.3,
        max_tokens=1800,
    )
    log.token_usage = response.usage
    log.complete(TaskStatus.SUCCESS)

    metrics = _compute_metrics(state)
    metrics.total_agents_invoked += 1
    metrics.successful_agents += 1
    metrics.total_tokens += response.usage.get("total_tokens", 0)

    duration_ms = (datetime.now(timezone.utc) - t_start).total_seconds() * 1000

    logger.info(
        "aggregator.done",
        provider=response.provider,
        response_length=len(response.content),
        latency_ms=round(response.latency_ms, 1),
    )

    return {
        "final_response": response.content,
        "agent_logs": [log],
        "execution_metrics": metrics,
        "execution_timeline": [{
            "agent": "aggregator", "display_name": "Aggregator",
            "status": "completed",
            "start_time": t_start.isoformat(),
            "end_time": datetime.now(timezone.utc).isoformat(),
            "duration_ms": round(duration_ms, 1),
            "summary": f"Response synthesised ({len(response.content)} chars)",
            "is_parallel": False,
        }],
    }


def get_llm_provider():
    from backend.services.llm import get_llm_provider as _get
    return _get()
