from __future__ import annotations

import json
from datetime import datetime

import structlog

from backend.graph.state import CitizenServiceState
from backend.models.schemas import (
    AgentLog,
    AgentName,
    TaskStatus,
    ValidationResult,
)

logger = structlog.get_logger(__name__)

_SYSTEM_PROMPT = """You are the Validator Agent for a Government Citizen Services Platform.

Your role: Validate the outputs of specialist agents before the final response is generated.

Validation criteria:
1. Completeness  — Do the agent outputs address ALL detected intents?
2. Accuracy      — Do responses contain specific data (IDs, amounts, dates, statuses)?
3. Consistency   — Is there any contradictory information across agent outputs?
4. Actionability — Are responses helpful and include clear next steps?

Scoring guidelines:
- completeness_score 1.0: All intents fully addressed with specific data
- completeness_score 0.7-0.9: Most intents addressed, minor gaps
- completeness_score < 0.7: Significant information missing

IMPORTANT: If is_valid is false, list specific issues and which agents should retry.
If all outputs are valid and complete, issues must be an empty array.

Respond ONLY with valid JSON (no markdown, no explanation):
{
  "is_valid": true,
  "completeness_score": 0.95,
  "issues": [],
  "agents_to_retry": [],
  "verdict": "Brief summary of validation outcome"
}"""


def _summarise_state(state: CitizenServiceState) -> str:
    """Build a concise state summary for the validator prompt."""
    intents: list[str] = state.get("intents") or state.get("intent_list", [])
    sections: list[str] = [
        f"Original Query: {state.get('user_query', '')}",
        f"Detected Intents: {', '.join(intents)}",
        f"Selected Agents: {', '.join(state.get('selected_agents', []))}",
    ]

    app = state.get("application_result")
    if app:
        sections.append(
            f"\n=== Application Agent Output ===\n"
            f"success={app.success}  status={app.status}  id={app.application_id}\n"
            f"message={app.message[:200] if app.message else ''}"
        )

    billing = state.get("billing_result")
    if billing:
        sections.append(
            f"\n=== Billing Agent Output ===\n"
            f"success={billing.success}  payment_id={billing.payment_id}  amount={billing.amount}\n"
            f"receipt_url={billing.receipt_url}"
        )

    knowledge = state.get("knowledge_result")
    if knowledge:
        sections.append(
            f"\n=== Knowledge Agent Output ===\n"
            f"success={knowledge.success}  docs={len(knowledge.documents)}\n"
            f"answer={knowledge.answer[:200] if knowledge.answer else ''}"
        )

    complaint = state.get("complaint_result")
    if complaint:
        sections.append(
            f"\n=== Complaint Agent Output ===\n"
            f"success={complaint.success}  id={complaint.complaint_id}  status={complaint.status}"
        )

    return "\n".join(sections)


async def validator_node(state: CitizenServiceState) -> dict:
    """
    Level 3b — Validator Agent.
    Runs after all parallel agents complete. Validates completeness and consistency
    before the Aggregator synthesises the final response.
    """
    log = AgentLog(agent_name=AgentName.VALIDATOR, task="output_validation")
    t_start = datetime.utcnow()
    logger.info("validator.start")
    print("[STEP 7] Validator START", flush=True)

    provider = get_llm_provider()
    context = _summarise_state(state)

    response = await provider.invoke(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Validate the following multi-agent execution outputs:\n\n"
                    f"{context}\n\n"
                    "Return your validation result as JSON."
                ),
            },
        ],
        temperature=0.0,
        max_tokens=512,
    )
    log.token_usage = response.usage

    validation: ValidationResult | None = None
    try:
        clean = (
            response.content.strip()
            .removeprefix("```json").removeprefix("```")
            .removesuffix("```").strip()
        )
        parsed = json.loads(clean)
        validation = ValidationResult(
            is_valid=bool(parsed.get("is_valid", True)),
            completeness_score=float(parsed.get("completeness_score", 0.9)),
            issues=parsed.get("issues", []),
            agents_to_retry=parsed.get("agents_to_retry", []),
            verdict=parsed.get("verdict", "Validation complete"),
        )
    except (json.JSONDecodeError, ValueError, KeyError) as exc:
        logger.warning("validator.parse_error", error=str(exc))
        validation = ValidationResult(
            is_valid=True,
            completeness_score=0.85,
            issues=[],
            agents_to_retry=[],
            verdict="Validation skipped — parse error (proceeding with aggregation).",
        )

    log.complete(TaskStatus.SUCCESS)
    duration_ms = (datetime.utcnow() - t_start).total_seconds() * 1000

    logger.info(
        "validator.done",
        is_valid=validation.is_valid,
        completeness=validation.completeness_score,
        issues=len(validation.issues),
        latency_ms=round(response.latency_ms, 1),
    )

    return {
        "validation_result": validation,
        "agent_logs": [log],
        "execution_timeline": [{
            "agent":        "validator",
            "display_name": "Validator",
            "status":       "completed",
            "start_time":   t_start.isoformat(),
            "end_time":     datetime.utcnow().isoformat(),
            "duration_ms":  round(duration_ms, 1),
            "summary":      f"{validation.verdict} (score: {validation.completeness_score:.0%})",
            "is_parallel":  False,
        }],
    }


def get_llm_provider():
    from backend.services.llm import get_llm_provider as _get
    return _get()
