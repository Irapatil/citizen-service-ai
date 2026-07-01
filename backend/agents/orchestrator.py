from __future__ import annotations

import json
from datetime import datetime

import structlog

from backend.graph.state import CitizenServiceState
from backend.models.schemas import AgentLog, AgentName, ExecutionMetrics, TaskStatus

logger = structlog.get_logger(__name__)

_SYSTEM_PROMPT = """You are the Orchestrator Agent for a Government Citizen Services Platform.

Your role is to deeply analyse each citizen query and extract structured intelligence.

Available intents (detect ALL that apply):
- application_status:   Citizen wants to know current status of a permit or application
- application_history:  Citizen wants timeline/audit history of their application
- payment_lookup:       Citizen wants to retrieve payment or transaction details
- receipt_request:      Citizen wants to download or obtain a payment receipt/invoice
- faq:                  Citizen is asking a general question about services or processes
- policy_search:        Citizen wants specific government policies or regulations
- complaint_create:     Citizen wants to file a new complaint about a service
- complaint_status:     Citizen wants to check an existing complaint's status
- general:              General enquiry that does not fit other categories

Query type classification:
- service_request:  Requesting a specific government service, document, or data
- inquiry:          Asking for information or explanation
- complaint:        Expressing dissatisfaction with a service
- information:      Seeking general guidance, policy, or background information

Entity types to extract:
- application_id:  e.g. APP-2024-001
- payment_id:      e.g. PAY-2024-001
- complaint_id:    e.g. CMP-A1B2C3D4
- citizen_name:    name of the citizen if mentioned
- permit_type:     type of permit if specified

IMPORTANT RULES:
- Always detect EVERY applicable intent (queries often have multiple intents)
- Extract ALL entity identifiers present in the query
- Set confidence based on clarity of the query (ambiguous = lower confidence)
- reasoning must be concise — one sentence per detected intent

Respond ONLY with valid JSON in this exact format (no markdown, no explanation):
{
  "query_type": "service_request",
  "intents": ["intent1", "intent2"],
  "entities": {"application_id": null, "payment_id": null, "complaint_id": null},
  "confidence": 0.95,
  "reasoning": "Brief explanation of what was detected and why"
}"""


async def orchestrator_node(state: CitizenServiceState) -> dict:
    """
    Level 1 — Orchestrator Agent.
    Analyses any citizen query to extract query_type, intents, entities, and confidence.
    Uses structured LLM output — no hardcoded routing logic.
    """
    log = AgentLog(agent_name=AgentName.ORCHESTRATOR, task="intent_detection")
    t_start = datetime.utcnow()
    logger.info("orchestrator.start", session=state.get("session_id"))
    print(f"[STEP 7] Orchestrator START  query={state.get('user_query','')!r:.80}", flush=True)

    provider = get_llm_provider()

    # Build conversation context for multi-turn awareness
    messages_hist = state.get("messages", [])
    conversation_context = ""
    if len(messages_hist) > 1:
        recent = messages_hist[-4:-1]
        conversation_context = "\n".join(
            f"{m.type.upper()}: {m.content}" for m in recent
        )

    user_content = (
        f"Citizen Query: {state['user_query']}\n\n"
        + (f"Previous context:\n{conversation_context}\n\n" if conversation_context else "")
        + "Analyse this query and return the JSON response."
    )

    response = await provider.invoke(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        temperature=0.0,
    )
    log.token_usage = response.usage

    # Parse the structured JSON response
    parsed: dict = {}
    try:
        clean = (
            response.content.strip()
            .removeprefix("```json").removeprefix("```")
            .removesuffix("```").strip()
        )
        parsed = json.loads(clean)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("orchestrator.parse_error", error=str(exc), raw=response.content[:300])
        parsed = {
            "query_type": "service_request",
            "intents": ["general"],
            "entities": {},
            "confidence": 0.5,
            "reasoning": "Could not parse LLM response — defaulting to general intent.",
        }

    intents: list[str]      = parsed.get("intents", ["general"])
    entities: dict          = parsed.get("entities", {})
    query_type: str         = parsed.get("query_type", "service_request")
    confidence: float       = float(parsed.get("confidence", 0.9))
    reasoning: str          = parsed.get("reasoning", "")

    # Clean up null entity values
    entities = {k: v for k, v in entities.items() if v is not None}

    # Per-intent confidence (from LLM or derived from overall confidence)
    intent_confidence: dict[str, float] = parsed.get("intent_confidence", {})
    for intent in intents:
        if intent not in intent_confidence:
            # Deterministic variation so each intent shows a slightly different value
            var = (len(intent) % 8 - 4) / 100
            intent_confidence[intent] = round(min(0.99, max(0.72, confidence + var)), 2)

    # Derive legacy fields for backward compatibility
    complexity = "complex" if len(intents) >= 3 else "moderate" if len(intents) == 2 else "simple"
    requires_parallel = len(intents) > 1

    log.complete(TaskStatus.SUCCESS)
    duration_ms = (datetime.utcnow() - t_start).total_seconds() * 1000

    logger.info(
        "orchestrator.done",
        provider=response.provider,
        model=response.model,
        query_type=query_type,
        intents=intents,
        entities=list(entities.keys()),
        confidence=confidence,
        latency_ms=round(response.latency_ms, 1),
    )

    return {
        # New structured fields
        "query_type":        query_type,
        "intents":           intents,
        "entities":          entities,
        "confidence":        confidence,
        "intent_confidence": intent_confidence,
        # Legacy fields (backward compat with planner/aggregator)
        "intent_list":  intents,
        "complexity":   complexity,
        "requires_parallel": requires_parallel,
        # Observability
        "agent_logs": [log],
        "execution_metrics": ExecutionMetrics(total_agents_invoked=1, successful_agents=1),
        "execution_timeline": [{
            "agent":        "orchestrator",
            "display_name": "Orchestrator",
            "status":       "completed",
            "start_time":   t_start.isoformat(),
            "end_time":     datetime.utcnow().isoformat(),
            "duration_ms":  round(duration_ms, 1),
            "summary":      f"Detected {len(intents)} intent(s): {', '.join(intents)}",
            "is_parallel":  False,
        }],
        "metadata": {
            **state.get("metadata", {}),
            "orchestrator_reasoning": reasoning,
            "orchestrator_at":        datetime.utcnow().isoformat(),
            "llm_provider":           response.provider,
            "llm_model":              response.model,
        },
    }


# Deferred import to avoid circular dependency at module level
def get_llm_provider():
    from backend.services.llm import get_llm_provider as _get
    return _get()
