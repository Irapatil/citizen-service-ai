"""
Mock LLM Provider — runs the full multi-agent workflow with zero API calls.

Dynamic version: analyses the actual query for ANY input rather than returning
hardcoded permit-renewal answers. Selects agents, builds tasks, and synthesises
responses based on the real context passed to each agent.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, AsyncGenerator, List, Type

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel, Field

from backend.services.llm.base_provider import BaseLLMProvider, LLMResponse

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Intent detection helpers
# ---------------------------------------------------------------------------

_INTENT_KEYWORDS: dict[str, list[str]] = {
    "application_status": [
        "status", "update", "where is", "how is", "application", "permit",
        "renewal", "submitted", "progress", "track", "pending", "approved",
    ],
    "application_history": [
        "history", "timeline", "audit", "log", "events", "what happened",
        "previous", "when was",
    ],
    "payment_lookup": [
        "payment", "paid", "transaction", "charged", "amount", "billing",
        "how much", "invoice",
    ],
    "receipt_request": [
        "receipt", "download receipt", "copy of receipt", "proof of payment",
        "tax invoice", "receipt pdf",
    ],
    "faq": [
        "how do i", "how can i", "what is", "what are", "explain",
        "tell me about", "information about", "operating hours", "contact",
        "when do", "how long",
    ],
    "policy_search": [
        "policy", "regulation", "rule", "requirement", "procedure",
        "guideline", "law", "legislation",
    ],
    "complaint_create": [
        "complaint", "complain", "lodge", "file a complaint", "unhappy",
        "frustrated", "not acceptable", "delay", "problem", "issue",
        "dissatisfied",
    ],
    "complaint_status": [
        "complaint status", "complaint update", "my complaint", "cmp-",
        "what happened to my complaint",
    ],
}

_INTENT_TO_AGENT: dict[str, str] = {
    "application_status": "application_agent",
    "application_history": "application_agent",
    "payment_lookup": "billing_agent",
    "receipt_request": "billing_agent",
    "faq": "knowledge_agent",
    "policy_search": "knowledge_agent",
    "complaint_create": "complaint_agent",
    "complaint_status": "complaint_agent",
    "general": "knowledge_agent",
}

_INTENT_TO_TASK: dict[str, str] = {
    "application_status": "check_application_status",
    "application_history": "retrieve_application_history",
    "payment_lookup": "get_payment_details",
    "receipt_request": "generate_receipt_pdf",
    "faq": "search_faq",
    "policy_search": "search_policy_documents",
    "complaint_create": "create_complaint",
    "complaint_status": "check_complaint_status",
    "general": "retrieve_documents",
}


def _detect_intents(query: str) -> tuple[list[str], dict[str, Any], float]:
    """Returns (intents, entities, confidence)."""
    q = query.lower()
    found: list[str] = []

    # Test complaint_status before complaint_create (more specific)
    for intent in ["complaint_status", "complaint_create", "application_history",
                   "application_status", "receipt_request", "payment_lookup",
                   "policy_search", "faq"]:
        keywords = _INTENT_KEYWORDS[intent]
        if any(kw in q for kw in keywords):
            found.append(intent)

    # Deduplicate agents — keep only the most specific intent per agent
    agent_seen: set[str] = set()
    deduped: list[str] = []
    for intent in found:
        agent = _INTENT_TO_AGENT.get(intent, "knowledge_agent")
        if agent not in agent_seen:
            deduped.append(intent)
            agent_seen.add(agent)

    if not deduped:
        deduped = ["general"]

    # Entity extraction
    entities: dict[str, Any] = {}
    app_m = re.search(r"APP[-\s]?\d{4}[-\s]?\d+", query, re.IGNORECASE)
    if app_m:
        entities["application_id"] = app_m.group().replace(" ", "-").upper()
    pay_m = re.search(r"PAY[-\s]?\d{4}[-\s]?\w+", query, re.IGNORECASE)
    if pay_m:
        entities["payment_id"] = pay_m.group().replace(" ", "-").upper()
    cmp_m = re.search(r"CMP[-\s]?\w{6,}", query, re.IGNORECASE)
    if cmp_m:
        entities["complaint_id"] = cmp_m.group().replace(" ", "-").upper()

    confidence = 0.94 if len(deduped) > 0 and deduped != ["general"] else 0.72
    return deduped, entities, confidence


def _build_orchestrator_response(query: str) -> str:
    intents, entities, confidence = _detect_intents(query)

    if len(intents) > 1:
        query_type = "multi_intent"
    elif intents[0] in ("application_status", "application_history"):
        query_type = "service_request"
    elif intents[0] in ("payment_lookup", "receipt_request"):
        query_type = "service_request"
    elif intents[0] in ("complaint_create", "complaint_status"):
        query_type = "complaint"
    elif intents[0] in ("faq", "policy_search", "general"):
        query_type = "inquiry"
    else:
        query_type = "service_request"

    agent_descs = {
        "application_status": "application status and renewal tracking",
        "application_history": "application audit history",
        "payment_lookup": "payment transaction lookup",
        "receipt_request": "payment receipt download",
        "faq": "general service enquiry",
        "policy_search": "government policy lookup",
        "complaint_create": "new complaint submission",
        "complaint_status": "existing complaint status",
        "general": "general knowledge retrieval",
    }
    reasoning = (
        f"Detected {len(intents)} intent(s): "
        + ", ".join(f"'{i}' ({agent_descs.get(i, i)})" for i in intents)
        + "."
    )
    if len(intents) > 1:
        reasoning += " Multiple independent intents identified — parallel execution recommended."

    # Per-intent confidence (slight deterministic variation per intent name)
    intent_confidence = {}
    for intent in intents:
        var = (len(intent) % 8 - 4) / 100
        intent_confidence[intent] = round(min(0.98, max(0.76, confidence + var)), 2)

    return json.dumps({
        "query_type": query_type,
        "intents": intents,
        "intent_confidence": intent_confidence,
        "entities": entities,
        "confidence": confidence,
        "reasoning": reasoning,
        "intent_list": intents,
        "complexity": "moderate" if len(intents) > 1 else "simple",
        "requires_parallel": len(intents) > 1,
    })


def _build_planner_response(intents: list[str], entities: dict) -> str:
    app_id = entities.get("application_id", "APP-2024-001")
    pay_id = entities.get("payment_id")
    cmp_id = entities.get("complaint_id")

    selected_agents: list[str] = []
    tasks: list[dict] = []

    for intent in intents:
        agent = _INTENT_TO_AGENT.get(intent, "knowledge_agent")
        task_name = _INTENT_TO_TASK.get(intent, "retrieve_documents")

        if agent not in selected_agents:
            selected_agents.append(agent)

        params: dict[str, Any] = {}
        if agent == "application_agent":
            params["application_id"] = app_id
        elif agent == "billing_agent":
            if pay_id:
                params["payment_id"] = pay_id
            params["application_id"] = app_id
        elif agent == "complaint_agent":
            if cmp_id:
                params["complaint_id"] = cmp_id

        tasks.append({"agent": agent, "task": task_name, "parameters": params, "priority": 1})

    execution_mode = "parallel" if len(selected_agents) > 1 else "single"
    reasoning = (
        f"Selected {len(selected_agents)} agent(s): {', '.join(selected_agents)}. "
        f"Execution mode: {execution_mode}."
    )

    # Derive skipped agents and explanations
    _all_specialists = ["application_agent", "billing_agent", "knowledge_agent", "complaint_agent"]
    skipped_agents = [a for a in _all_specialists if a not in selected_agents]

    _intent_reason = {
        "application_status": "permit/application status information",
        "application_history": "application audit history and timeline",
        "payment_lookup": "payment transaction details",
        "receipt_request": "a payment receipt or invoice download",
        "faq": "general government service information",
        "policy_search": "government policy and procedure information",
        "complaint_create": "filing a new service complaint",
        "complaint_status": "existing complaint status information",
        "general": "general information lookup",
    }
    _agent_intent_map = {
        "application_agent": ["application_status", "application_history"],
        "billing_agent":     ["payment_lookup", "receipt_request"],
        "knowledge_agent":   ["faq", "policy_search", "general"],
        "complaint_agent":   ["complaint_create", "complaint_status"],
    }

    agent_explanations: dict[str, str] = {}
    for agent in selected_agents:
        matched = [i for i in intents if i in _agent_intent_map.get(agent, [])]
        if matched:
            reasons = " and ".join(_intent_reason.get(i, i.replace("_", " ")) for i in matched)
            agent_explanations[agent] = f"Selected because the query requires {reasons}."
        else:
            agent_explanations[agent] = "Selected to handle this aspect of the request."
    for agent in skipped_agents:
        capable = _agent_intent_map.get(agent, [])
        first = capable[0].replace("_", " ") if capable else "this type of request"
        agent_explanations[agent] = f"Skipped — no '{first}' intent was detected in this query."

    return json.dumps({
        "selected_agents":   selected_agents,
        "skipped_agents":    skipped_agents,
        "agent_explanations": agent_explanations,
        "tasks": tasks,
        "execution_mode": execution_mode,
        "reasoning": reasoning,
        "execution_plan": {
            "tasks": tasks,
            "parallel_groups": [selected_agents] if execution_mode == "parallel" else [[a] for a in selected_agents],
        },
    })


def _build_validator_response(context: str) -> str:
    has_app = "Application Information" in context or "application_id" in context.lower()
    has_billing = "Billing Information" in context or "payment_id" in context.lower()
    has_knowledge = "Knowledge Base" in context or "documents" in context.lower()
    has_complaint = "Complaint Information" in context or "complaint_id" in context.lower()

    active_count = sum([has_app, has_billing, has_knowledge, has_complaint])
    score = min(0.95, 0.7 + active_count * 0.08)

    return json.dumps({
        "is_valid": True,
        "completeness_score": round(score, 2),
        "issues": [],
        "agents_to_retry": [],
        "verdict": f"All {active_count} active agent(s) returned valid responses. Ready for aggregation.",
    })


def _build_aggregator_response(context: str, query: str) -> str:
    """Generate a dynamic response by parsing the structured context."""
    sections: list[str] = []

    # Determine greeting based on query
    q_lower = query.lower() if query else ""

    # Extract and format application info
    if "Application Information" in context:
        m_app_id = re.search(r"Application ID:\s*(\S+)", context)
        m_status = re.search(r"Status:\s*(.+)", context)
        m_submitted = re.search(r"Submitted:\s*(.+)", context)
        m_updated = re.search(r"Last Updated:\s*(.+)", context)

        app_id = m_app_id.group(1) if m_app_id else "your application"
        status = m_status.group(1).strip() if m_status else "Under Review"
        submitted = m_submitted.group(1).strip() if m_submitted else "recently"
        updated = m_updated.group(1).strip() if m_updated else "recently"

        sections.append(
            f"## Application Status — {app_id}\n\n"
            f"Your application is currently **{status}**.\n\n"
            f"| Field | Details |\n|---|---|\n"
            f"| Application ID | {app_id} |\n"
            f"| Submitted | {submitted} |\n"
            f"| Last Updated | {updated} |\n"
            f"| Status | **{status}** |"
        )

    # Extract billing info
    if "Billing Information" in context:
        m_pay_id = re.search(r"Payment ID:\s*(\S+)", context)
        m_amount = re.search(r"Amount:\s*\$?([\d,.]+)", context)
        m_pay_date = re.search(r"Payment Date:\s*(.+)", context)
        m_invoice = re.search(r"Invoice:\s*(\S+)", context)
        m_receipt = re.search(r"Receipt URL:\s*(\S+)", context)

        pay_id = m_pay_id.group(1) if m_pay_id else "your payment"
        amount = m_amount.group(1) if m_amount else "N/A"
        pay_date = m_pay_date.group(1).strip() if m_pay_date else "on file"
        invoice = m_invoice.group(1) if m_invoice else "N/A"
        receipt_url = m_receipt.group(1) if m_receipt and m_receipt.group(1) != "Not" else None

        receipt_line = (
            f"| Receipt Download | [{receipt_url}]({receipt_url}) |"
            if receipt_url else
            "| Receipt | Available via Citizen Portal |\n"
            "| Portal | Citizen Portal → Payments → Download Receipt |"
        )

        sections.append(
            f"## Payment Receipt — {pay_id}\n\n"
            f"Your payment of **${amount}** was processed on {pay_date}.\n\n"
            f"| Field | Details |\n|---|---|\n"
            f"| Payment ID | {pay_id} |\n"
            f"| Amount | ${amount} |\n"
            f"| Invoice | {invoice} |\n"
            f"{receipt_line}"
        )

    # Extract knowledge info
    if "Knowledge Base" in context:
        m_sources = re.search(r"Sources:\s*(.+)", context)
        m_answer = re.search(r"Agent Response:\s*(.+?)(?:\n===|$)", context, re.DOTALL)
        sources = m_sources.group(1).strip() if m_sources else "Knowledge Base"
        answer = m_answer.group(1).strip()[:600] if m_answer else ""

        if answer:
            sections.append(f"## Information\n\n{answer}\n\n*Source: {sources}*")

    # Extract complaint info
    if "Complaint Information" in context:
        m_cmp_id = re.search(r"Complaint ID:\s*(\S+)", context)
        m_cmp_status = re.search(r"Status:\s*(\S+)", context)
        m_escalation = re.search(r"Escalation Level:\s*(\S+)", context)

        cmp_id = m_cmp_id.group(1) if m_cmp_id else "your complaint"
        cmp_status = m_cmp_status.group(1).strip() if m_cmp_status else "Registered"
        escalation = m_escalation.group(1) if m_escalation else "Level 1"

        sections.append(
            f"## Complaint — {cmp_id}\n\n"
            f"Your complaint has been registered and is **{cmp_status}** (Escalation: {escalation}).\n\n"
            "You will be contacted within **2 business days** to acknowledge your complaint."
        )

    # Generic fallback if no sections were populated
    if not sections:
        sections.append(
            "Your request has been processed. Our team will follow up with detailed information "
            "via your registered contact details within 2 business days."
        )

    # Next steps
    next_steps: list[str] = []
    if "Under Review" in context or "Pending" in context:
        next_steps.append("Monitor your application status via the Citizen Portal at portal.gov/track")
    if "Receipt URL" in context or "Billing Information" in context:
        next_steps.append("Download your payment receipt from the Citizen Portal → Payments")
    if "Complaint" in context:
        next_steps.append("Retain your Complaint ID for future reference")
    if not next_steps:
        next_steps.append("Contact us at **1800-CITIZEN** (Mon-Fri, 8 AM-5 PM) for further assistance")

    next_steps_md = "\n".join(f"{i+1}. {s}" for i, s in enumerate(next_steps))

    body = "\n\n---\n\n".join(sections)
    return (
        f"{body}\n\n---\n\n"
        f"## Next Steps\n\n{next_steps_md}\n\n"
        f"*Government Citizen Services — CitizenAI Enterprise Platform*"
    )


# ---------------------------------------------------------------------------
# Tool argument builder — picks sensible args per tool name + context
# ---------------------------------------------------------------------------

def _build_tool_args(tool_name: str, human_content: str) -> dict[str, Any]:
    """Extract IDs from the human message to pass as tool arguments."""
    app_m = re.search(r"APP[-\s]?\d{4}[-\s]?\d+", human_content, re.IGNORECASE)
    app_id = app_m.group().replace(" ", "-").upper() if app_m else "APP-2024-001"

    pay_m = re.search(r"PAY[-\s]?\d{4}[-\s]?\w+", human_content, re.IGNORECASE)
    pay_id = pay_m.group().replace(" ", "-").upper() if pay_m else "PAY-2024-001"

    cmp_m = re.search(r"CMP[-\s]?\w{6,}", human_content, re.IGNORECASE)
    cmp_id = cmp_m.group().replace(" ", "-").upper() if cmp_m else None

    # Derive a search query from the citizen query line
    q_m = re.search(r"Citizen Query:\s*(.+?)(?:\n|$)", human_content)
    query_text = q_m.group(1).strip() if q_m else human_content[:100]

    mapping: dict[str, dict[str, Any]] = {
        "get_application_status":  {"application_id": app_id},
        "get_application_history": {"application_id": app_id},
        "get_payment_details":     {"payment_id": pay_id, "application_id": app_id},
        "generate_receipt_pdf":    {"payment_id": pay_id},
        "retrieve_documents":      {"query": query_text, "top_k": 4},
        "search_policies":         {"topic": query_text},
        "check_complaint_status":  {"complaint_id": cmp_id or "CMP-001"},
        "create_complaint": {
            "subject": "Service Query",
            "description": query_text[:200],
            "category": "Service Issue",
            "priority": "Medium",
        },
    }
    return mapping.get(tool_name, {})


# ---------------------------------------------------------------------------
# Internal mock chat model
# ---------------------------------------------------------------------------

class _MockChatModel(BaseChatModel):
    """
    LangChain-compatible mock chat model.
    Calls the first relevant tool on the first invocation;
    returns a context-aware text response after tool results arrive.
    """

    mock_tools: List[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "mock-citizen-ai"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        has_tool_results = any(isinstance(m, ToolMessage) for m in messages)

        # Extract human message content for context
        human_content = ""
        for m in messages:
            if isinstance(m, HumanMessage):
                human_content = str(m.content)

        if not has_tool_results and self.mock_tools:
            # Pick the most relevant tool based on task names mentioned in the human message
            selected_tool = self._select_tool(human_content)
            args = _build_tool_args(selected_tool.name, human_content)
            ai_msg = AIMessage(
                content="",
                tool_calls=[{
                    "name": selected_tool.name,
                    "args": args,
                    "id": f"call_mock_{selected_tool.name}_001",
                    "type": "tool_call",
                }],
            )
        else:
            # Build a context-aware response from tool results
            tool_summary = self._summarise_tool_results(messages)
            ai_msg = AIMessage(content=tool_summary)

        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    def _select_tool(self, human_content: str) -> Any:
        """Pick the most relevant tool based on task names in the human message."""
        content_lower = human_content.lower()
        # Task name to preferred tool mapping
        priority: dict[str, str] = {
            "history": "get_application_history",
            "status": "get_application_status",
            "receipt": "generate_receipt_pdf",
            "payment": "get_payment_details",
            "complaint_status": "check_complaint_status",
            "complaint": "create_complaint",
            "policy": "search_policies",
            "document": "retrieve_documents",
            "faq": "retrieve_documents",
        }
        tool_map = {t.name: t for t in self.mock_tools}
        for keyword, preferred_name in priority.items():
            if keyword in content_lower and preferred_name in tool_map:
                return tool_map[preferred_name]
        return self.mock_tools[0]

    def _summarise_tool_results(self, messages) -> str:
        """Extract key data from tool results to build a coherent agent response."""
        results: list[str] = []
        for m in messages:
            if isinstance(m, ToolMessage):
                try:
                    data = json.loads(str(m.content))
                    if isinstance(data, dict):
                        inner = data.get("data", data)
                        if isinstance(inner, dict) and inner:
                            key_pairs = ", ".join(
                                f"{k}: {v}" for k, v in list(inner.items())[:5]
                                if not isinstance(v, (dict, list))
                            )
                            results.append(key_pairs)
                except (json.JSONDecodeError, TypeError):
                    pass
        if results:
            return "Successfully retrieved: " + "; ".join(results)
        return "The request has been processed successfully using the government services database."

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        await asyncio.sleep(0)
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def bind_tools(self, tools, **kwargs):
        return _MockChatModel(mock_tools=list(tools))


# ---------------------------------------------------------------------------
# Public mock provider
# ---------------------------------------------------------------------------

class MockLLMProvider(BaseLLMProvider):
    """
    Zero-API-call provider for demo/offline mode.
    Dynamically analyses the query to route, plan, validate, and aggregate.
    """

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "mock-dynamic-v2"

    def get_chat_model(self, temperature: float = 0.0, max_tokens: int = 4096,
                       streaming: bool = False, **kwargs) -> BaseChatModel:
        return _MockChatModel()

    async def invoke(self, messages: list[dict[str, str]], temperature: float = 0.0,
                     max_tokens: int = 4096) -> LLMResponse:
        t0 = time.monotonic()

        # Extract system and user content
        system_content = ""
        user_content = ""
        for m in messages:
            if not isinstance(m, dict):
                continue
            if m.get("role") == "system":
                system_content = m.get("content", "").lower()
            elif m.get("role") == "user":
                user_content = m.get("content", "")

        # Parse actual citizen query from user content
        query_match = re.search(r"(?:Citizen Query|Original Query|query):\s*(.+?)(?:\n|$)",
                                user_content, re.IGNORECASE)
        query = query_match.group(1).strip() if query_match else user_content[:200]

        # Detect call type from system prompt
        is_orchestrator = "available intents" in system_content and "query_type" in system_content
        is_planner = "selected_agents" in system_content or "execution_mode" in system_content
        is_validator = "validat" in system_content and ("completeness_score" in system_content or "is_valid" in system_content)
        is_aggregator = ("synthesise" in system_content or "synthesize" in system_content) and "citizen" in system_content

        if is_orchestrator:
            content = _build_orchestrator_response(query)
            logger.info("mock_provider.orchestrator", query_preview=query[:60])

        elif is_planner:
            # Re-detect intents from the orchestrator output in context
            intents_m = re.search(r'"intents":\s*(\[.+?\])', user_content)
            entities_m = re.search(r'"entities":\s*(\{.+?\})', user_content)
            try:
                intents = json.loads(intents_m.group(1)) if intents_m else []
                entities = json.loads(entities_m.group(1)) if entities_m else {}
            except (json.JSONDecodeError, AttributeError):
                intents, entities, _ = _detect_intents(query)
            if not intents:
                intents, entities, _ = _detect_intents(query)
            content = _build_planner_response(intents, entities)
            logger.info("mock_provider.planner", agents=json.loads(content).get("selected_agents"))

        elif is_validator:
            content = _build_validator_response(user_content)
            logger.info("mock_provider.validator")

        elif is_aggregator:
            content = _build_aggregator_response(user_content, query)
            logger.info("mock_provider.aggregator", chars=len(content))

        else:
            content = "Your request has been processed by the government services platform."

        latency_ms = (time.monotonic() - t0) * 1000
        return LLMResponse(
            content=content,
            provider="mock",
            model="mock-dynamic-v2",
            usage={"input_tokens": 200, "output_tokens": 350, "total_tokens": 550},
            latency_ms=round(latency_ms, 2),
        )

    async def stream(self, messages: list[dict[str, str]], temperature: float = 0.2,
                     max_tokens: int = 4096) -> AsyncGenerator[str, None]:
        response = await self.invoke(messages, temperature, max_tokens)
        for word in response.content.split():
            yield word + " "

    def structured_output(self, prompt: str, schema: Type[BaseModel],
                          system_prompt: str = "") -> BaseModel:
        try:
            return schema()
        except Exception:
            return schema.model_construct()

    def bind_tools(self, tools, temperature: float = 0.0) -> BaseChatModel:
        return _MockChatModel(mock_tools=list(tools))

    def get_embeddings(self):
        return None
