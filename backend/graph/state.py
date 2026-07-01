from __future__ import annotations

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from backend.models.schemas import (
    AgentLog,
    ApplicationResult,
    BillingResult,
    ComplaintResult,
    ExecutionMetrics,
    ExecutionPlan,
    KnowledgeResult,
    ValidationResult,
)


# ---------------------------------------------------------------------------
# Reducer helpers
# ---------------------------------------------------------------------------


def _append_logs(existing: list, new: list) -> list:
    """Merge agent log lists (used as a LangGraph reducer)."""
    return existing + new


def _append_timeline(existing: list, new: list) -> list:
    """Append execution timeline events (used as a LangGraph reducer)."""
    return existing + new


def _merge_dict(existing: dict, new: dict) -> dict:
    return {**existing, **new}


# ---------------------------------------------------------------------------
# Shared Graph State
# ---------------------------------------------------------------------------


class CitizenServiceState(TypedDict):
    # ── Core query ────────────────────────────────────────────────────────
    user_query: str
    session_id: str
    citizen_id: Optional[str]

    # ── LangChain message history ─────────────────────────────────────────
    messages: Annotated[list[BaseMessage], add_messages]

    # ── Orchestrator outputs (new structured format) ──────────────────────
    query_type: str                     # "service_request"|"inquiry"|"complaint"|"information"
    intents: list[str]                  # ["application_status", "receipt_request"]
    entities: dict[str, Any]            # {"application_id": "APP-2024-001"}
    confidence: float                   # 0.0 – 1.0
    intent_confidence: dict[str, float] # {"application_status": 0.96, "receipt_request": 0.92}

    # ── Legacy orchestrator fields (kept for backward compat) ─────────────
    intent_list: list[str]              # same as intents
    complexity: str                     # "simple" | "moderate" | "complex"
    requires_parallel: bool

    # ── Planner outputs ───────────────────────────────────────────────────
    selected_agents: list[str]          # ["application_agent", "billing_agent"]
    skipped_agents: list[str]           # ["knowledge_agent", "complaint_agent"]
    agent_explanations: dict[str, str]  # why each agent was selected or skipped
    execution_mode: str                 # "parallel" | "sequential"
    execution_plan: Optional[ExecutionPlan]

    # ── Knowledge / RAG results ───────────────────────────────────────────
    rag_documents: list[dict[str, Any]] # documents retrieved by knowledge agent

    # ── Per-agent results (None = agent was not invoked) ──────────────────
    application_result: Optional[ApplicationResult]
    billing_result: Optional[BillingResult]
    knowledge_result: Optional[KnowledgeResult]
    complaint_result: Optional[ComplaintResult]

    # ── Validator output ──────────────────────────────────────────────────
    validation_result: Optional[ValidationResult]

    # ── Aggregator final answer ───────────────────────────────────────────
    final_response: str

    # ── Execution timeline (reducer appends per-agent events) ─────────────
    execution_timeline: Annotated[list[dict[str, Any]], _append_timeline]

    # ── Observability ─────────────────────────────────────────────────────
    agent_logs: Annotated[list[AgentLog], _append_logs]
    execution_metrics: ExecutionMetrics

    # ── Extra context ─────────────────────────────────────────────────────
    metadata: dict[str, Any]
