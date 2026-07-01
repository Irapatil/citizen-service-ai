from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AgentName(str, Enum):
    ORCHESTRATOR = "orchestrator"
    PLANNER      = "planner"
    APPLICATION  = "application_agent"
    BILLING      = "billing_agent"
    KNOWLEDGE    = "knowledge_agent"
    COMPLAINT    = "complaint_agent"
    VALIDATOR    = "validator"
    AGGREGATOR   = "aggregator"


class IntentType(str, Enum):
    APPLICATION_STATUS  = "application_status"
    APPLICATION_HISTORY = "application_history"
    PAYMENT_LOOKUP      = "payment_lookup"
    RECEIPT_REQUEST     = "receipt_request"
    FAQ                 = "faq"
    POLICY_SEARCH       = "policy_search"
    COMPLAINT_CREATE    = "complaint_create"
    COMPLAINT_STATUS    = "complaint_status"
    GENERAL             = "general"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED  = "failed"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Orchestrator structured output
# ---------------------------------------------------------------------------


class OrchestratorOutput(BaseModel):
    """Structured output from the Orchestrator Agent."""
    query_type: str = "service_request"
    intents: list[str] = Field(default_factory=list)
    entities: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.9
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Planner structured output
# ---------------------------------------------------------------------------


class PlannerOutput(BaseModel):
    """Structured output from the Task Planner Agent."""
    selected_agents: list[str] = Field(default_factory=list)
    tasks: list[dict[str, Any]] = Field(default_factory=list)
    execution_mode: str = "parallel"
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Validator structured output
# ---------------------------------------------------------------------------


class ValidationResult(BaseModel):
    """Structured output from the Validator Agent."""
    is_valid: bool = True
    completeness_score: float = 1.0
    issues: list[str] = Field(default_factory=list)
    agents_to_retry: list[str] = Field(default_factory=list)
    verdict: str = "All outputs validated successfully"


# ---------------------------------------------------------------------------
# Task & Execution Plan
# ---------------------------------------------------------------------------


class Task(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent: AgentName
    task: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    priority: int = 1


class ExecutionPlan(BaseModel):
    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tasks: list[Task]
    reasoning: str = ""
    estimated_agents: list[AgentName] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent Results
# ---------------------------------------------------------------------------


class ApplicationResult(BaseModel):
    application_id: Optional[str] = None
    status: Optional[str] = None
    submitted_date: Optional[str] = None
    last_updated: Optional[str] = None
    history: list[dict[str, Any]] = Field(default_factory=list)
    message: str = ""
    success: bool = False
    error: Optional[str] = None


class BillingResult(BaseModel):
    payment_id: Optional[str] = None
    amount: Optional[float] = None
    payment_date: Optional[str] = None
    receipt_url: Optional[str] = None
    invoice_number: Optional[str] = None
    message: str = ""
    success: bool = False
    error: Optional[str] = None


class KnowledgeResult(BaseModel):
    documents: list[dict[str, Any]] = Field(default_factory=list)
    policies: list[dict[str, Any]] = Field(default_factory=list)
    answer: str = ""
    sources: list[str] = Field(default_factory=list)
    success: bool = False
    error: Optional[str] = None


class ComplaintResult(BaseModel):
    complaint_id: Optional[str] = None
    status: Optional[str] = None
    created_at: Optional[str] = None
    escalation_level: Optional[int] = None
    message: str = ""
    success: bool = False
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Execution Timeline
# ---------------------------------------------------------------------------


class TimelineEntry(BaseModel):
    """A single event in the agent execution timeline."""
    agent: str
    display_name: str
    status: str = "pending"   # pending | running | completed | skipped | failed
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_ms: Optional[float] = None
    summary: str = ""
    is_parallel: bool = False


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


class AgentLog(BaseModel):
    agent_name: AgentName
    task: str = ""
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    latency_ms: Optional[float] = None
    status: TaskStatus = TaskStatus.PENDING
    token_usage: dict[str, int] = Field(default_factory=dict)
    error: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def complete(self, status: TaskStatus = TaskStatus.SUCCESS, error: str | None = None) -> None:
        self.end_time = datetime.utcnow()
        self.latency_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.status = status
        if error:
            self.error = error


class ExecutionMetrics(BaseModel):
    total_agents_invoked: int = 0
    successful_agents: int = 0
    failed_agents: int = 0
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    parallel_groups: int = 0


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


class ConversationTurn(BaseModel):
    turn_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    user_query: str
    final_response: str
    agent_outputs: dict[str, Any] = Field(default_factory=dict)
    intents: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# API Request / Response
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    user_query: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    citizen_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    session_id: str
    final_response: str
    query_type: str = "service_request"
    intents_detected: list[str]
    entities: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.9
    selected_agents: list[str] = Field(default_factory=list)
    execution_mode: str = "parallel"
    agents_used: list[str]
    execution_metrics: ExecutionMetrics
    execution_timeline: list[dict[str, Any]] = Field(default_factory=list)
    validation_result: Optional[dict[str, Any]] = None
    application_result: Optional[ApplicationResult] = None
    billing_result: Optional[BillingResult] = None
    knowledge_result: Optional[KnowledgeResult] = None
    complaint_result: Optional[ComplaintResult] = None
    success: bool = True
    error: Optional[str] = None
