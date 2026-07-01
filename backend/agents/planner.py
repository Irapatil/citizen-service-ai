from __future__ import annotations

import json
from datetime import datetime

import structlog

from backend.graph.state import CitizenServiceState
from backend.models.schemas import (
    AgentLog,
    AgentName,
    ExecutionPlan,
    Task,
    TaskStatus,
)

logger = structlog.get_logger(__name__)

_SYSTEM_PROMPT = """You are the Task Planner for a Government Citizen Services Platform.

Your role: Convert detected intents and entities into a precise, executable plan.

Available agents and their capabilities:
- application_agent:  Permit status, application history, case tracking, renewal status
- billing_agent:      Payment lookup, receipt generation, invoice retrieval, transaction details
- knowledge_agent:    Government FAQs, policy documents, procedures, citizen guidance (RAG)
- complaint_agent:    File new complaints, track existing complaints, escalation management

Execution modes:
- parallel:    Multiple agents run simultaneously (use when agents are independent)
- sequential:  Agents run one after another (use only when agent B needs agent A's output)

RULES:
- Select ONLY the agents needed to address ALL detected intents
- Set execution_mode to "parallel" whenever 2+ independent agents are selected
- Include all entities (IDs, names) in the relevant task parameters
- Task names must be specific and descriptive (e.g., "check_application_status")
- If an entity ID is not known, use sensible defaults or omit it

Respond ONLY with valid JSON (no markdown, no explanation):
{
  "selected_agents": ["agent_name_1", "agent_name_2"],
  "tasks": [
    {
      "agent": "agent_name",
      "task": "descriptive_task_name",
      "parameters": {"key": "value"},
      "priority": 1
    }
  ],
  "execution_mode": "parallel",
  "reasoning": "Brief explanation of agent selection and execution strategy"
}"""

# Maps planner agent names → AgentName enum values
_AGENT_NAME_MAP: dict[str, AgentName] = {
    "application_agent": AgentName.APPLICATION,
    "billing_agent":     AgentName.BILLING,
    "knowledge_agent":   AgentName.KNOWLEDGE,
    "complaint_agent":   AgentName.COMPLAINT,
}


async def planner_node(state: CitizenServiceState) -> dict:
    """
    Level 2 — Task Planner Agent.
    Converts detected intents into a concrete ExecutionPlan and selects agents.
    Uses pure LLM reasoning — no hardcoded intent-to-agent mapping.
    """
    log = AgentLog(agent_name=AgentName.PLANNER, task="execution_planning")
    t_start = datetime.utcnow()
    logger.info("planner.start", intents=state.get("intents"), entities=state.get("entities"))
    print(f"[STEP 7] Planner START  intents={state.get('intents')}", flush=True)

    provider = get_llm_provider()

    intents: list[str]  = state.get("intents") or state.get("intent_list", ["general"])
    entities: dict      = state.get("entities", {})
    query_type: str     = state.get("query_type", "service_request")
    user_query: str     = state.get("user_query", "")

    user_content = (
        f"Citizen Query: {user_query}\n\n"
        f"Query Type: {query_type}\n"
        f"Detected Intents: {json.dumps(intents)}\n"
        f"Extracted Entities: {json.dumps(entities)}\n\n"
        "Generate the execution plan."
    )

    response = await provider.invoke(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        temperature=0.0,
    )
    log.token_usage = response.usage

    parsed: dict = {}
    try:
        clean = (
            response.content.strip()
            .removeprefix("```json").removeprefix("```")
            .removesuffix("```").strip()
        )
        parsed = json.loads(clean)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("planner.parse_error", error=str(exc))
        parsed = {}

    selected_agents: list[str]     = parsed.get("selected_agents", ["knowledge_agent"])
    raw_tasks: list[dict]          = parsed.get("tasks", [])
    execution_mode: str            = parsed.get("execution_mode", "parallel")
    reasoning: str                 = parsed.get("reasoning", "")
    agent_explanations: dict       = parsed.get("agent_explanations", {})

    # Derive skipped agents (all specialist agents not in selected_agents)
    _all_specialists = ["application_agent", "billing_agent", "knowledge_agent", "complaint_agent"]
    skipped_agents: list[str] = parsed.get(
        "skipped_agents",
        [a for a in _all_specialists if a not in selected_agents],
    )

    # Validate and normalise agent names
    valid_agents = list(_AGENT_NAME_MAP.keys())
    selected_agents = [a for a in selected_agents if a in valid_agents]
    if not selected_agents:
        selected_agents = ["knowledge_agent"]

    # Build LangGraph-compatible Task objects for each task
    tasks: list[Task] = []
    for rt in raw_tasks:
        agent_str = rt.get("agent", "")
        if agent_str not in _AGENT_NAME_MAP:
            continue
        try:
            tasks.append(Task(
                agent=_AGENT_NAME_MAP[agent_str],
                task=rt.get("task", "general_query"),
                parameters={**rt.get("parameters", {}), "query": user_query},
                priority=int(rt.get("priority", 1)),
            ))
        except (ValueError, KeyError):
            continue

    # Ensure at least one task per selected agent
    if not tasks:
        for a in selected_agents:
            if a in _AGENT_NAME_MAP:
                tasks.append(Task(
                    agent=_AGENT_NAME_MAP[a],
                    task="general_query",
                    parameters={"query": user_query, **entities},
                ))

    plan = ExecutionPlan(
        tasks=tasks,
        reasoning=reasoning,
        estimated_agents=[_AGENT_NAME_MAP[a] for a in selected_agents if a in _AGENT_NAME_MAP],
    )

    log.complete(TaskStatus.SUCCESS)
    duration_ms = (datetime.utcnow() - t_start).total_seconds() * 1000

    logger.info(
        "planner.done",
        provider=response.provider,
        selected_agents=selected_agents,
        execution_mode=execution_mode,
        tasks=[t.task for t in tasks],
        latency_ms=round(response.latency_ms, 1),
    )

    return {
        "selected_agents":   selected_agents,
        "skipped_agents":    skipped_agents,
        "agent_explanations": agent_explanations,
        "execution_mode":    execution_mode,
        "execution_plan":    plan,
        "requires_parallel": len(selected_agents) > 1,
        "agent_logs": [log],
        "execution_timeline": [{
            "agent":        "planner",
            "display_name": "Task Planner",
            "status":       "completed",
            "start_time":   t_start.isoformat(),
            "end_time":     datetime.utcnow().isoformat(),
            "duration_ms":  round(duration_ms, 1),
            "summary":      f"Selected {len(selected_agents)} agent(s): {', '.join(selected_agents)}",
            "is_parallel":  False,
        }],
    }


def get_llm_provider():
    from backend.services.llm import get_llm_provider as _get
    return _get()
