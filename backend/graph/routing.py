from __future__ import annotations

from backend.graph.state import CitizenServiceState

# Map agent names (as returned by planner) → node names in the graph
_AGENT_TO_NODE: dict[str, str] = {
    "application_agent": "application_agent_node",
    "billing_agent":     "billing_agent_node",
    "knowledge_agent":   "knowledge_agent_node",
    "complaint_agent":   "complaint_agent_node",
}


def route_to_agents(state: CitizenServiceState) -> list[str]:
    """
    Conditional edge function.
    Returns the list of agent node names to run in parallel, based on
    the `selected_agents` list produced by the Planner Agent.

    This function contains NO keyword matching, NO hardcoded scenarios.
    Routing is 100% driven by the planner's LLM-based decision.
    """
    selected: list[str] = state.get("selected_agents", [])

    # Resolve agent names → graph node names
    nodes: list[str] = []
    seen: set[str] = set()
    for agent_name in selected:
        node = _AGENT_TO_NODE.get(agent_name)
        if node and node not in seen:
            nodes.append(node)
            seen.add(node)

    if nodes:
        return nodes

    # Fallback: if planner output was unusable, use execution_plan (backward compat)
    plan = state.get("execution_plan")
    if plan and plan.tasks:
        from backend.models.schemas import AgentName
        legacy_map: dict[AgentName, str] = {
            AgentName.APPLICATION: "application_agent_node",
            AgentName.BILLING:     "billing_agent_node",
            AgentName.KNOWLEDGE:   "knowledge_agent_node",
            AgentName.COMPLAINT:   "complaint_agent_node",
        }
        fallback: list[str] = []
        seen_agents: set[AgentName] = set()
        for task in plan.tasks:
            if task.agent in legacy_map and task.agent not in seen_agents:
                fallback.append(legacy_map[task.agent])
                seen_agents.add(task.agent)
        if fallback:
            return fallback

    return ["knowledge_agent_node"]
