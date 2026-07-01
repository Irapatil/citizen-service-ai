from __future__ import annotations

from functools import lru_cache

import structlog
from langgraph.graph import StateGraph, START, END

from backend.agents.orchestrator import orchestrator_node
from backend.agents.planner import planner_node
from backend.agents.application_agent import application_agent_node
from backend.agents.billing_agent import billing_agent_node
from backend.agents.knowledge_agent import knowledge_agent_node
from backend.agents.complaint_agent import complaint_agent_node
from backend.agents.validator import validator_node
from backend.agents.aggregator import aggregator_node
from backend.graph.routing import route_to_agents
from backend.graph.state import CitizenServiceState

logger = structlog.get_logger(__name__)

_PARALLEL_NODES = [
    "application_agent_node",
    "billing_agent_node",
    "knowledge_agent_node",
    "complaint_agent_node",
]


def build_graph() -> StateGraph:
    """
    Construct the multi-agent LangGraph StateGraph.

    Topology:
      START
        → orchestrator_node           (intent detection, entity extraction)
        → planner_node                (agent selection, execution plan)
        → [conditional parallel fan-out via route_to_agents()]
            → application_agent_node  ─┐
            → billing_agent_node      ─┤
            → knowledge_agent_node    ─┤→ validator_node → aggregator_node → END
            → complaint_agent_node    ─┘

    The validator runs after ALL selected parallel agents complete.
    The aggregator synthesises the validated outputs into a final response.
    """
    graph = StateGraph(CitizenServiceState)

    # ── Register nodes ────────────────────────────────────────────────
    graph.add_node("orchestrator_node",      orchestrator_node)
    graph.add_node("planner_node",           planner_node)
    graph.add_node("application_agent_node", application_agent_node)
    graph.add_node("billing_agent_node",     billing_agent_node)
    graph.add_node("knowledge_agent_node",   knowledge_agent_node)
    graph.add_node("complaint_agent_node",   complaint_agent_node)
    graph.add_node("validator_node",         validator_node)
    graph.add_node("aggregator_node",        aggregator_node)

    # ── Static edges ──────────────────────────────────────────────────
    graph.add_edge(START, "orchestrator_node")
    graph.add_edge("orchestrator_node", "planner_node")

    # ── Conditional parallel fan-out: planner → selected agents ───────
    graph.add_conditional_edges(
        "planner_node",
        route_to_agents,
        {node: node for node in _PARALLEL_NODES},
    )

    # ── All parallel agents converge at validator ─────────────────────
    for node in _PARALLEL_NODES:
        graph.add_edge(node, "validator_node")

    # ── Validator → Aggregator → END ──────────────────────────────────
    graph.add_edge("validator_node", "aggregator_node")
    graph.add_edge("aggregator_node", END)

    return graph


@lru_cache(maxsize=1)
def get_compiled_graph():
    """Return a singleton compiled graph (thread-safe, shared across requests)."""
    graph = build_graph()
    compiled = graph.compile()
    logger.info("graph.compiled", nodes=list(compiled.nodes.keys()))
    return compiled
