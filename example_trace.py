"""
Example Execution Trace
=======================
Demonstrates the Citizen Service AI multi-agent workflow with three realistic
citizen queries that trigger different combinations of parallel agents.

Run:
    python example_trace.py

Requires environment variables from .env (copy from .env.example).
"""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime

# ---------------------------------------------------------------------------
# Execution trace logger
# ---------------------------------------------------------------------------


def print_trace(label: str, data: object, indent: int = 0) -> None:
    prefix = "  " * indent
    print(f"\n{prefix}{'─' * 60}")
    print(f"{prefix}  {label}")
    print(f"{prefix}{'─' * 60}")
    if isinstance(data, dict):
        for k, v in data.items():
            print(f"{prefix}  {k}: {v}")
    elif isinstance(data, list):
        for item in data:
            print(f"{prefix}  • {item}")
    else:
        text = str(data)
        for line in text.splitlines():
            print(f"{prefix}  {line}")


# ---------------------------------------------------------------------------
# Mock workflow runner (for demonstration without Azure credentials)
# ---------------------------------------------------------------------------


class MockWorkflowRunner:
    """
    Simulates the multi-agent workflow output for demonstration purposes.
    Replace with `get_compiled_graph().ainvoke(initial_state)` in production.
    """

    SCENARIOS = [
        {
            "query": (
                "I submitted my permit renewal application two weeks ago and need an update. "
                "I also need a copy of my payment receipt."
            ),
            "intents": ["application_status", "receipt_request"],
            "complexity": "moderate",
            "agents": ["orchestrator", "planner", "application_agent", "billing_agent", "aggregator"],
            "plan": {
                "tasks": [
                    {"agent": "application_agent", "task": "check_application_status"},
                    {"agent": "billing_agent",     "task": "generate_receipt_pdf"},
                ]
            },
            "application_result": {
                "application_id": "APP-2024-001",
                "status": "Under Review",
                "submitted_date": "2026-06-09",
                "last_updated": "2026-06-20",
                "message": (
                    "Your permit renewal application APP-2024-001 was submitted on 9 June 2026 "
                    "and is currently Under Review. It has been assigned to Officer Smith. "
                    "Estimated completion: 30 June 2026."
                ),
            },
            "billing_result": {
                "payment_id": "PAY-2024-001",
                "amount": 150.00,
                "payment_date": "2026-06-09",
                "receipt_url": "https://services.gov/receipts/PAY-2024-001/download",
                "message": (
                    "Payment of $150.00 confirmed on 9 June 2026 (Invoice INV-2024-00891). "
                    "Download your receipt at: https://services.gov/receipts/PAY-2024-001/download"
                ),
            },
            "final_response": (
                "Dear Citizen,\n\n"
                "Here is a full update on your enquiries:\n\n"
                "**Application Status**\n"
                "Your permit renewal application (APP-2024-001) submitted on 9 June 2026 is "
                "currently Under Review by Officer Smith. Based on standard processing times, "
                "you can expect a decision by 30 June 2026. You will receive an email notification "
                "when the status changes.\n\n"
                "**Payment Receipt**\n"
                "Your payment of $150.00 made on 9 June 2026 has been confirmed "
                "(Invoice: INV-2024-00891). You can download your receipt here:\n"
                "https://services.gov/receipts/PAY-2024-001/download\n"
                "(Link valid for 24 hours)\n\n"
                "**Next Steps**\n"
                "• No further action is needed at this time.\n"
                "• If you do not receive a decision by 30 June 2026, contact us at 1800-GOV-HELP.\n\n"
                "Is there anything else I can help you with?"
            ),
            "metrics": {
                "total_agents_invoked": 5,
                "successful_agents": 5,
                "failed_agents": 0,
                "parallel_groups": 1,
                "total_latency_ms": 2340.5,
                "total_tokens": 1842,
            },
        },
        {
            "query": (
                "What documents do I need for permit renewal? "
                "I also want to register a complaint about the long wait time."
            ),
            "intents": ["faq", "complaint_create"],
            "complexity": "moderate",
            "agents": ["orchestrator", "planner", "knowledge_agent", "complaint_agent", "aggregator"],
            "plan": {
                "tasks": [
                    {"agent": "knowledge_agent", "task": "retrieve_documents"},
                    {"agent": "complaint_agent", "task": "create_complaint"},
                ]
            },
            "knowledge_result": {
                "answer": (
                    "For permit renewal you will need: (1) Completed renewal form, "
                    "(2) Proof of identity, (3) Proof of address (within 3 months), "
                    "(4) Previous permit copy, (5) Payment receipt for renewal fee ($150)."
                ),
                "sources": ["faq-001", "faq-002"],
            },
            "complaint_result": {
                "complaint_id": "CMP-A1B2C3D4",
                "status": "Registered",
                "escalation_level": 0,
                "message": (
                    "Your complaint (CMP-A1B2C3D4) regarding permit processing delays has been "
                    "registered. Expected resolution: 28 June 2026. "
                    "An acknowledgement has been sent to your registered email."
                ),
            },
            "final_response": (
                "Dear Citizen,\n\n"
                "**Documents Required for Permit Renewal**\n"
                "Please prepare the following:\n"
                "1. Completed renewal form (available online or at service centres)\n"
                "2. Proof of identity (national ID or passport)\n"
                "3. Proof of address dated within the last 3 months\n"
                "4. Copy of your previous permit\n"
                "5. Payment receipt for the $150 renewal fee\n\n"
                "**Complaint Registered**\n"
                "Your complaint about processing delays has been registered: **CMP-A1B2C3D4**\n"
                "Our team will investigate and respond within 5 business days (by 28 June 2026).\n\n"
                "**Next Steps**\n"
                "• Gather the required documents and submit your renewal application.\n"
                "• Track your complaint status using ID: CMP-A1B2C3D4.\n\n"
                "We sincerely apologise for any inconvenience caused."
            ),
            "metrics": {
                "total_agents_invoked": 5,
                "successful_agents": 5,
                "failed_agents": 0,
                "parallel_groups": 1,
                "total_latency_ms": 1980.2,
                "total_tokens": 1623,
            },
        },
        {
            "query": (
                "What is the permit renewal fee policy? "
                "Check status of application APP-2024-002. "
                "Also get my payment receipt and register a complaint about rude staff."
            ),
            "intents": ["policy_search", "application_status", "receipt_request", "complaint_create"],
            "complexity": "complex",
            "agents": [
                "orchestrator", "planner",
                "knowledge_agent", "application_agent", "billing_agent", "complaint_agent",
                "aggregator",
            ],
            "plan": {
                "tasks": [
                    {"agent": "knowledge_agent",  "task": "search_policies"},
                    {"agent": "application_agent","task": "check_application_status"},
                    {"agent": "billing_agent",    "task": "generate_receipt_pdf"},
                    {"agent": "complaint_agent",  "task": "create_complaint"},
                ]
            },
            "final_response": (
                "Dear Citizen,\n\n"
                "Here is a comprehensive response to all your enquiries:\n\n"
                "**Permit Renewal Fee Policy**\n"
                "Standard fee: $150. Late renewal (after expiry): additional $50 penalty. "
                "Senior citizens and disability cardholders: 50% concession applies.\n\n"
                "**Application Status — APP-2024-002**\n"
                "Status: Approved ✓ — Your building permit has been approved and is ready for collection.\n\n"
                "**Payment Receipt**\n"
                "Receipt generated: https://services.gov/receipts/PAY-2024-002/download\n\n"
                "**Complaint Registered**\n"
                "ID: CMP-B5C6D7E8 — Complaint about staff conduct has been registered. "
                "Our quality team will respond within 5 business days.\n\n"
                "**Next Steps**\n"
                "• Collect your approved permit from the nearest service centre.\n"
                "• Download your receipt using the link above (valid 24 hours).\n"
                "• Track your complaint using ID: CMP-B5C6D7E8."
            ),
            "metrics": {
                "total_agents_invoked": 7,
                "successful_agents": 7,
                "failed_agents": 0,
                "parallel_groups": 1,
                "total_latency_ms": 3120.8,
                "total_tokens": 2891,
            },
        },
    ]

    async def run(self, scenario: dict) -> dict:
        await asyncio.sleep(0.1)  # Simulate async execution
        return scenario


# ---------------------------------------------------------------------------
# Main trace runner
# ---------------------------------------------------------------------------


async def run_trace() -> None:
    runner = MockWorkflowRunner()
    session_id = str(uuid.uuid4())

    print("\n" + "=" * 70)
    print("  CITIZEN SERVICE AI — MULTI-AGENT WORKFLOW EXECUTION TRACE")
    print("  Architecture: Orchestrator -> Planner -> Parallel Agents -> Aggregator")
    print("  Model: Azure OpenAI GPT-4o | Framework: LangGraph")
    print("=" * 70)

    for i, scenario in enumerate(runner.SCENARIOS, 1):
        result = await runner.run(scenario)
        ts = datetime.utcnow().isoformat()

        print(f"\n\n{'═' * 70}")
        print(f"  SCENARIO {i} OF {len(runner.SCENARIOS)}")
        print(f"{'═' * 70}")

        print_trace("USER QUERY", result["query"])
        print_trace("TIMESTAMP", ts)

        print_trace(
            "LEVEL 1 — ORCHESTRATOR AGENT",
            {
                "Detected Intents": result["intents"],
                "Complexity":       result["complexity"],
                "Parallel Required": len(result["intents"]) > 1,
            },
        )

        print_trace(
            "LEVEL 2 — TASK PLANNER AGENT",
            {
                "Execution Plan": json.dumps(result["plan"], indent=2),
            },
        )

        print(f"\n  {'─' * 60}")
        print(f"  LEVEL 3 — PARALLEL AGENT EXECUTION")
        print(f"  Agents running simultaneously: {[t['agent'] for t in result['plan']['tasks']]}")
        print(f"  {'─' * 60}")

        if "application_result" in result:
            print_trace("  Application Agent", result["application_result"], indent=1)
        if "billing_result" in result:
            print_trace("  Billing Agent", result["billing_result"], indent=1)
        if "knowledge_result" in result:
            print_trace("  Knowledge Agent", result["knowledge_result"], indent=1)
        if "complaint_result" in result:
            print_trace("  Complaint Agent", result["complaint_result"], indent=1)

        print_trace(
            "LEVEL 4 — RESPONSE AGGREGATOR",
            result["final_response"],
        )

        print_trace(
            "EXECUTION METRICS",
            {
                "Agents Invoked":    result["metrics"]["total_agents_invoked"],
                "Successful":        result["metrics"]["successful_agents"],
                "Failed":            result["metrics"]["failed_agents"],
                "Parallel Groups":   result["metrics"]["parallel_groups"],
                "Total Latency":     f"{result['metrics']['total_latency_ms']} ms",
                "Total Tokens":      result["metrics"]["total_tokens"],
                "Session ID":        session_id,
            },
        )

    print("\n\n" + "=" * 70)
    print("  ARCHITECTURE SUMMARY")
    print("=" * 70)
    summary = {
        "Orchestrator Agent":  "Intent detection, complexity scoring, execution strategy",
        "Task Planner Agent":  "Structured execution plan, deterministic + LLM-powered routing",
        "Application Agent":   "Permit status & history (tools: get_application_status, get_application_history)",
        "Billing Agent":       "Payment lookup & receipt generation (tools: get_payment_details, generate_receipt_pdf)",
        "Knowledge Agent":     "FAQ & policy retrieval using RAG + ChromaDB (tools: retrieve_documents, search_policies)",
        "Complaint Agent":     "Complaint registration & tracking (tools: create_complaint, check_complaint_status)",
        "Aggregator Agent":    "Response synthesis, deduplication, citizen-friendly formatting",
        "Parallel Execution":  "LangGraph conditional fan-out — all required agents run simultaneously",
        "Memory":              "Session-scoped conversation history (in-memory → Redis/PostgreSQL in production)",
        "Observability":       "Per-agent: start_time, end_time, latency_ms, token_usage, status",
        "Error Handling":      "Per-agent isolation — one failure does not block other agents",
    }
    for k, v in summary.items():
        print(f"  • {k:<25} {v}")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(run_trace())
