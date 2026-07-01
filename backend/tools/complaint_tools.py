from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Any

from langchain_core.tools import tool


_COMPLAINTS: dict[str, dict[str, Any]] = {}


@tool
def create_complaint(
    subject: str,
    description: str,
    category: str = "General",
    application_id: str | None = None,
    priority: str = "Medium",
) -> dict[str, Any]:
    """
    Register a new citizen complaint.

    Args:
        subject: One-line summary of the complaint.
        description: Detailed description of the issue.
        category: Category (e.g., 'Delay', 'Misconduct', 'Billing Error').
        application_id: Optional linked application reference.
        priority: 'Low', 'Medium', 'High', or 'Critical'.

    Returns:
        Complaint ID, tracking number, and expected resolution timeline.
    """
    complaint_id = f"CMP-{uuid.uuid4().hex[:8].upper()}"
    now = datetime.utcnow()
    expected_resolution = now + timedelta(days=5)

    record = {
        "complaint_id": complaint_id,
        "subject": subject,
        "description": description,
        "category": category,
        "application_id": application_id,
        "priority": priority,
        "status": "Registered",
        "escalation_level": 0,
        "created_at": now.isoformat(),
        "expected_resolution": expected_resolution.strftime("%Y-%m-%d"),
        "assigned_to": "Citizen Services Team",
        "acknowledgement_sent": True,
    }
    _COMPLAINTS[complaint_id] = record

    return {
        "success": True,
        "data": record,
        "message": (
            f"Your complaint has been successfully registered with ID: {complaint_id}. "
            f"You will receive an acknowledgement email shortly. "
            f"Expected resolution by: {expected_resolution.strftime('%B %d, %Y')}."
        ),
    }


@tool
def check_complaint_status(complaint_id: str) -> dict[str, Any]:
    """
    Check the current status and escalation level of an existing complaint.

    Args:
        complaint_id: The complaint reference number (e.g., CMP-A1B2C3D4).

    Returns:
        Current status, escalation level, and resolution timeline.
    """
    record = _COMPLAINTS.get(complaint_id.upper())
    if not record:
        # Simulate an existing complaint
        record = {
            "complaint_id": complaint_id,
            "subject": "Permit Application Delay",
            "status": "Under Investigation",
            "escalation_level": 1,
            "created_at": (datetime.utcnow() - timedelta(days=7)).isoformat(),
            "expected_resolution": (datetime.utcnow() + timedelta(days=3)).strftime("%Y-%m-%d"),
            "assigned_to": "Senior Officer — Licensing",
            "last_update": "Complaint escalated to Level 1 due to no resolution within 5 days.",
        }

    return {"success": True, "data": record}
