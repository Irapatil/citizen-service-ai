from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta
from typing import Any

from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Simulated data store (replace with real DB calls in production)
# ---------------------------------------------------------------------------

_APPLICATIONS: dict[str, dict[str, Any]] = {
    "APP-2024-001": {
        "application_id": "APP-2024-001",
        "type": "Permit Renewal",
        "status": "Under Review",
        "submitted_date": "2026-06-09",
        "last_updated": "2026-06-20",
        "applicant_name": "John Doe",
        "assigned_officer": "Officer Smith",
        "estimated_completion": "2026-06-30",
        "notes": "Additional documentation requested on June 15th.",
    },
    "APP-2024-002": {
        "application_id": "APP-2024-002",
        "type": "Building Permit",
        "status": "Approved",
        "submitted_date": "2026-05-01",
        "last_updated": "2026-06-01",
        "applicant_name": "Jane Smith",
        "assigned_officer": "Officer Johnson",
        "estimated_completion": "2026-06-01",
        "notes": "Permit approved. Ready for collection.",
    },
}


@tool
def get_application_status(application_id: str) -> dict[str, Any]:
    """
    Retrieve the current status of a citizen's application by its ID.

    Args:
        application_id: The unique application identifier (e.g., APP-2024-001).

    Returns:
        A dictionary with status, dates, assigned officer, and notes.
    """
    app = _APPLICATIONS.get(application_id.upper())
    if not app:
        # Simulate a found application with realistic data when ID is unknown
        app = {
            "application_id": application_id,
            "type": "Permit Renewal",
            "status": "Under Review",
            "submitted_date": (datetime.utcnow() - timedelta(days=14)).strftime("%Y-%m-%d"),
            "last_updated": (datetime.utcnow() - timedelta(days=3)).strftime("%Y-%m-%d"),
            "applicant_name": "Citizen",
            "assigned_officer": "Officer Williams",
            "estimated_completion": (datetime.utcnow() + timedelta(days=7)).strftime("%Y-%m-%d"),
            "notes": "Your application is currently under review. "
                     "You will be notified once a decision is made.",
        }
    return {"success": True, "data": app}


@tool
def get_application_history(application_id: str) -> dict[str, Any]:
    """
    Retrieve the full audit history / timeline of an application.

    Args:
        application_id: The unique application identifier.

    Returns:
        A list of timeline events with dates and descriptions.
    """
    base_date = datetime.utcnow() - timedelta(days=14)
    history = [
        {
            "event": "Application Submitted",
            "date": base_date.strftime("%Y-%m-%d"),
            "description": "Application received and assigned reference number.",
        },
        {
            "event": "Acknowledgement Sent",
            "date": (base_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            "description": "Confirmation email sent to applicant.",
        },
        {
            "event": "Document Verification",
            "date": (base_date + timedelta(days=5)).strftime("%Y-%m-%d"),
            "description": "Supporting documents verified by department.",
        },
        {
            "event": "Additional Documents Requested",
            "date": (base_date + timedelta(days=6)).strftime("%Y-%m-%d"),
            "description": "Applicant asked to provide updated address proof.",
        },
        {
            "event": "Under Review",
            "date": (base_date + timedelta(days=11)).strftime("%Y-%m-%d"),
            "description": "Application assigned to senior officer for final review.",
        },
    ]
    return {"success": True, "application_id": application_id, "history": history}
