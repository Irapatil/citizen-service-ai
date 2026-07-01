from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Any

from langchain_core.tools import tool


_PAYMENTS: dict[str, dict[str, Any]] = {
    "PAY-2024-001": {
        "payment_id": "PAY-2024-001",
        "application_id": "APP-2024-001",
        "amount": 150.00,
        "currency": "USD",
        "payment_date": "2026-06-09",
        "payment_method": "Online Banking",
        "transaction_ref": "TXN-789456123",
        "status": "Completed",
        "description": "Permit Renewal Fee",
        "invoice_number": "INV-2024-00891",
    },
}


@tool
def get_payment_details(
    payment_id: str | None = None,
    application_id: str | None = None,
) -> dict[str, Any]:
    """
    Look up payment details by payment ID or application ID.

    Args:
        payment_id: Optional payment reference (e.g., PAY-2024-001).
        application_id: Optional application reference to find linked payments.

    Returns:
        Payment details including amount, date, method and status.
    """
    # Search by payment_id first
    if payment_id:
        record = _PAYMENTS.get(payment_id.upper())
        if record:
            return {"success": True, "data": record}

    # Search by application_id
    if application_id:
        for rec in _PAYMENTS.values():
            if rec.get("application_id", "").upper() == application_id.upper():
                return {"success": True, "data": rec}

    # Simulate realistic payment when not in store
    simulated = {
        "payment_id": payment_id or f"PAY-{uuid.uuid4().hex[:8].upper()}",
        "application_id": application_id or "UNKNOWN",
        "amount": 150.00,
        "currency": "USD",
        "payment_date": (datetime.utcnow() - timedelta(days=14)).strftime("%Y-%m-%d"),
        "payment_method": "Online Banking",
        "transaction_ref": f"TXN-{uuid.uuid4().hex[:9].upper()}",
        "status": "Completed",
        "description": "Permit Renewal Fee",
        "invoice_number": f"INV-2024-{uuid.uuid4().hex[:5].upper()}",
    }
    return {"success": True, "data": simulated}


@tool
def generate_receipt_pdf(payment_id: str) -> dict[str, Any]:
    """
    Generate and return a downloadable receipt PDF link for a given payment.

    Args:
        payment_id: The payment reference number.

    Returns:
        A dict with receipt_url, generated_at, and expiry.
    """
    receipt_url = f"https://services.gov/receipts/{payment_id}/download"
    return {
        "success": True,
        "payment_id": payment_id,
        "receipt_url": receipt_url,
        "format": "PDF",
        "generated_at": datetime.utcnow().isoformat(),
        "expires_at": (datetime.utcnow() + timedelta(hours=24)).isoformat(),
        "message": (
            f"Your receipt for payment {payment_id} has been generated. "
            f"Download it at: {receipt_url}"
        ),
    }
