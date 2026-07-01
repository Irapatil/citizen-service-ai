from __future__ import annotations

"""
ChromaDB-backed knowledge store with 5 document categories and 20 documents.
Falls back to in-memory keyword search when chromadb is unavailable.
"""

import os
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Document corpus — 5 categories, 20 documents
# ---------------------------------------------------------------------------

_DOCUMENTS: list[dict[str, Any]] = [
    # ── Permit Renewal ──────────────────────────────────────────────────────
    {
        "id": "permit-renewal-001",
        "category": "permit_renewal",
        "title": "Permit Renewal Process Overview",
        "content": (
            "Citizens must renew permits at least 30 days before expiry. "
            "Renewal applications can be submitted online via the Citizen Portal or in person at any Service Centre. "
            "Standard processing time is 10-15 business days. "
            "Expedited processing (5 business days) is available for an additional $50 fee. "
            "You will receive email and SMS notifications at each stage of processing."
        ),
        "source": "Permit Renewal Policy v3.2",
    },
    {
        "id": "permit-renewal-002",
        "category": "permit_renewal",
        "title": "Required Documents for Permit Renewal",
        "content": (
            "To renew a permit you must provide: "
            "(1) Completed Form PR-200 (available online). "
            "(2) Proof of identity (passport, driver licence, or national ID). "
            "(3) Copy of the existing permit. "
            "(4) Payment of the renewal fee ($150 for residential, $300 for commercial). "
            "Incomplete applications will be returned within 3 business days with a deficiency notice."
        ),
        "source": "Permit Renewal Policy v3.2",
    },
    {
        "id": "permit-renewal-003",
        "category": "permit_renewal",
        "title": "Permit Renewal Status Tracking",
        "content": (
            "Track your permit renewal status using your Application ID on the Citizen Portal at portal.gov/track. "
            "Application IDs follow the format APP-YYYY-NNNN. "
            "Status stages: Received -> Under Review -> Approved -> Issued. "
            "If your application remains in Under Review for more than 12 business days, "
            "contact the Permit Office on 1800-PERMIT."
        ),
        "source": "Citizen Service Guide 2024",
    },
    {
        "id": "permit-renewal-004",
        "category": "permit_renewal",
        "title": "Permit Renewal Fees and Exemptions",
        "content": (
            "Standard renewal fees: Residential $150, Commercial $300, Industrial $600. "
            "Fee exemptions are available for seniors (65+), full-time students, and low-income households (income < $30,000/year). "
            "To claim an exemption, attach the relevant supporting documentation to Form PR-200. "
            "Fees are non-refundable once processing has commenced."
        ),
        "source": "Payment Policy Document 2024",
    },
    # ── Government FAQ ───────────────────────────────────────────────────────
    {
        "id": "faq-001",
        "category": "faq",
        "title": "How long does application processing take?",
        "content": (
            "Standard processing times by application type: "
            "Permit renewals: 10-15 business days. "
            "New permit applications: 20-30 business days. "
            "Licence amendments: 5-10 business days. "
            "Planning approvals: 45-90 business days. "
            "Urgent applications can be escalated by contacting your local Service Centre. "
            "Processing times may be extended during peak periods (July-August, December)."
        ),
        "source": "Government FAQ 2024",
    },
    {
        "id": "faq-002",
        "category": "faq",
        "title": "Service Centre Operating Hours",
        "content": (
            "Service Centres are open Monday to Friday, 8:30 AM to 4:30 PM. "
            "Extended hours (until 6:00 PM) are available on Tuesdays and Thursdays. "
            "Saturday appointments are available at major centres (book online). "
            "Online services are available 24/7 via the Citizen Portal. "
            "Phone support: 1800-CITIZEN (Mon-Fri, 8:00 AM-5:00 PM)."
        ),
        "source": "Government FAQ 2024",
    },
    {
        "id": "faq-003",
        "category": "faq",
        "title": "How to Update Contact Details",
        "content": (
            "Update your contact details via the Citizen Portal under My Profile. "
            "You can update: email address, phone number, postal address. "
            "Changes to legal name or date of birth require supporting documentation "
            "and must be completed in person at a Service Centre. "
            "Contact updates take effect immediately for future communications."
        ),
        "source": "Government FAQ 2024",
    },
    {
        "id": "faq-004",
        "category": "faq",
        "title": "What happens if my application is rejected?",
        "content": (
            "If your application is rejected, you will receive a written notice detailing the reasons. "
            "You have 30 days from the rejection date to: "
            "(1) Correct and resubmit the application (no additional fee). "
            "(2) Lodge a formal appeal via Form AP-100. "
            "Appeals are reviewed by an independent panel within 20 business days. "
            "If the appeal is upheld, the original fee is refunded."
        ),
        "source": "Citizen Service Guide 2024",
    },
    # ── Citizen Service Guide ────────────────────────────────────────────────
    {
        "id": "service-guide-001",
        "category": "service_guide",
        "title": "Citizen Portal Registration and Login",
        "content": (
            "Register for the Citizen Portal at portal.gov/register using your email address and a government-issued ID. "
            "Two-factor authentication (2FA) is mandatory for all accounts. "
            "Forgotten password: use the Reset Password link on the login page - a reset link will be sent to your registered email. "
            "Account lockouts after 5 failed attempts; contact support to unlock."
        ),
        "source": "Citizen Service Guide 2024",
    },
    {
        "id": "service-guide-002",
        "category": "service_guide",
        "title": "Submitting a New Application Online",
        "content": (
            "To submit a new application: "
            "(1) Log in to the Citizen Portal. "
            "(2) Select New Application from the dashboard. "
            "(3) Choose the application type and complete all required fields. "
            "(4) Upload supporting documents (PDF or JPG, max 10 MB each). "
            "(5) Pay the applicable fee via credit card, debit card, or BPay. "
            "(6) Download your acknowledgement receipt - keep this as proof of submission."
        ),
        "source": "Citizen Service Guide 2024",
    },
    {
        "id": "service-guide-003",
        "category": "service_guide",
        "title": "Checking Application Status",
        "content": (
            "Check your application status at any time via the Citizen Portal under My Applications. "
            "You can also use the Application ID lookup at portal.gov/track (no login required). "
            "Status updates are sent via email and SMS at key milestones. "
            "For applications older than 6 months, contact the relevant department directly."
        ),
        "source": "Citizen Service Guide 2024",
    },
    # ── Payment Policy ───────────────────────────────────────────────────────
    {
        "id": "payment-policy-001",
        "category": "payment_policy",
        "title": "Accepted Payment Methods",
        "content": (
            "Government services accept: Visa, Mastercard, AMEX (online); "
            "EFTPOS (in person); BPay (reference number on invoice); "
            "Bank transfer (EFT) for amounts over $1,000. "
            "Cheques are no longer accepted as of 1 January 2024. "
            "Payment plans are available for fees over $500 - contact the Billing Office."
        ),
        "source": "Payment Policy Document 2024",
    },
    {
        "id": "payment-policy-002",
        "category": "payment_policy",
        "title": "Requesting a Payment Receipt",
        "content": (
            "Payment receipts are automatically emailed to your registered address within 2 business hours of payment. "
            "To request a duplicate receipt: log in to the Citizen Portal -> Payments -> Download Receipt. "
            "Receipts are available for 7 years after payment. "
            "For tax invoice purposes, receipts include GST breakdown, ABN, and invoice number. "
            "Lost receipts older than 7 years: contact the Billing Office with your payment details."
        ),
        "source": "Payment Policy Document 2024",
    },
    {
        "id": "payment-policy-003",
        "category": "payment_policy",
        "title": "Refund Policy",
        "content": (
            "Refunds are processed within 10 business days of approval. "
            "Refunds are available for: cancelled applications (before processing), "
            "overpayments, and successful fee exemption claims. "
            "Refunds are returned to the original payment method. "
            "To request a refund, complete Form RF-50 and attach your payment receipt. "
            "Processing fees of $25 may apply for international bank transfers."
        ),
        "source": "Payment Policy Document 2024",
    },
    {
        "id": "payment-policy-004",
        "category": "payment_policy",
        "title": "Late Payment and Penalties",
        "content": (
            "Permit renewals submitted after the expiry date incur a late fee of 25% of the standard fee. "
            "Unpaid fees accrue interest at 8% per annum after 30 days. "
            "Non-payment may result in suspension of the associated permit or licence. "
            "Payment plans are available - contact the Billing Office within 10 days of the due date."
        ),
        "source": "Payment Policy Document 2024",
    },
    # ── Complaint Handling ───────────────────────────────────────────────────
    {
        "id": "complaint-policy-001",
        "category": "complaint_handling",
        "title": "How to Lodge a Complaint",
        "content": (
            "Complaints can be lodged via: "
            "(1) Online: Citizen Portal -> Lodge a Complaint. "
            "(2) Phone: 1800-CITIZEN (Mon-Fri, 8 AM-5 PM). "
            "(3) In writing: Citizen Services, GPO Box 1000. "
            "Provide: your name, contact details, application/permit ID (if relevant), "
            "description of the issue, and your preferred resolution. "
            "You will receive a complaint ID within 1 business day of submission."
        ),
        "source": "Complaint Handling Policy v2.1",
    },
    {
        "id": "complaint-policy-002",
        "category": "complaint_handling",
        "title": "Complaint Resolution Timeframes",
        "content": (
            "Standard complaints: acknowledged within 2 business days, resolved within 15 business days. "
            "Complex complaints: up to 30 business days. "
            "Urgent complaints (involving safety, hardship, or significant financial impact): escalated within 24 hours. "
            "If unresolved within the timeframe, the complaint is automatically escalated to a senior officer. "
            "Check your complaint status using your Complaint ID at portal.gov/complaints."
        ),
        "source": "Complaint Handling Policy v2.1",
    },
    {
        "id": "complaint-policy-003",
        "category": "complaint_handling",
        "title": "External Complaint Escalation",
        "content": (
            "If you are unsatisfied with the resolution, you may escalate to: "
            "Ombudsman Office: ombudsman.gov - handles complaints about government process and procedure. "
            "Administrative Appeals Tribunal: aat.gov - reviews administrative decisions. "
            "These external bodies operate independently of the department. "
            "Escalating externally does not forfeit your right to internal resolution."
        ),
        "source": "Complaint Handling Policy v2.1",
    },
    {
        "id": "complaint-policy-004",
        "category": "complaint_handling",
        "title": "Complaint Categories and Priority",
        "content": (
            "Complaint categories: Service Delay, Billing Error, Staff Conduct, Policy Dispute, Technical Issue, Other. "
            "Priority levels: Low (general enquiry), Medium (service impacted), "
            "High (significant impact), Critical (safety or financial emergency). "
            "Critical complaints are assigned to a dedicated case manager within 2 hours. "
            "Priority is assessed at intake and may be revised as more information becomes available."
        ),
        "source": "Complaint Handling Policy v2.1",
    },
    {
        "id": "complaint-policy-005",
        "category": "complaint_handling",
        "title": "Complaint Outcomes and Remedies",
        "content": (
            "Possible complaint outcomes: Full upheld (remedy provided), Partial upheld, Not upheld (no action). "
            "Remedies may include: apology, policy review, fee waiver, expedited processing, compensation. "
            "All complaint outcomes are documented and used to improve service delivery. "
            "You will receive a written outcome letter within 2 business days of the decision."
        ),
        "source": "Complaint Handling Policy v2.1",
    },
]


# ---------------------------------------------------------------------------
# ChromaDB wrapper with in-memory keyword fallback
# ---------------------------------------------------------------------------

class ChromaService:
    """Vector store backed by ChromaDB; falls back to keyword search."""

    def __init__(self) -> None:
        self._collection = None
        self._init_chroma()

    def _init_chroma(self) -> None:
        try:
            import chromadb  # type: ignore
            from chromadb.config import Settings as ChromaSettings

            db_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "data", "chroma_db",
            )
            os.makedirs(db_path, exist_ok=True)

            client = chromadb.PersistentClient(
                path=db_path,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = client.get_or_create_collection(
                name="citizen_services",
                metadata={"hnsw:space": "cosine"},
            )

            existing_ids = set(self._collection.get()["ids"])
            new_docs = [d for d in _DOCUMENTS if d["id"] not in existing_ids]
            if new_docs:
                self._collection.add(
                    ids=[d["id"] for d in new_docs],
                    documents=[d["content"] for d in new_docs],
                    metadatas=[
                        {"title": d["title"], "category": d["category"], "source": d["source"]}
                        for d in new_docs
                    ],
                )
                logger.info("chroma.seeded", count=len(new_docs))
            else:
                logger.info("chroma.ready", docs=len(existing_ids))

        except Exception as exc:
            logger.warning("chroma.unavailable", reason=str(exc), fallback="keyword")
            self._collection = None

    # ── Public API ───────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5, category: str | None = None) -> list[dict[str, Any]]:
        if self._collection is not None:
            return self._vector_search(query, top_k, category)
        return self._keyword_search(query, top_k, category)

    def search_by_category(self, category: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self._collection is not None:
            try:
                result = self._collection.get(where={"category": category})
                docs = []
                for i, doc_id in enumerate(result["ids"]):
                    meta = result["metadatas"][i]
                    docs.append({
                        "id": doc_id,
                        "title": meta.get("title", ""),
                        "category": meta.get("category", ""),
                        "source": meta.get("source", ""),
                        "content": result["documents"][i],
                        "score": 1.0,
                    })
                return docs[:top_k]
            except Exception:
                pass
        return [d for d in _DOCUMENTS if d["category"] == category][:top_k]

    # ── Internal ─────────────────────────────────────────────────────────────

    def _vector_search(self, query: str, top_k: int, category: str | None) -> list[dict[str, Any]]:
        try:
            kwargs: dict[str, Any] = {
                "query_texts": [query],
                "n_results": min(top_k, len(_DOCUMENTS)),
            }
            if category:
                kwargs["where"] = {"category": category}
            result = self._collection.query(**kwargs)
            docs = []
            for i, doc_id in enumerate(result["ids"][0]):
                meta = result["metadatas"][0][i]
                distance = result["distances"][0][i] if result.get("distances") else 0.0
                docs.append({
                    "id": doc_id,
                    "title": meta.get("title", ""),
                    "category": meta.get("category", ""),
                    "source": meta.get("source", ""),
                    "content": result["documents"][0][i],
                    "score": round(1.0 - distance, 4),
                })
            return docs
        except Exception as exc:
            logger.warning("chroma.vector_search_failed", error=str(exc))
            return self._keyword_search(query, top_k, category)

    def _keyword_search(self, query: str, top_k: int, category: str | None) -> list[dict[str, Any]]:
        tokens = set(query.lower().split())
        scored: list[tuple[float, dict]] = []
        for doc in _DOCUMENTS:
            if category and doc["category"] != category:
                continue
            text = (doc["title"] + " " + doc["content"]).lower()
            hits = sum(1 for t in tokens if t in text)
            if hits:
                scored.append((hits, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [
            {**d, "score": round(h / max(len(tokens), 1), 4)}
            for h, d in scored[:top_k]
        ]
        return results if results else _DOCUMENTS[:top_k]


# Singleton
_chroma_service: ChromaService | None = None


def get_chroma_service() -> ChromaService:
    global _chroma_service
    if _chroma_service is None:
        _chroma_service = ChromaService()
    return _chroma_service
