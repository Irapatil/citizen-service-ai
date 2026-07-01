from __future__ import annotations

from typing import Any

import structlog
from langchain_core.tools import tool

logger = structlog.get_logger(__name__)


@tool
def retrieve_documents(query: str, top_k: int = 5) -> dict[str, Any]:
    """
    Semantically search the government knowledge base for documents relevant to the query.

    Args:
        query: The user's question or search phrase.
        top_k: Maximum number of documents to return (default 5).

    Returns:
        A list of matching documents with content, source, and relevance score.
    """
    try:
        from backend.services.chroma_service import get_chroma_service
        service = get_chroma_service()
        docs = service.search(query, top_k=top_k)
        return {
            "success": True,
            "query": query,
            "documents": docs,
            "count": len(docs),
            "backend": "chromadb" if service._collection is not None else "keyword",
        }
    except Exception as exc:
        logger.warning("retrieve_documents.error", error=str(exc))
        return {
            "success": False,
            "query": query,
            "documents": [],
            "count": 0,
            "error": str(exc),
        }


@tool
def search_policies(topic: str) -> dict[str, Any]:
    """
    Search government policy documents for a specific topic.

    Args:
        topic: The policy topic to search (e.g. 'permit renewal', 'complaint handling').

    Returns:
        Matching policy documents with summaries, sources, and effective dates.
    """
    try:
        from backend.services.chroma_service import get_chroma_service
        service = get_chroma_service()

        # Determine the most relevant category based on topic keywords
        topic_lower = topic.lower()
        category_hints = {
            "permit": "permit_renewal",
            "renewal": "permit_renewal",
            "licence": "permit_renewal",
            "complaint": "complaint_handling",
            "escalat": "complaint_handling",
            "payment": "payment_policy",
            "receipt": "payment_policy",
            "refund": "payment_policy",
            "fee": "payment_policy",
            "billing": "payment_policy",
            "faq": "faq",
            "hours": "faq",
            "contact": "faq",
            "service": "service_guide",
            "portal": "service_guide",
            "submit": "service_guide",
            "application": "service_guide",
        }

        category = None
        for keyword, cat in category_hints.items():
            if keyword in topic_lower:
                category = cat
                break

        docs = service.search(topic, top_k=5, category=category)

        # Format as policy-style response
        policies = [
            {
                "id": d["id"],
                "title": d["title"],
                "summary": d["content"][:300],
                "source": d["source"],
                "category": d["category"],
                "score": d.get("score", 0.0),
            }
            for d in docs
        ]

        return {
            "success": True,
            "topic": topic,
            "policies": policies,
            "count": len(policies),
        }
    except Exception as exc:
        logger.warning("search_policies.error", error=str(exc))
        return {
            "success": False,
            "topic": topic,
            "policies": [],
            "count": 0,
            "error": str(exc),
        }
