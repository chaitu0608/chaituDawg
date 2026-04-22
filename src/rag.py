"""Deterministic local RAG utilities for AutoStream.

Phase 3 scope:
- Load product facts from the local JSON knowledge base.
- Route pricing/feature/policy questions with deterministic retrieval.
- Return concise, structured answers without fabricating facts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import re
from typing import Any


DEFAULT_KB_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.json"


class QueryType(str, Enum):
    """Supported deterministic query intents for local KB answering."""

    PLAN_DETAILS = "plan_details"
    PLAN_COMPARISON = "plan_comparison"
    REFUND_POLICY = "refund_policy"
    SUPPORT_POLICY = "support_policy"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PlanRecord:
    """Canonical pricing and feature information for a plan."""

    name: str
    price: str
    limits: str
    resolution: str
    extras: tuple[str, ...]


@dataclass(frozen=True)
class PolicyRecord:
    """Canonical policy facts from the local KB."""

    refund_policy: str
    support_policy: str


@dataclass(frozen=True)
class KnowledgeBase:
    """Parsed immutable representation of the local knowledge base."""

    company: str
    product: str
    plans: tuple[PlanRecord, ...]
    policies: PolicyRecord


@dataclass(frozen=True)
class RAGResult:
    """Structured deterministic answer result."""

    answer: str
    query_type: QueryType
    matched_plan: str | None
    kb_context: tuple[str, ...]


def load_knowledge_base(kb_path: str | Path = DEFAULT_KB_PATH) -> KnowledgeBase:
    """Load and validate local KB JSON from disk."""
    kb_file = Path(kb_path)
    with kb_file.open("r", encoding="utf-8") as file_handle:
        raw_kb: dict[str, Any] = json.load(file_handle)

    _validate_raw_kb(raw_kb)

    plans = tuple(
        PlanRecord(
            name=plan["name"],
            price=plan["price"],
            limits=plan["limits"],
            resolution=plan["resolution"],
            extras=tuple(plan.get("extras", [])),
        )
        for plan in raw_kb["plans"]
    )

    policies = raw_kb["policies"]
    return KnowledgeBase(
        company=raw_kb["company"],
        product=raw_kb["product"],
        plans=plans,
        policies=PolicyRecord(
            refund_policy=policies["refund_policy"],
            support_policy=policies["support_policy"],
        ),
    )


def answer_from_kb(question: str, kb: KnowledgeBase | None = None) -> RAGResult:
    """Answer supported questions using deterministic retrieval from local KB facts."""
    knowledge_base = kb or load_knowledge_base()
    normalized_question = _normalize_text(question)

    query_type, matched_plan = _route_query(normalized_question, knowledge_base)

    if query_type == QueryType.REFUND_POLICY:
        answer = f"AutoStream policy: {knowledge_base.policies.refund_policy}."
        return RAGResult(
            answer=answer,
            query_type=query_type,
            matched_plan=None,
            kb_context=(knowledge_base.policies.refund_policy,),
        )

    if query_type == QueryType.SUPPORT_POLICY:
        answer = f"AutoStream support policy: {knowledge_base.policies.support_policy}."
        return RAGResult(
            answer=answer,
            query_type=query_type,
            matched_plan=None,
            kb_context=(knowledge_base.policies.support_policy,),
        )

    if query_type == QueryType.PLAN_COMPARISON:
        basic_plan = _require_plan(knowledge_base, "Basic Plan")
        pro_plan = _require_plan(knowledge_base, "Pro Plan")
        answer = (
            "Plan comparison: "
            f"{_format_plan_comparison_segment(basic_plan)} "
            f"{_format_plan_comparison_segment(pro_plan)}"
        )
        return RAGResult(
            answer=answer,
            query_type=query_type,
            matched_plan=None,
            kb_context=(
                _plan_context_line(basic_plan),
                _plan_context_line(pro_plan),
            ),
        )

    if query_type == QueryType.PLAN_DETAILS:
        if matched_plan is not None:
            plan = _require_plan(knowledge_base, matched_plan)
            return RAGResult(
                answer=_format_single_plan_answer(plan),
                query_type=query_type,
                matched_plan=plan.name,
                kb_context=(_plan_context_line(plan),),
            )

        # For broad pricing/feature questions, provide both plan facts from KB.
        basic_plan = _require_plan(knowledge_base, "Basic Plan")
        pro_plan = _require_plan(knowledge_base, "Pro Plan")
        answer = (
            "Current plans: "
            f"{_format_single_plan_answer(basic_plan)} "
            f"{_format_single_plan_answer(pro_plan)}"
        )
        return RAGResult(
            answer=answer,
            query_type=query_type,
            matched_plan=None,
            kb_context=(
                _plan_context_line(basic_plan),
                _plan_context_line(pro_plan),
            ),
        )

    return RAGResult(
        answer=(
            "I can answer pricing, features, and policies from our local knowledge base. "
            "Ask about the Basic Plan, Pro Plan, refund policy, or support policy."
        ),
        query_type=QueryType.UNKNOWN,
        matched_plan=None,
        kb_context=(),
    )


def _validate_raw_kb(raw_kb: dict[str, Any]) -> None:
    required_keys = ("company", "product", "plans", "policies")
    for key in required_keys:
        if key not in raw_kb:
            raise ValueError(f"Knowledge base is missing required key: {key}")

    if not isinstance(raw_kb["plans"], list) or len(raw_kb["plans"]) == 0:
        raise ValueError("Knowledge base plans must be a non-empty list")

    for index, plan in enumerate(raw_kb["plans"]):
        for key in ("name", "price", "limits", "resolution"):
            if key not in plan:
                raise ValueError(f"Plan at index {index} is missing required key: {key}")

    policies = raw_kb["policies"]
    for key in ("refund_policy", "support_policy"):
        if key not in policies:
            raise ValueError(f"Knowledge base policies are missing required key: {key}")


def _route_query(normalized_question: str, kb: KnowledgeBase) -> tuple[QueryType, str | None]:
    matched_plan = _detect_plan(normalized_question, kb)

    if _contains_any(normalized_question, ("refund", "refunds", "money back")):
        return QueryType.REFUND_POLICY, None

    if _contains_any(normalized_question, ("support", "24/7", "help desk", "customer support")):
        return QueryType.SUPPORT_POLICY, None

    asks_comparison = _contains_any(
        normalized_question,
        ("compare", "comparison", "difference", "vs", "versus", "better"),
    )
    if asks_comparison or _mentions_both_plans(normalized_question):
        return QueryType.PLAN_COMPARISON, None

    plan_detail_terms = (
        "price",
        "pricing",
        "plan",
        "plans",
        "feature",
        "features",
        "limit",
        "limits",
        "resolution",
        "720p",
        "4k",
        "captions",
    )
    if matched_plan is not None and _contains_any(normalized_question, plan_detail_terms):
        return QueryType.PLAN_DETAILS, matched_plan

    if _contains_any(normalized_question, plan_detail_terms):
        return QueryType.PLAN_DETAILS, None

    if matched_plan is not None:
        return QueryType.PLAN_DETAILS, matched_plan

    return QueryType.UNKNOWN, None


def _format_single_plan_answer(plan: PlanRecord) -> str:
    answer = f"Here is the {plan.name}: {plan.price}, {plan.limits}, {plan.resolution} resolution"
    if plan.extras:
        answer += f", {', '.join(plan.extras)}"
    return answer + "."


def _format_plan_comparison_segment(plan: PlanRecord) -> str:
    segment = f"{plan.name} -> {plan.price}, {plan.limits}, {plan.resolution} resolution"
    if plan.extras:
        segment += f", {', '.join(plan.extras)}"
    return segment + "."


def _plan_context_line(plan: PlanRecord) -> str:
    extras = ", ".join(plan.extras) if plan.extras else "none"
    return (
        f"{plan.name}: price={plan.price}; limits={plan.limits}; "
        f"resolution={plan.resolution}; extras={extras}"
    )


def _require_plan(kb: KnowledgeBase, plan_name: str) -> PlanRecord:
    for plan in kb.plans:
        if plan.name == plan_name:
            return plan
    raise ValueError(f"Plan not found in KB: {plan_name}")


def _detect_plan(normalized_question: str, kb: KnowledgeBase) -> str | None:
    basic_aliases = ("basic", "basic plan")
    pro_aliases = ("pro", "pro plan")

    if _contains_any(normalized_question, basic_aliases):
        return "Basic Plan"

    if _contains_any(normalized_question, pro_aliases):
        return "Pro Plan"

    for plan in kb.plans:
        if _contains_any(normalized_question, (plan.name.lower(),)):
            return plan.name

    return None


def _mentions_both_plans(normalized_question: str) -> bool:
    return _contains_any(normalized_question, ("basic", "basic plan")) and _contains_any(
        normalized_question,
        ("pro", "pro plan"),
    )


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    for term in terms:
        pattern = r"\b" + re.escape(term).replace(r"\ ", r"\s+") + r"\b"
        if re.search(pattern, text):
            return True
    return False


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())
