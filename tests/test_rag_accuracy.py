"""Accuracy tests for deterministic local KB answering."""

from __future__ import annotations

from src.rag import QueryType, answer_from_kb, load_knowledge_base


def test_kb_loads_canonical_facts() -> None:
    kb = load_knowledge_base()

    assert kb.company == "AutoStream"
    assert kb.product == "Automated video editing SaaS for content creators"
    assert len(kb.plans) == 2
    assert kb.policies.refund_policy == "No refunds after 7 days"
    assert kb.policies.support_policy == "24/7 support available only on Pro plan"


def test_pro_plan_answer_is_exact() -> None:
    result = answer_from_kb("Can you share the Pro plan features and pricing?")

    assert result.query_type == QueryType.PLAN_DETAILS
    assert result.answer == "Here is the Pro Plan: $79/month, Unlimited videos, 4K resolution, AI captions."


def test_basic_plan_answer_is_exact() -> None:
    result = answer_from_kb("What does the Basic plan include?")

    assert result.query_type == QueryType.PLAN_DETAILS
    assert result.answer == "Here is the Basic Plan: $29/month, 10 videos/month, 720p resolution."


def test_refund_policy_answer_is_exact() -> None:
    result = answer_from_kb("What is your refund policy?")

    assert result.query_type == QueryType.REFUND_POLICY
    assert result.answer == "AutoStream policy: No refunds after 7 days."


def test_support_policy_answer_is_exact() -> None:
    result = answer_from_kb("Do you offer 24/7 support?")

    assert result.query_type == QueryType.SUPPORT_POLICY
    assert result.answer == "AutoStream support policy: 24/7 support available only on Pro plan."


def test_plan_comparison_answer_is_exact() -> None:
    result = answer_from_kb("Compare Basic vs Pro plans")

    assert result.query_type == QueryType.PLAN_COMPARISON
    assert (
        result.answer
        == "Plan comparison: Basic Plan -> $29/month, 10 videos/month, 720p resolution. "
        "Pro Plan -> $79/month, Unlimited videos, 4K resolution, AI captions."
    )


def test_general_pricing_question_returns_both_plans() -> None:
    result = answer_from_kb("What are your pricing plans?")

    assert result.query_type == QueryType.PLAN_DETAILS
    assert (
        result.answer
        == "Current plans: Here is the Basic Plan: $29/month, 10 videos/month, 720p resolution. "
        "Here is the Pro Plan: $79/month, Unlimited videos, 4K resolution, AI captions."
    )


def test_unknown_question_returns_safe_guidance() -> None:
    result = answer_from_kb("What are your office hours in Tokyo?")

    assert result.query_type == QueryType.UNKNOWN
    assert result.answer == (
        "I can answer pricing, features, and policies from our local knowledge base. "
        "Ask about the Basic Plan, Pro Plan, refund policy, or support policy."
    )
