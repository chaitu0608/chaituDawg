"""Phase 6 benchmark assertions for deploy-readiness thresholds."""

from __future__ import annotations

from src.evaluation import evaluate_all


def test_phase6_metrics_meet_thresholds() -> None:
    summary = evaluate_all()

    assert summary.overall_passed is True

    metrics = summary.metrics
    assert metrics["intent_classification_accuracy"].value >= 0.95
    assert metrics["rag_factual_accuracy"].value >= 1.0
    assert metrics["tool_call_precision"].value >= 1.0
    assert metrics["lead_slot_completion"].value >= 1.0
    assert metrics["memory_retention"].value >= 1.0


def test_phase6_summary_contains_required_metric_keys() -> None:
    summary = evaluate_all()

    expected_keys = {
        "intent_classification_accuracy",
        "rag_factual_accuracy",
        "tool_call_precision",
        "lead_slot_completion",
        "memory_retention",
    }
    assert set(summary.metrics.keys()) == expected_keys
