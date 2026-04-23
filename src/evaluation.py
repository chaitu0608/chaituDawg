"""Phase 6 evaluation utilities for benchmark metrics and reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TypedDict

from src.agent import AutoStreamAgent
from src.intents import IntentClassifier, IntentLabel
from src.memory import SimpleMemStore
from src.rag import QueryType, answer_from_kb


class _ToolCallPrecisionScenario(TypedDict):
    id: str
    messages: tuple[str, ...]
    expected_tool_calls: tuple[bool, ...]


@dataclass(frozen=True)
class MetricResult:
    """A single computed metric with target and pass/fail status."""

    name: str
    value: float
    target: float
    passed: bool
    details: dict[str, Any]


@dataclass(frozen=True)
class EvaluationSummary:
    """All phase-6 metrics grouped into one summary payload."""

    generated_at: str
    metrics: dict[str, MetricResult]
    overall_passed: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary payload."""
        return {
            "generated_at": self.generated_at,
            "overall_passed": self.overall_passed,
            "metrics": {name: asdict(metric) for name, metric in self.metrics.items()},
        }

    def to_markdown(self) -> str:
        """Render the summary as a concise markdown report."""
        lines = [
            "# Phase 6 Metrics Report",
            "",
            f"Generated at (UTC): {self.generated_at}",
            "",
            "| Metric | Value | Target | Passed |",
            "|---|---:|---:|:---:|",
        ]

        for metric_name, metric in self.metrics.items():
            lines.append(
                f"| {metric_name} | {_as_percent(metric.value)} | {_as_percent(metric.target)} | {'yes' if metric.passed else 'no'} |"
            )

        lines.extend(
            [
                "",
                f"Overall Passed: {'yes' if self.overall_passed else 'no'}",
                "",
                "## Details",
            ]
        )

        for metric_name, metric in self.metrics.items():
            lines.append(f"- {metric_name}: {metric.details}")

        return "\n".join(lines) + "\n"


def evaluate_all() -> EvaluationSummary:
    """Compute all required phase-6 metrics in a deterministic way."""
    metrics = {
        "intent_classification_accuracy": evaluate_intent_classification_accuracy(),
        "rag_factual_accuracy": evaluate_rag_factual_accuracy(),
        "tool_call_precision": evaluate_tool_call_precision(),
        "lead_slot_completion": evaluate_lead_slot_completion(),
        "memory_retention": evaluate_memory_retention(),
    }

    overall_passed = all(metric.passed for metric in metrics.values())
    return EvaluationSummary(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        metrics=metrics,
        overall_passed=overall_passed,
    )


def evaluate_intent_classification_accuracy() -> MetricResult:
    """Measure rule-first intent classifier quality on a synthetic dataset."""
    classifier = IntentClassifier()

    dataset: tuple[tuple[str, IntentLabel], ...] = (
        ("hello there", IntentLabel.CASUAL_GREETING),
        ("hey team", IntentLabel.CASUAL_GREETING),
        ("good morning", IntentLabel.CASUAL_GREETING),
        ("yo", IntentLabel.CASUAL_GREETING),
        ("hi", IntentLabel.CASUAL_GREETING),
        ("what are your pricing plans", IntentLabel.PRODUCT_OR_PRICING_INQUIRY),
        ("compare basic and pro plans", IntentLabel.PRODUCT_OR_PRICING_INQUIRY),
        ("do you offer 24/7 support", IntentLabel.PRODUCT_OR_PRICING_INQUIRY),
        ("what is your refund policy", IntentLabel.PRODUCT_OR_PRICING_INQUIRY),
        ("features in pro plan", IntentLabel.PRODUCT_OR_PRICING_INQUIRY),
        ("resolution for basic plan", IntentLabel.PRODUCT_OR_PRICING_INQUIRY),
        ("price for pro", IntentLabel.PRODUCT_OR_PRICING_INQUIRY),
        ("I want to try this", IntentLabel.HIGH_INTENT_LEAD),
        ("sign me up please", IntentLabel.HIGH_INTENT_LEAD),
        ("lets start now", IntentLabel.HIGH_INTENT_LEAD),
        ("subscribe me", IntentLabel.HIGH_INTENT_LEAD),
        ("get started today", IntentLabel.HIGH_INTENT_LEAD),
        ("ready to buy", IntentLabel.HIGH_INTENT_LEAD),
        ("book a demo", IntentLabel.HIGH_INTENT_LEAD),
    )

    mismatches: list[dict[str, str]] = []
    correct = 0

    for text, expected in dataset:
        predicted = classifier.classify(text).label
        if predicted == expected:
            correct += 1
        else:
            mismatches.append(
                {
                    "text": text,
                    "expected": expected.value,
                    "predicted": predicted.value,
                }
            )

    accuracy = correct / len(dataset)
    target = 0.95
    return MetricResult(
        name="intent_classification_accuracy",
        value=accuracy,
        target=target,
        passed=accuracy >= target,
        details={
            "samples": len(dataset),
            "correct": correct,
            "mismatches": mismatches,
        },
    )


def evaluate_rag_factual_accuracy() -> MetricResult:
    """Measure exact-answer correctness for local KB questions."""
    cases: tuple[tuple[str, QueryType, str], ...] = (
        (
            "Can you share the Pro plan features and pricing?",
            QueryType.PLAN_DETAILS,
            "Here is the Pro Plan: $79/month, Unlimited videos, 4K resolution, AI captions.",
        ),
        (
            "What does the Basic plan include?",
            QueryType.PLAN_DETAILS,
            "Here is the Basic Plan: $29/month, 10 videos/month, 720p resolution.",
        ),
        (
            "What is your refund policy?",
            QueryType.REFUND_POLICY,
            "AutoStream policy: No refunds after 7 days.",
        ),
        (
            "Do you offer 24/7 support?",
            QueryType.SUPPORT_POLICY,
            "AutoStream support policy: 24/7 support available only on Pro plan.",
        ),
        (
            "Compare Basic vs Pro plans",
            QueryType.PLAN_COMPARISON,
            "Plan comparison: Basic Plan -> $29/month, 10 videos/month, 720p resolution. Pro Plan -> $79/month, Unlimited videos, 4K resolution, AI captions.",
        ),
        (
            "What are your pricing plans?",
            QueryType.PLAN_DETAILS,
            "Current plans: Here is the Basic Plan: $29/month, 10 videos/month, 720p resolution. Here is the Pro Plan: $79/month, Unlimited videos, 4K resolution, AI captions.",
        ),
    )

    exact_matches = 0
    mismatches: list[dict[str, str]] = []

    for question, expected_type, expected_answer in cases:
        result = answer_from_kb(question)
        is_match = result.query_type == expected_type and result.answer == expected_answer
        if is_match:
            exact_matches += 1
        else:
            mismatches.append(
                {
                    "question": question,
                    "expected_type": expected_type.value,
                    "actual_type": result.query_type.value,
                    "expected_answer": expected_answer,
                    "actual_answer": result.answer,
                }
            )

    accuracy = exact_matches / len(cases)
    target = 1.0
    return MetricResult(
        name="rag_factual_accuracy",
        value=accuracy,
        target=target,
        passed=accuracy >= target,
        details={
            "samples": len(cases),
            "exact_matches": exact_matches,
            "mismatches": mismatches,
        },
    )


def evaluate_tool_call_precision() -> MetricResult:
    """Measure premature-tool-call avoidance across lead-flow turns."""
    scenarios: tuple[_ToolCallPrecisionScenario, ...] = (
        {
            "id": "partial_fields",
            "messages": ("Sign me up", "My name is Ava", "ava@example.com"),
            "expected_tool_calls": (False, False, False),
        },
        {
            "id": "complete_fields",
            "messages": ("I want to try", "My name is Leo", "leo@example.com", "YouTube"),
            "expected_tool_calls": (False, False, False, True),
        },
        {
            "id": "already_captured",
            "messages": (
                "Get started",
                "I am Nina",
                "nina@example.com",
                "Instagram",
                "Sign me up again",
            ),
            "expected_tool_calls": (False, False, False, True, False),
        },
    )

    tp = 0
    fp = 0
    fn = 0
    mismatches: list[dict[str, str]] = []

    for scenario in scenarios:
        with TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "simplemem_sessions.json"
            spy = _LeadCaptureSpy()
            agent = AutoStreamAgent(
                memory_store=SimpleMemStore(storage_path=storage_path),
                lead_capture_tool=spy,
            )

            session_id = f"precision-{scenario['id']}"
            for index, message in enumerate(scenario["messages"]):
                expected = scenario["expected_tool_calls"][index]
                actual = agent.handle_message(session_id, message).tool_called

                if expected and actual:
                    tp += 1
                elif not expected and actual:
                    fp += 1
                elif expected and not actual:
                    fn += 1

                if actual != expected:
                    mismatches.append(
                        {
                            "scenario": scenario["id"],
                            "turn": str(index + 1),
                            "message": message,
                            "expected": str(expected),
                            "actual": str(actual),
                        }
                    )

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    target = 1.0
    return MetricResult(
        name="tool_call_precision",
        value=precision,
        target=target,
        passed=precision >= target and fp == 0 and fn == 0,
        details={
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "mismatches": mismatches,
        },
    )


def evaluate_lead_slot_completion() -> MetricResult:
    """Measure completion reliability once all required slot values are provided."""
    completion_sequences = (
        ("I want to try", "My name is Maya", "maya@example.com", "YouTube"),
        ("Start pro plan", "I am Raj", "raj@example.com", "Instagram"),
        ("Subscribe", "Lina", "lina@example.com", "TikTok"),
    )

    successful = 0
    failures: list[dict[str, str]] = []

    for index, sequence in enumerate(completion_sequences):
        with TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "simplemem_sessions.json"
            spy = _LeadCaptureSpy()
            agent = AutoStreamAgent(
                memory_store=SimpleMemStore(storage_path=storage_path),
                lead_capture_tool=spy,
            )

            session_id = f"slot-completion-{index}"
            last_response = None
            for message in sequence:
                last_response = agent.handle_message(session_id, message)

            state = agent.inspect_session_state(session_id)
            captured_once = len(spy.calls) == 1
            called_on_completion = bool(last_response and last_response.tool_called)
            lead_captured_state = state.get("lead_status") == "captured"

            if captured_once and called_on_completion and lead_captured_state:
                successful += 1
            else:
                failures.append(
                    {
                        "session": session_id,
                        "captured_once": str(captured_once),
                        "called_on_completion": str(called_on_completion),
                        "lead_status": str(state.get("lead_status")),
                    }
                )

    completion_rate = successful / len(completion_sequences)
    target = 1.0
    return MetricResult(
        name="lead_slot_completion",
        value=completion_rate,
        target=target,
        passed=completion_rate >= target,
        details={
            "sessions": len(completion_sequences),
            "successful": successful,
            "failures": failures,
        },
    )


def evaluate_memory_retention() -> MetricResult:
    """Verify state restoration and context continuity over six turns."""
    with TemporaryDirectory() as temp_dir:
        storage_path = Path(temp_dir) / "simplemem_sessions.json"
        spy = _LeadCaptureSpy()
        store = SimpleMemStore(storage_path=storage_path)
        agent = AutoStreamAgent(memory_store=store, lead_capture_tool=spy)

        session_id = "memory-six-turn"
        turns = (
            "Get started",
            "My name is Aria",
            "aria@example.com",
            "YouTube",
            "Thanks",
            "What is the Pro plan price?",
        )
        for message in turns:
            agent.handle_message(session_id, message)

        restored_store = SimpleMemStore(storage_path=storage_path)
        restored_state = restored_store.load_state(session_id)
        recent_context = restored_store.restore_recent_context(session_id, max_turns=6)

        conditions = {
            "turn_count_at_least_six": restored_state.turn_count >= 6,
            "twelve_messages_stored": len(restored_state.messages) >= 12,
            "lead_name_persisted": restored_state.lead_name == "Aria",
            "lead_email_persisted": restored_state.lead_email == "aria@example.com",
            "lead_platform_persisted": restored_state.lead_platform == "YouTube",
            "lead_captured": restored_state.lead_status == "captured",
            "restored_recent_context_has_twelve_messages": len(recent_context) == 12,
        }

    passed = all(conditions.values())
    value = 1.0 if passed else 0.0
    target = 1.0
    return MetricResult(
        name="memory_retention",
        value=value,
        target=target,
        passed=passed,
        details=conditions,
    )


def _as_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


class _LeadCaptureSpy:
    """Minimal callable tool spy for evaluation scenarios."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, name: str, email: str, platform: str) -> str:
        self.calls.append((name, email, platform))
        return f"Lead captured successfully: {name}, {email}, {platform}"
