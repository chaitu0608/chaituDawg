"""Unit tests for deterministic + fallback intent classification."""

from __future__ import annotations

import pytest  # pyright: ignore[reportMissingImports]

from src.intents import IntentClassifier, IntentLabel


class DummyFallback:
    """Simple test double for the optional LLM fallback."""

    def __init__(self, label: str, confidence: float) -> None:
        self.label = label
        self.confidence = confidence
        self.calls = 0

    def classify_intent(self, text: str, labels: list[str]) -> tuple[str, float]:
        self.calls += 1
        assert isinstance(text, str)
        assert len(labels) == 3
        return self.label, self.confidence


@pytest.mark.parametrize(
    "message",
    [
        "I want to try this today",
        "Sign me up please",
        "Let's start now",
        "I want to subscribe immediately",
        "Can we get started?",
    ],
)
def test_high_intent_phrases_are_detected(message: str) -> None:
    classifier = IntentClassifier()

    result = classifier.classify(message)

    assert result.label == IntentLabel.HIGH_INTENT_LEAD
    assert result.method == "rule_based"
    assert result.confidence >= 0.9


def test_product_or_pricing_inquiry_is_detected() -> None:
    classifier = IntentClassifier()

    result = classifier.classify("What is the price of the Pro plan and what features are included?")

    assert result.label == IntentLabel.PRODUCT_OR_PRICING_INQUIRY
    assert result.method == "rule_based"


def test_casual_greeting_is_detected() -> None:
    classifier = IntentClassifier()

    result = classifier.classify("Hey team")

    assert result.label == IntentLabel.CASUAL_GREETING
    assert result.method == "rule_based"


def test_ambiguous_text_uses_llm_fallback() -> None:
    fallback = DummyFallback("product_or_pricing_inquiry", 0.78)
    classifier = IntentClassifier(llm_fallback=fallback)

    result = classifier.classify("I am exploring options")

    assert fallback.calls == 1
    assert result.label == IntentLabel.PRODUCT_OR_PRICING_INQUIRY
    assert result.method == "llm_fallback"
    assert result.confidence == 0.78


def test_rules_prevent_fallback_for_clear_intent() -> None:
    fallback = DummyFallback("casual_greeting", 0.33)
    classifier = IntentClassifier(llm_fallback=fallback)

    result = classifier.classify("pricing details for the basic plan")

    assert fallback.calls == 0
    assert result.label == IntentLabel.PRODUCT_OR_PRICING_INQUIRY
    assert result.method == "rule_based"


def test_invalid_fallback_label_reverts_to_safe_default() -> None:
    fallback = DummyFallback("unknown_label", 0.99)
    classifier = IntentClassifier(llm_fallback=fallback)

    result = classifier.classify("This is ambiguous text")

    assert fallback.calls == 1
    assert result.label == IntentLabel.CASUAL_GREETING
    assert result.method == "rule_based_default"
