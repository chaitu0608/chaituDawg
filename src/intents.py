"""Intent classification module for the AutoStream conversational agent.

Phase 2 scope:
- Deterministic rule-based classification first.
- Optional LLM fallback only for ambiguous inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Optional, Protocol, Sequence


class IntentLabel(str, Enum):
    """Supported intent labels for routing."""

    CASUAL_GREETING = "casual_greeting"
    PRODUCT_OR_PRICING_INQUIRY = "product_or_pricing_inquiry"
    HIGH_INTENT_LEAD = "high_intent_lead"


@dataclass(frozen=True)
class IntentResult:
    """Normalized intent response for downstream routing."""

    label: IntentLabel
    confidence: float
    method: str
    matched_terms: tuple[str, ...]


class LLMIntentFallback(Protocol):
    """Optional fallback interface used only when rule-based routing is ambiguous."""

    def classify_intent(self, text: str, labels: Sequence[str]) -> tuple[str, float]:
        """Return a (label, confidence) tuple for the provided text."""


# High-intent phrases are intentionally explicit, including required trigger phrases.
HIGH_INTENT_TERMS: tuple[str, ...] = (
    "i want to try",
    "sign me up",
    "start",
    "subscribe",
    "get started",
    "start pro plan",
    "start basic plan",
    "ready to buy",
    "upgrade",
    "book a demo",
)

INQUIRY_TERMS: tuple[str, ...] = (
    "price",
    "pricing",
    "plan",
    "plans",
    "feature",
    "features",
    "policy",
    "policies",
    "refund",
    "support",
    "resolution",
    "limit",
    "limits",
    "compare",
    "comparison",
)

GREETING_TERMS: tuple[str, ...] = (
    "hello",
    "hi",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "yo",
)


class IntentClassifier:
    """Hybrid classifier: deterministic rules first, optional LLM fallback second."""

    def __init__(self, llm_fallback: Optional[LLMIntentFallback] = None) -> None:
        self._llm_fallback = llm_fallback

    def classify(self, user_text: str) -> IntentResult:
        """Classify input text into one of the supported intent labels."""
        normalized_text = _normalize_text(user_text)

        # Empty messages are treated as lightweight greetings to keep flow friendly.
        if not normalized_text:
            return IntentResult(
                label=IntentLabel.CASUAL_GREETING,
                confidence=0.55,
                method="rule_based_default",
                matched_terms=(),
            )

        rule_result = _classify_with_rules(normalized_text)
        if rule_result is not None:
            return rule_result

        if self._llm_fallback is not None:
            raw_label, raw_confidence = self._llm_fallback.classify_intent(
                text=user_text,
                labels=[label.value for label in IntentLabel],
            )
            parsed_label = _parse_label(raw_label)
            if parsed_label is not None:
                return IntentResult(
                    label=parsed_label,
                    confidence=_clamp_confidence(raw_confidence),
                    method="llm_fallback",
                    matched_terms=(),
                )

        return IntentResult(
            label=IntentLabel.CASUAL_GREETING,
            confidence=0.51,
            method="rule_based_default",
            matched_terms=(),
        )


def _classify_with_rules(normalized_text: str) -> Optional[IntentResult]:
    high_intent_matches = _find_matches(normalized_text, HIGH_INTENT_TERMS)
    if high_intent_matches:
        return IntentResult(
            label=IntentLabel.HIGH_INTENT_LEAD,
            confidence=0.97,
            method="rule_based",
            matched_terms=high_intent_matches,
        )

    inquiry_matches = _find_matches(normalized_text, INQUIRY_TERMS)
    if inquiry_matches:
        return IntentResult(
            label=IntentLabel.PRODUCT_OR_PRICING_INQUIRY,
            confidence=0.94,
            method="rule_based",
            matched_terms=inquiry_matches,
        )

    greeting_matches = _find_matches(normalized_text, GREETING_TERMS)
    if greeting_matches:
        return IntentResult(
            label=IntentLabel.CASUAL_GREETING,
            confidence=0.9,
            method="rule_based",
            matched_terms=greeting_matches,
        )

    return None


def _find_matches(normalized_text: str, terms: Sequence[str]) -> tuple[str, ...]:
    matches: list[str] = []
    for term in terms:
        # Apply whole-word style matching so short tokens (for example "hi") do
        # not accidentally match inside unrelated words (for example "this").
        pattern = r"\b" + re.escape(term).replace(r"\ ", r"\s+") + r"\b"
        if re.search(pattern, normalized_text):
            matches.append(term)
    return tuple(matches)


def _parse_label(raw_label: str) -> Optional[IntentLabel]:
    for label in IntentLabel:
        if raw_label == label.value:
            return label
    return None


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, value))
