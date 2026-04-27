"""Tooling and extraction helpers for lead qualification.

Phase 5 scope:
- Validate lead fields.
- Normalize user-provided platform values.
- Execute a mocked lead-capture tool only after full slot completion.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)
EMAIL_TOKEN_PATTERN = re.compile(r"([A-Za-z0-9._%+-]{1,64})@([A-Za-z0-9.-]+\.[A-Za-z]{2,})")

# Phrases that indicate signup intent, not a person's name (word-boundary matched).
_HIGH_INTENT_NAME_FALSE_POSITIVE_TERMS: tuple[str, ...] = (
    "i want to try",
    "sign me up",
    "get started",
    "subscribe",
    "start pro plan",
    "start basic plan",
    "ready to buy",
    "upgrade",
    "book a demo",
    "get started today",
    "start now",
    "lets start",
    "let's start",
    "start",
    "resume signup",
    "continue signup",
    "continue sign up",
)

EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")

PLATFORM_ALIASES: dict[str, str] = {
    "youtube": "YouTube",
    "yt": "YouTube",
    "instagram": "Instagram",
    "insta": "Instagram",
    "tiktok": "TikTok",
    "tik tok": "TikTok",
    "linkedin": "LinkedIn",
    "twitch": "Twitch",
}


@dataclass(frozen=True)
class LeadCapturePayload:
    """Validated lead payload for downstream tool execution."""

    name: str
    email: str
    platform: str


def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """Mocked lead capture tool used for controlled integration tests."""
    message = f"Lead captured successfully: {name}, {email}, {platform}"
    logger.info("%s", mask_email_in_text(message))
    return message


def is_valid_email(email: str) -> bool:
    """Return True when email meets a basic production-safe validation check."""
    return EMAIL_PATTERN.fullmatch(email.strip()) is not None


def extract_email(text: str) -> Optional[str]:
    """Extract and validate the first email-like token from text."""
    candidate = text.strip().strip(".,;:!?")
    if is_valid_email(candidate):
        return candidate.lower()

    # Fallback extraction for mixed text messages.
    token_pattern = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    match = token_pattern.search(text)
    if match and is_valid_email(match.group(0)):
        return match.group(0).lower()

    return None


def normalize_platform(value: str) -> Optional[str]:
    """Normalize common platform aliases to canonical values."""
    normalized = " ".join(value.strip().lower().split())
    return PLATFORM_ALIASES.get(normalized)


def extract_platform(text: str) -> Optional[str]:
    """Extract a supported platform mention from free-form text."""
    normalized_text = " ".join(text.strip().lower().split())
    for alias, canonical in PLATFORM_ALIASES.items():
        pattern = r"\b" + re.escape(alias).replace(r"\ ", r"\s+") + r"\b"
        if re.search(pattern, normalized_text):
            return canonical
    return None


def extract_name(text: str) -> Optional[str]:
    """Extract a probable contact name from free-form user input."""
    normalized = " ".join(text.strip().split())
    if not normalized:
        return None

    explicit_patterns = (
        r"\bmy name is\s+([A-Za-z][A-Za-z'\- ]{0,48})$",
        r"\bi am\s+([A-Za-z][A-Za-z'\- ]{0,48})$",
        r"\bi'm\s+([A-Za-z][A-Za-z'\- ]{0,48})$",
    )
    for pattern in explicit_patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            return _clean_name(match.group(1))

    if _signals_high_intent_chatter(normalized):
        return None

    # If user only sends a short name-like message, accept it as a name.
    if _looks_like_plain_name(normalized):
        return _clean_name(normalized)

    return None


def validate_lead_payload(name: str, email: str, platform: str) -> tuple[bool, str]:
    """Validate lead fields before tool invocation."""
    cleaned_name = _clean_name(name)
    if not cleaned_name:
        return False, "name"

    if not is_valid_email(email):
        return False, "email"

    canonical_platform = normalize_platform(platform)
    if canonical_platform is None:
        return False, "platform"

    return True, ""


def normalize_lead_payload(name: str, email: str, platform: str) -> LeadCapturePayload:
    """Return canonical lead payload values used by validation and tool calls."""
    cleaned_name = _clean_name(name) or ""
    cleaned_email = email.strip().lower()
    canonical_platform = normalize_platform(platform)
    cleaned_platform = canonical_platform or platform.strip()
    return LeadCapturePayload(
        name=cleaned_name,
        email=cleaned_email,
        platform=cleaned_platform,
    )


def supported_platforms() -> tuple[str, ...]:
    """Return canonical platform options in stable order."""
    return ("YouTube", "Instagram", "TikTok", "LinkedIn", "Twitch")


def mask_email_in_text(text: str) -> str:
    """Mask email usernames in logs and snapshots to reduce PII exposure."""
    def _replace(match: re.Match[str]) -> str:
        username = match.group(1)
        domain = match.group(2)
        if len(username) <= 2:
            masked = "*" * len(username)
        else:
            masked = username[0] + ("*" * (len(username) - 2)) + username[-1]
        return f"{masked}@{domain}"

    return EMAIL_TOKEN_PATTERN.sub(_replace, text)


def _signals_high_intent_chatter(text: str) -> bool:
    """True when the line reads like a signup CTA, not a contact name."""
    lowered = text.lower()
    for term in _HIGH_INTENT_NAME_FALSE_POSITIVE_TERMS:
        pattern = r"\b" + re.escape(term).replace(r"\ ", r"\s+") + r"\b"
        if re.search(pattern, lowered):
            return True
    return False


def _looks_like_plain_name(value: str) -> bool:
    if "@" in value:
        return False

    if any(char.isdigit() for char in value):
        return False

    words = value.split()
    if not 1 <= len(words) <= 4:
        return False

    blocked_tokens = {
        "pricing",
        "plan",
        "start",
        "subscribe",
        "support",
        "refund",
        "youtube",
        "instagram",
        "tiktok",
        "linkedin",
        "twitch",
        "resume",
        "signup",
    }
    return not any(word.lower() in blocked_tokens for word in words)


def _clean_name(value: str) -> Optional[str]:
    stripped = " ".join(value.strip().split())
    stripped = stripped.strip(".,;:!?")
    if not stripped:
        return None
    return " ".join(token.capitalize() for token in stripped.split())
