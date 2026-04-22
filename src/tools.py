"""Tooling and extraction helpers for lead qualification.

Phase 5 scope:
- Validate lead fields.
- Normalize user-provided platform values.
- Execute a mocked lead-capture tool only after full slot completion.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


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
    print(message)
    return message


def is_valid_email(email: str) -> bool:
    """Return True when email meets a basic production-safe validation check."""
    return EMAIL_PATTERN.fullmatch(email.strip()) is not None


def extract_email(text: str) -> str | None:
    """Extract and validate the first email-like token from text."""
    candidate = text.strip().strip(".,;:!?")
    if is_valid_email(candidate):
        return candidate

    # Fallback extraction for mixed text messages.
    token_pattern = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    match = token_pattern.search(text)
    if match and is_valid_email(match.group(0)):
        return match.group(0)

    return None


def normalize_platform(value: str) -> str | None:
    """Normalize common platform aliases to canonical values."""
    normalized = " ".join(value.strip().lower().split())
    return PLATFORM_ALIASES.get(normalized)


def extract_platform(text: str) -> str | None:
    """Extract a supported platform mention from free-form text."""
    normalized_text = " ".join(text.strip().lower().split())
    for alias, canonical in PLATFORM_ALIASES.items():
        pattern = r"\b" + re.escape(alias).replace(r"\ ", r"\s+") + r"\b"
        if re.search(pattern, normalized_text):
            return canonical
    return None


def extract_name(text: str) -> str | None:
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


def supported_platforms() -> tuple[str, ...]:
    """Return canonical platform options in stable order."""
    return ("YouTube", "Instagram", "TikTok", "LinkedIn", "Twitch")


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
    }
    return not any(word.lower() in blocked_tokens for word in words)


def _clean_name(value: str) -> str | None:
    stripped = " ".join(value.strip().split())
    stripped = stripped.strip(".,;:!?")
    if not stripped:
        return None
    return " ".join(token.capitalize() for token in stripped.split())
