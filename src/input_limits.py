"""Central limits for untrusted inbound data (DoS and log-injection hardening)."""

from __future__ import annotations

import re

# Session identifiers come from webhooks (phone, opaque id). Restrict to safe ASCII.
MAX_SESSION_ID_LEN = 256
_SESSION_ID_RE = re.compile(r"^[\x20-\x7E]{1,256}$")

# Inbound user text per turn (ReDoS / memory pressure mitigation).
MAX_USER_MESSAGE_CHARS = 12_000

# Persisted message bodies and lead-field strings.
MAX_PERSISTED_MESSAGE_CHARS = 16_000
MAX_LEAD_FIELD_CHARS = 512

# Cap stored timeline when loading potentially hostile JSON files.
MAX_STORED_MESSAGES = 1_000


def validate_session_id(session_id: str) -> str:
    """Return session_id if safe; raise ValueError otherwise."""
    if not isinstance(session_id, str):
        raise TypeError("session_id must be a string")
    if len(session_id) == 0 or len(session_id) > MAX_SESSION_ID_LEN:
        raise ValueError("session_id has invalid length")
    if _SESSION_ID_RE.fullmatch(session_id) is None:
        raise ValueError("session_id contains invalid characters")
    return session_id


def clamp_user_message(text: str) -> str:
    """Truncate inbound user text to a bounded length."""
    if not isinstance(text, str):
        raise TypeError("user message must be a string")
    if len(text) <= MAX_USER_MESSAGE_CHARS:
        return text
    return text[:MAX_USER_MESSAGE_CHARS]


def clamp_persisted_text(text: str, *, max_chars: int = MAX_PERSISTED_MESSAGE_CHARS) -> str:
    """Truncate text before persistence or outbound echo of stored fields."""
    if len(text) > max_chars:
        return text[:max_chars]
    return text
