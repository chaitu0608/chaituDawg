"""Lead-flow helper utilities extracted from agent orchestration."""

from __future__ import annotations

import hashlib

from src.memory import SessionState
from src.tools import LeadCapturePayload, normalize_platform, supported_platforms


def next_missing_field(state: SessionState) -> str:
    if not state.lead_name:
        return "name"
    if not state.lead_email:
        return "email"
    if not state.lead_platform:
        return "platform"
    return ""


def slot_summary(state: SessionState) -> str:
    name = state.lead_name or "missing"
    email = state.lead_email or "missing"
    platform = state.lead_platform or "missing"
    return f"Collected so far - name: {name}, email: {email}, platform: {platform}."


def payload_from_state(state: SessionState) -> LeadCapturePayload:
    platform = normalize_platform(state.lead_platform or "")
    if not state.lead_name or not state.lead_email or not platform:
        raise ValueError("Lead payload requested before all fields were complete")

    return LeadCapturePayload(
        name=state.lead_name,
        email=state.lead_email,
        platform=platform,
    )


def lead_capture_fingerprint(payload: LeadCapturePayload) -> str:
    normalized = f"{payload.name.lower()}|{payload.email.lower()}|{payload.platform.lower()}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def slot_retry_count(state: SessionState, slot_name: str) -> int:
    if slot_name == "email":
        return state.email_retry_count
    if slot_name == "platform":
        return state.platform_retry_count
    return 0


def slot_resume_prompt(next_field: str) -> str:
    if next_field == "name":
        return "When you're ready, we can continue signup. What is your name?"
    if next_field == "email":
        return "When you're ready, we can continue signup. What is the best email to reach you?"
    if next_field == "platform":
        requested_platforms = ", ".join(supported_platforms())
        return (
            "When you're ready, we can continue signup. "
            f"Which platform do you primarily create on? ({requested_platforms})"
        )
    return ""


def is_pause_or_uncertain_message(text: str) -> bool:
    normalized = " ".join(text.strip().lower().split())
    if normalized in {"skip lead", "skip", "maybe later", "not sure", "later"}:
        return True
    return any(
        phrase in normalized
        for phrase in (
            "maybe later",
            "not sure",
            "we can do this later",
            "continue later",
            "skip lead",
            "skip this",
        )
    )


def is_resume_request(text: str) -> bool:
    normalized = " ".join(text.strip().lower().split())
    if normalized in {"resume signup", "resume", "continue signup"}:
        return True
    return any(
        phrase in normalized
        for phrase in (
            "resume signup",
            "continue signup",
            "continue sign up",
            "continue with signup",
        )
    )
