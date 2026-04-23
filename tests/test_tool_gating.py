"""Phase 5 tests for lead qualification and strict tool gating."""

from __future__ import annotations

import hashlib
from pathlib import Path
import time

from src.agent import AutoStreamAgent
from src.memory import SimpleMemStore


class LeadCaptureSpy:
    """Captures tool invocations for deterministic assertions."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, name: str, email: str, platform: str) -> str:
        self.calls.append((name, email, platform))
        return f"Lead captured successfully: {name}, {email}, {platform}"


def _build_agent(tmp_path: Path) -> tuple[AutoStreamAgent, LeadCaptureSpy]:
    spy = LeadCaptureSpy()
    memory = SimpleMemStore(storage_path=tmp_path / "simplemem_sessions.json")
    return AutoStreamAgent(memory_store=memory, lead_capture_tool=spy), spy


def test_tool_not_called_before_all_required_fields(tmp_path: Path) -> None:
    agent, spy = _build_agent(tmp_path)
    session_id = "partial-fields-session"

    first = agent.handle_message(session_id, "Sign me up")
    second = agent.handle_message(session_id, "My name is Ava")
    third = agent.handle_message(session_id, "ava@example.com")

    assert first.tool_called is False
    assert second.tool_called is False
    assert third.tool_called is False
    assert spy.calls == []
    assert "email: ava@example.com" in third.text
    assert "platform: missing" in third.text


def test_tool_called_when_all_fields_are_collected(tmp_path: Path) -> None:
    agent, spy = _build_agent(tmp_path)
    session_id = "complete-fields-session"

    agent.handle_message(session_id, "I want to try AutoStream")
    agent.handle_message(session_id, "My name is Maya")
    agent.handle_message(session_id, "maya@example.com")
    response = agent.handle_message(session_id, "YouTube")

    assert response.tool_called is True
    assert len(spy.calls) == 1
    assert spy.calls[0] == ("Maya", "maya@example.com", "YouTube")
    assert "Lead captured successfully: Maya, maya@example.com, YouTube" in response.text


def test_tool_not_called_twice_after_capture(tmp_path: Path) -> None:
    agent, spy = _build_agent(tmp_path)
    session_id = "one-shot-session"

    agent.handle_message(session_id, "Start pro plan")
    agent.handle_message(session_id, "I am Leo")
    agent.handle_message(session_id, "leo@example.com")
    captured = agent.handle_message(session_id, "instagram")
    follow_up = agent.handle_message(session_id, "Please sign me up again")

    assert captured.tool_called is True
    assert len(spy.calls) == 1
    assert follow_up.tool_called is False
    assert len(spy.calls) == 1
    assert "already captured" in follow_up.text.lower()


def test_lead_flow_continues_across_non_high_intent_turns(tmp_path: Path) -> None:
    agent, spy = _build_agent(tmp_path)
    session_id = "flow-continuation-session"

    kickoff = agent.handle_message(session_id, "Get started")
    plain_name = agent.handle_message(session_id, "Nina")
    plain_email = agent.handle_message(session_id, "nina@example.com")
    final = agent.handle_message(session_id, "TikTok")

    assert kickoff.tool_called is False
    assert plain_name.tool_called is False
    assert plain_email.tool_called is False
    assert final.tool_called is True
    assert len(spy.calls) == 1


def test_agent_asks_one_missing_field_at_a_time(tmp_path: Path) -> None:
    agent, _spy = _build_agent(tmp_path)
    session_id = "ask-one-field-session"

    first = agent.handle_message(session_id, "subscribe me")
    second = agent.handle_message(session_id, "My name is Aria")
    third = agent.handle_message(session_id, "aria@example.com")

    assert "name: missing" in first.text
    assert "What is your name?" in first.text

    assert "name: Aria" in second.text
    assert "email: missing" in second.text
    assert "What is the best email to reach you?" in second.text

    assert "email: aria@example.com" in third.text
    assert "platform: missing" in third.text
    assert "Which platform do you primarily create on?" in third.text


def test_mid_lead_pricing_question_uses_kb_without_losing_slots(tmp_path: Path) -> None:
    agent, spy = _build_agent(tmp_path)
    session_id = "mid-lead-kb-session"

    agent.handle_message(session_id, "Sign me up")
    agent.handle_message(session_id, "My name is Zoe")
    kb_turn = agent.handle_message(session_id, "What is your refund policy?")

    assert kb_turn.tool_called is False
    assert kb_turn.intent == "product_or_pricing_inquiry"
    assert "No refunds after 7 days" in kb_turn.text
    assert spy.calls == []

    state = agent.inspect_session_state(session_id)
    assert state["lead_name"] == "Zoe"
    assert state["lead_email"] is None
    assert state["lead_platform"] is None
    assert state["lead_status"] == "collecting"


def test_only_first_missing_slot_is_accepted_per_turn(tmp_path: Path) -> None:
    """While collecting email, a name-shaped message must not overwrite or fill other slots."""
    agent, spy = _build_agent(tmp_path)
    session_id = "one-slot-per-turn"

    agent.handle_message(session_id, "I want to try")
    agent.handle_message(session_id, "My name is Quinn")

    stray = agent.handle_message(session_id, "My name is Wrongname")

    assert stray.tool_called is False
    assert spy.calls == []
    assert "name: Quinn" in stray.text
    assert "email: missing" in stray.text
    state = agent.inspect_session_state(session_id)
    assert state["lead_name"] == "Quinn"
    assert state["lead_email"] is None


def test_when_collecting_name_email_only_message_does_not_advance_slots(tmp_path: Path) -> None:
    agent, spy = _build_agent(tmp_path)
    session_id = "email-only-while-name"

    agent.handle_message(session_id, "Subscribe")
    reply = agent.handle_message(session_id, "lead@example.com")

    assert reply.tool_called is False
    assert spy.calls == []
    assert "name: missing" in reply.text
    state = agent.inspect_session_state(session_id)
    assert state["lead_name"] is None
    assert state["lead_email"] is None


def test_mid_flow_faq_includes_resume_prompt(tmp_path: Path) -> None:
    agent, _spy = _build_agent(tmp_path)
    session_id = "faq-resume-session"

    agent.handle_message(session_id, "Sign me up")
    response = agent.handle_message(session_id, "What is your refund policy?")

    assert "No refunds after 7 days" in response.text
    assert "When you're ready, we can continue signup." in response.text


def test_clarification_pause_keeps_state_and_stops_collection(tmp_path: Path) -> None:
    agent, spy = _build_agent(tmp_path)
    session_id = "clarification-session"

    agent.handle_message(session_id, "Sign me up")
    pause_turn = agent.handle_message(session_id, "maybe later")

    assert pause_turn.tool_called is False
    assert "pause lead collection" in pause_turn.text.lower()
    assert spy.calls == []
    state = agent.inspect_session_state(session_id)
    assert state["lead_status"] == "new"
    assert state["memory_snapshot"]["lead_collection_paused"] is True


def test_slot_retry_fallback_for_email_then_skip(tmp_path: Path) -> None:
    agent, spy = _build_agent(tmp_path)
    session_id = "retry-fallback-session"

    agent.handle_message(session_id, "Sign me up")
    agent.handle_message(session_id, "My name is Ava")
    first_miss = agent.handle_message(session_id, "not-an-email")
    second_miss = agent.handle_message(session_id, "still-not-an-email")
    skip_reply = agent.handle_message(session_id, "skip lead")

    assert "What is the best email to reach you?" in first_miss.text
    assert "reply 'skip lead'" in second_miss.text.lower()
    assert "pause lead collection" in skip_reply.text.lower()
    assert spy.calls == []


def test_resume_signup_after_pause_continues_slot_collection(tmp_path: Path) -> None:
    agent, _spy = _build_agent(tmp_path)
    session_id = "resume-after-pause-session"

    agent.handle_message(session_id, "Sign me up")
    agent.handle_message(session_id, "maybe later")
    resumed = agent.handle_message(session_id, "resume signup")

    assert "What is your name?" in resumed.text
    state = agent.inspect_session_state(session_id)
    assert state["memory_snapshot"]["lead_collection_paused"] is False


def test_rate_limit_hook_blocks_processing(tmp_path: Path) -> None:
    spy = LeadCaptureSpy()
    memory = SimpleMemStore(storage_path=tmp_path / "simplemem_sessions.json")
    agent = AutoStreamAgent(
        memory_store=memory,
        lead_capture_tool=spy,
        rate_limit_hook=lambda _session_id: False,
    )

    response = agent.handle_message("limited-session", "Sign me up")

    assert response.intent == "rate_limited"
    assert "too quickly" in response.text
    assert response.tool_called is False


def test_idempotent_fingerprint_prevents_duplicate_capture_call(tmp_path: Path) -> None:
    spy = LeadCaptureSpy()
    memory = SimpleMemStore(storage_path=tmp_path / "simplemem_sessions.json")
    agent = AutoStreamAgent(memory_store=memory, lead_capture_tool=spy)
    session_id = "idempotent-session"

    memory.update_lead_fields(
        session_id,
        name="Ava",
        email="ava@example.com",
        platform="YouTube",
    )
    # Pre-store same payload fingerprint to simulate webhook retry.
    memory.set_lead_capture_fingerprint(
        session_id,
        hashlib.sha256("ava|ava@example.com|youtube".encode("utf-8")).hexdigest(),
    )

    response = agent.handle_message(session_id, "continue")

    assert response.tool_called is False
    assert "already captured for this exact payload" in response.text
    assert spy.calls == []


def test_tool_failure_opens_circuit_and_returns_fallback(tmp_path: Path) -> None:
    def failing_tool(_name: str, _email: str, _platform: str) -> str:
        raise RuntimeError("downstream unavailable")

    memory = SimpleMemStore(storage_path=tmp_path / "simplemem_sessions.json")
    agent = AutoStreamAgent(
        memory_store=memory,
        lead_capture_tool=failing_tool,
        circuit_fail_threshold=1,
        circuit_cooldown_seconds=60.0,
    )
    session_id = "circuit-session"

    memory.update_lead_fields(
        session_id,
        name="Maya",
        email="maya@example.com",
        platform="Instagram",
    )
    first = agent.handle_message(session_id, "continue")
    second = agent.handle_message(session_id, "continue")

    assert "Lead capture failed" in first.text
    assert "temporarily unavailable" in second.text
    assert first.tool_called is False
    assert second.tool_called is False


def test_tool_timeout_returns_safe_failure_message(tmp_path: Path) -> None:
    def slow_tool(name: str, email: str, platform: str) -> str:
        time.sleep(0.03)
        return f"Lead captured successfully: {name}, {email}, {platform}"

    memory = SimpleMemStore(storage_path=tmp_path / "simplemem_sessions.json")
    agent = AutoStreamAgent(
        memory_store=memory,
        lead_capture_tool=slow_tool,
        tool_timeout_seconds=0.001,
        circuit_fail_threshold=5,
    )
    session_id = "timeout-session"
    memory.update_lead_fields(session_id, name="Lina", email="lina@example.com", platform="TikTok")

    response = agent.handle_message(session_id, "continue")
    assert "Lead capture failed" in response.text
    assert response.tool_called is False
