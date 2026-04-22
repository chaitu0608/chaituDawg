"""Phase 5 tests for lead qualification and strict tool gating."""

from __future__ import annotations

from pathlib import Path

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
