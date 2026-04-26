"""Tests for LangGraph orchestration wrapper."""

from __future__ import annotations

from pathlib import Path

from src.graph_agent import build_langgraph_agent


def test_langgraph_wrapper_preserves_lead_tool_gating(tmp_path: Path) -> None:
    agent = build_langgraph_agent(
        storage_path=tmp_path / "simplemem_sessions.json",
        enable_llm_polish=False,
    )
    session_id = "graph-gating-session"

    first = agent.handle_message(session_id, "Sign me up")
    second = agent.handle_message(session_id, "My name is Ava")
    third = agent.handle_message(session_id, "ava@example.com")
    fourth = agent.handle_message(session_id, "YouTube")

    assert first.tool_called is False
    assert second.tool_called is False
    assert third.tool_called is False
    assert fourth.tool_called is True
    assert "Lead captured successfully: Ava, ava@example.com, YouTube" in fourth.text


def test_langgraph_wrapper_persists_multi_turn_state(tmp_path: Path) -> None:
    agent = build_langgraph_agent(
        storage_path=tmp_path / "simplemem_sessions.json",
        enable_llm_polish=False,
    )
    session_id = "graph-memory-session"

    turns = (
        "Hey",
        "What are your pricing plans?",
        "I want to try the Pro plan",
        "My name is Maya",
        "maya@example.com",
        "YouTube",
    )
    for message in turns:
        agent.handle_message(session_id, message)

    state = agent.inspect_session_state(session_id)
    assert state["turn_count"] >= 6
    assert state["lead_status"] == "captured"
