"""Tests for inbound validation and persistence bounds."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agent import AutoStreamAgent
from src.input_limits import MAX_USER_MESSAGE_CHARS, clamp_user_message, validate_session_id
from src.memory import SimpleMemStore


def test_validate_session_id_rejects_newline() -> None:
    with pytest.raises(ValueError):
        validate_session_id("bad\nid")


def test_validate_session_id_accepts_whatsapp_like_ids() -> None:
    assert validate_session_id("+15551234567") == "+15551234567"
    assert validate_session_id("demo-session-1") == "demo-session-1"


def test_clamp_user_message_truncates() -> None:
    long = "x" * (MAX_USER_MESSAGE_CHARS + 50)
    out = clamp_user_message(long)
    assert len(out) == MAX_USER_MESSAGE_CHARS


def test_agent_rejects_invalid_session_id(tmp_path: Path) -> None:
    agent = AutoStreamAgent(memory_store=SimpleMemStore(storage_path=tmp_path / "m.json"))
    with pytest.raises(ValueError):
        agent.handle_message("\t", "hello")


def test_memory_load_rejects_invalid_session_id(tmp_path: Path) -> None:
    store = SimpleMemStore(storage_path=tmp_path / "m.json")
    with pytest.raises(ValueError):
        store.load_state("")
