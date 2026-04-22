"""Phase 4 tests for durable conversation memory behavior."""

from __future__ import annotations

from pathlib import Path

from src.memory import SimpleMemStore


def test_state_persists_across_six_turns(tmp_path: Path) -> None:
    storage_path = tmp_path / "simplemem_sessions.json"
    session_id = "session-six-turn"

    writer = SimpleMemStore(storage_path=storage_path)
    for index in range(6):
        writer.record_turn(
            session_id,
            user_message=f"user-message-{index}",
            assistant_message=f"assistant-message-{index}",
            intent="casual_greeting",
            kb_context_refs=("plans:basic",),
        )

    reader = SimpleMemStore(storage_path=storage_path)
    restored = reader.load_state(session_id)

    assert restored.turn_count == 6
    assert restored.intent == "casual_greeting"
    assert len(restored.messages) == 12
    assert restored.kb_context == ["plans:basic"]

    recent = reader.restore_recent_context(session_id, max_turns=6)
    assert len(recent) == 12
    assert recent[0]["content"] == "user-message-0"
    assert recent[-1]["content"] == "assistant-message-5"


def test_lead_fields_persist_and_become_ready(tmp_path: Path) -> None:
    storage_path = tmp_path / "simplemem_sessions.json"
    session_id = "lead-session"

    writer = SimpleMemStore(storage_path=storage_path)
    writer.update_lead_fields(session_id, name="Ava")
    writer.update_lead_fields(session_id, email="ava@example.com")

    mid_state = writer.load_state(session_id)
    assert mid_state.lead_status == "collecting"
    assert mid_state.lead_name == "Ava"
    assert mid_state.lead_email == "ava@example.com"
    assert mid_state.lead_platform is None

    writer.update_lead_fields(session_id, platform="YouTube")

    reader = SimpleMemStore(storage_path=storage_path)
    final_state = reader.load_state(session_id)

    assert final_state.lead_status == "ready"
    assert final_state.lead_name == "Ava"
    assert final_state.lead_email == "ava@example.com"
    assert final_state.lead_platform == "YouTube"
    assert reader.lead_fields_complete(session_id) is True


def test_memory_state_is_inspectable_for_logging(tmp_path: Path) -> None:
    storage_path = tmp_path / "simplemem_sessions.json"
    session_id = "inspect-session"

    store = SimpleMemStore(storage_path=storage_path)
    store.record_turn(
        session_id,
        user_message="Can you remind me of my progress?",
        assistant_message="You have shared your name so far.",
        intent="high_intent_lead",
        kb_context_refs=("policies:refund_policy", "policies:refund_policy"),
    )

    snapshot = store.inspect_state(session_id)

    assert snapshot["intent"] == "high_intent_lead"
    assert snapshot["lead_status"] == "new"
    assert snapshot["turn_count"] == 1
    assert snapshot["kb_context"] == ["policies:refund_policy"]

    memory_snapshot = snapshot["memory_snapshot"]
    assert memory_snapshot["session_id"] == session_id
    assert memory_snapshot["messages_stored"] == 2
    assert memory_snapshot["turn_count"] == 1
    assert memory_snapshot["lead_fields_complete"] is False
    assert len(memory_snapshot["recent_context_preview"]) == 2


def test_mark_lead_captured_is_sticky(tmp_path: Path) -> None:
    storage_path = tmp_path / "simplemem_sessions.json"
    session_id = "captured-session"

    store = SimpleMemStore(storage_path=storage_path)
    store.update_lead_fields(
        session_id,
        name="Maya",
        email="maya@example.com",
        platform="Instagram",
    )

    captured = store.mark_lead_captured(session_id)
    assert captured.lead_status == "captured"

    # Even if fields are updated again, captured status remains authoritative.
    store.update_lead_fields(session_id, platform="YouTube")
    reloaded = store.load_state(session_id)

    assert reloaded.lead_status == "captured"
    assert reloaded.lead_platform == "YouTube"
