"""Durable session memory wrapper for AutoStream.

Phase 4 scope:
- Persist conversation history and session state.
- Restore context across multiple turns.
- Keep lead fields durable until tool execution in later phases.

This module intentionally stores conversation state and retrieval references only.
Product facts remain in the local knowledge base and are never duplicated here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


DEFAULT_SIMPLEMEM_PATH = Path(__file__).resolve().parent.parent / "data" / "simplemem_sessions.json"


@dataclass(frozen=True)
class ConversationMessage:
    """A single persisted message in the conversation timeline."""

    role: str
    content: str


@dataclass
class SessionState:
    """Durable state fields used by the conversational workflow."""

    session_id: str
    messages: list[ConversationMessage] = field(default_factory=list)
    intent: str | None = None
    lead_status: str = "new"
    lead_name: str | None = None
    lead_email: str | None = None
    lead_platform: str | None = None
    kb_context: list[str] = field(default_factory=list)
    turn_count: int = 0
    memory_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_state_dict(self) -> dict[str, Any]:
        """Return a JSON-safe state payload including required workflow fields."""
        return {
            "messages": [asdict(message) for message in self.messages],
            "intent": self.intent,
            "lead_status": self.lead_status,
            "lead_name": self.lead_name,
            "lead_email": self.lead_email,
            "lead_platform": self.lead_platform,
            "kb_context": list(self.kb_context),
            "turn_count": self.turn_count,
            "memory_snapshot": dict(self.memory_snapshot),
        }


class SimpleMemStore:
    """SimpleMem-style durable session store with deterministic JSON persistence."""

    def __init__(self, storage_path: str | Path = DEFAULT_SIMPLEMEM_PATH) -> None:
        self._storage_path = Path(storage_path)
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)

    def load_state(self, session_id: str) -> SessionState:
        """Load a session state, creating an empty one when missing."""
        all_sessions = self._load_all_sessions()
        raw_state = all_sessions.get(session_id)
        if raw_state is None:
            state = _new_session_state(session_id)
            self._save_state(state, all_sessions=all_sessions)
            return state

        state = _deserialize_state(session_id=session_id, raw_state=raw_state)
        state.memory_snapshot = _build_memory_snapshot(state)
        return state

    def record_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        *,
        intent: str | None = None,
        kb_context_refs: tuple[str, ...] = (),
    ) -> SessionState:
        """Persist one user+assistant turn and update optional intent/context refs."""
        state = self.load_state(session_id)

        state.messages.append(ConversationMessage(role="user", content=user_message))
        state.messages.append(ConversationMessage(role="assistant", content=assistant_message))
        state.turn_count += 1

        if intent is not None:
            state.intent = intent

        if kb_context_refs:
            state.kb_context = _dedupe_preserve_order(kb_context_refs)

        state.memory_snapshot = _build_memory_snapshot(state)
        self._save_state(state)
        return state

    def update_intent(self, session_id: str, intent: str) -> SessionState:
        """Persist latest detected intent."""
        state = self.load_state(session_id)
        state.intent = intent
        state.memory_snapshot = _build_memory_snapshot(state)
        self._save_state(state)
        return state

    def begin_lead_collection(self, session_id: str) -> SessionState:
        """Set lead_status to collecting once qualification starts."""
        state = self.load_state(session_id)
        if state.lead_status == "new":
            state.lead_status = "collecting"
        state.memory_snapshot = _build_memory_snapshot(state)
        self._save_state(state)
        return state

    def update_lead_fields(
        self,
        session_id: str,
        *,
        name: str | None = None,
        email: str | None = None,
        platform: str | None = None,
    ) -> SessionState:
        """Persist lead slot values and refresh lead_status deterministically."""
        state = self.load_state(session_id)

        if name is not None:
            state.lead_name = name
        if email is not None:
            state.lead_email = email
        if platform is not None:
            state.lead_platform = platform

        state.lead_status = _derive_lead_status(state)
        state.memory_snapshot = _build_memory_snapshot(state)
        self._save_state(state)
        return state

    def mark_lead_captured(self, session_id: str) -> SessionState:
        """Mark the lead as captured after successful gated tool execution."""
        state = self.load_state(session_id)
        state.lead_status = "captured"
        state.memory_snapshot = _build_memory_snapshot(state)
        self._save_state(state)
        return state

    def lead_fields_complete(self, session_id: str) -> bool:
        """Check whether all required lead fields are currently present."""
        state = self.load_state(session_id)
        return _lead_fields_complete(state)

    def restore_recent_context(self, session_id: str, max_turns: int = 6) -> list[dict[str, str]]:
        """Return the most recent chat context as role/content dictionaries."""
        state = self.load_state(session_id)
        max_messages = max(1, max_turns) * 2
        recent_messages = state.messages[-max_messages:]
        return [asdict(message) for message in recent_messages]

    def inspect_state(self, session_id: str) -> dict[str, Any]:
        """Return full state payload for logging/debug inspection."""
        state = self.load_state(session_id)
        state.memory_snapshot = _build_memory_snapshot(state)
        return state.to_state_dict()

    def _save_state(self, state: SessionState, all_sessions: dict[str, Any] | None = None) -> None:
        sessions = all_sessions if all_sessions is not None else self._load_all_sessions()
        state.memory_snapshot = _build_memory_snapshot(state)
        sessions[state.session_id] = state.to_state_dict()
        self._write_all_sessions(sessions)

    def _load_all_sessions(self) -> dict[str, Any]:
        if not self._storage_path.exists():
            return {}

        with self._storage_path.open("r", encoding="utf-8") as file_handle:
            raw = json.load(file_handle)

        if not isinstance(raw, dict):
            raise ValueError("SimpleMem storage format must be a JSON object")

        return raw

    def _write_all_sessions(self, payload: dict[str, Any]) -> None:
        # Write atomically to avoid partial writes when persisting turn-by-turn state.
        with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=self._storage_path.parent) as temp_file:
            json.dump(payload, temp_file, indent=2)
            temp_file.write("\n")
            temp_path = Path(temp_file.name)

        temp_path.replace(self._storage_path)


def _new_session_state(session_id: str) -> SessionState:
    state = SessionState(session_id=session_id)
    state.memory_snapshot = _build_memory_snapshot(state)
    return state


def _deserialize_state(session_id: str, raw_state: dict[str, Any]) -> SessionState:
    messages = [
        ConversationMessage(
            role=item.get("role", "assistant"),
            content=item.get("content", ""),
        )
        for item in raw_state.get("messages", [])
    ]

    state = SessionState(
        session_id=session_id,
        messages=messages,
        intent=raw_state.get("intent"),
        lead_status=raw_state.get("lead_status", "new"),
        lead_name=raw_state.get("lead_name"),
        lead_email=raw_state.get("lead_email"),
        lead_platform=raw_state.get("lead_platform"),
        kb_context=list(raw_state.get("kb_context", [])),
        turn_count=int(raw_state.get("turn_count", 0)),
        memory_snapshot=dict(raw_state.get("memory_snapshot", {})),
    )
    state.memory_snapshot = _build_memory_snapshot(state)
    return state


def _build_memory_snapshot(state: SessionState) -> dict[str, Any]:
    return {
        "session_id": state.session_id,
        "messages_stored": len(state.messages),
        "turn_count": state.turn_count,
        "intent": state.intent,
        "lead_status": state.lead_status,
        "lead_fields_complete": _lead_fields_complete(state),
        "recent_context_preview": [message.content for message in state.messages[-4:]],
    }


def _derive_lead_status(state: SessionState) -> str:
    if state.lead_status == "captured":
        return "captured"

    if _lead_fields_complete(state):
        return "ready"

    if state.lead_name or state.lead_email or state.lead_platform:
        return "collecting"

    return "new"


def _lead_fields_complete(state: SessionState) -> bool:
    return all(
        value is not None and value.strip() != ""
        for value in (state.lead_name, state.lead_email, state.lead_platform)
    )


def _dedupe_preserve_order(values: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered
