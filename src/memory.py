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
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional, Union

from src.input_limits import (
    MAX_LEAD_FIELD_CHARS,
    MAX_STORED_MESSAGES,
    clamp_persisted_text,
    validate_session_id,
)
from src.tools import mask_email_in_text


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
    messages: List[ConversationMessage] = field(default_factory=list)
    intent: Optional[str] = None
    lead_status: str = "new"
    lead_name: Optional[str] = None
    lead_email: Optional[str] = None
    lead_platform: Optional[str] = None
    email_retry_count: int = 0
    platform_retry_count: int = 0
    lead_collection_paused: bool = False
    lead_capture_fingerprint: Optional[str] = None
    kb_context: List[str] = field(default_factory=list)
    turn_count: int = 0
    memory_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_state_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe state payload including required workflow fields."""
        return {
            "messages": [asdict(message) for message in self.messages],
            "intent": self.intent,
            "lead_status": self.lead_status,
            "lead_name": self.lead_name,
            "lead_email": self.lead_email,
            "lead_platform": self.lead_platform,
            "email_retry_count": self.email_retry_count,
            "platform_retry_count": self.platform_retry_count,
            "lead_collection_paused": self.lead_collection_paused,
            "lead_capture_fingerprint": self.lead_capture_fingerprint,
            "kb_context": list(self.kb_context),
            "turn_count": self.turn_count,
            "memory_snapshot": dict(self.memory_snapshot),
        }


class SimpleMemStore:
    """SimpleMem-style durable session store with deterministic JSON persistence."""

    def __init__(
        self,
        storage_path: Union[str, Path] = DEFAULT_SIMPLEMEM_PATH,
        *,
        enable_checksum: bool = False,
    ) -> None:
        self._storage_path = Path(storage_path)
        self._enable_checksum = enable_checksum
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)

    def load_state(self, session_id: str) -> SessionState:
        """Load a session state, creating an empty one when missing."""
        validate_session_id(session_id)
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
        intent: Optional[str] = None,
        kb_context_refs: tuple[str, ...] = (),
    ) -> SessionState:
        """Persist one user+assistant turn and update optional intent/context refs."""
        validate_session_id(session_id)
        state = self.load_state(session_id)

        state.messages.append(
            ConversationMessage(
                role="user",
                content=clamp_persisted_text(user_message),
            )
        )
        state.messages.append(
            ConversationMessage(
                role="assistant",
                content=clamp_persisted_text(assistant_message),
            )
        )
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
        validate_session_id(session_id)
        state = self.load_state(session_id)
        state.intent = intent
        state.memory_snapshot = _build_memory_snapshot(state)
        self._save_state(state)
        return state

    def begin_lead_collection(self, session_id: str) -> SessionState:
        """Set lead_status to collecting once qualification starts."""
        validate_session_id(session_id)
        state = self.load_state(session_id)
        if state.lead_status == "new":
            state.lead_status = "collecting"
        state.lead_collection_paused = False
        state.memory_snapshot = _build_memory_snapshot(state)
        self._save_state(state)
        return state

    def update_lead_fields(
        self,
        session_id: str,
        *,
        name: Optional[str] = None,
        email: Optional[str] = None,
        platform: Optional[str] = None,
    ) -> SessionState:
        """Persist lead slot values and refresh lead_status deterministically."""
        validate_session_id(session_id)
        state = self.load_state(session_id)

        if name is not None:
            state.lead_name = clamp_persisted_text(name, max_chars=MAX_LEAD_FIELD_CHARS)
        if email is not None:
            state.lead_email = clamp_persisted_text(email, max_chars=MAX_LEAD_FIELD_CHARS)
            state.email_retry_count = 0
        if platform is not None:
            state.lead_platform = clamp_persisted_text(platform, max_chars=MAX_LEAD_FIELD_CHARS)
            state.platform_retry_count = 0

        state.lead_collection_paused = False
        state.lead_status = _derive_lead_status(state)
        state.memory_snapshot = _build_memory_snapshot(state)
        self._save_state(state)
        return state

    def increment_slot_retry(self, session_id: str, slot_name: str) -> SessionState:
        """Increment retry counter for the specified required lead slot."""
        validate_session_id(session_id)
        state = self.load_state(session_id)
        if slot_name == "email":
            state.email_retry_count += 1
        elif slot_name == "platform":
            state.platform_retry_count += 1
        state.memory_snapshot = _build_memory_snapshot(state)
        self._save_state(state)
        return state

    def pause_lead_collection(self, session_id: str) -> SessionState:
        """Pause lead collection when user opts out or asks to continue later."""
        validate_session_id(session_id)
        state = self.load_state(session_id)
        state.lead_collection_paused = True
        if state.lead_status in {"collecting", "ready"}:
            state.lead_status = "new"
        state.memory_snapshot = _build_memory_snapshot(state)
        self._save_state(state)
        return state

    def mark_lead_captured(self, session_id: str) -> SessionState:
        """Mark the lead as captured after successful gated tool execution."""
        validate_session_id(session_id)
        state = self.load_state(session_id)
        state.lead_status = "captured"
        state.memory_snapshot = _build_memory_snapshot(state)
        self._save_state(state)
        return state

    def set_lead_capture_fingerprint(self, session_id: str, fingerprint: str) -> SessionState:
        """Persist idempotency fingerprint for the successful lead capture payload."""
        validate_session_id(session_id)
        state = self.load_state(session_id)
        state.lead_capture_fingerprint = fingerprint
        state.memory_snapshot = _build_memory_snapshot(state)
        self._save_state(state)
        return state

    def lead_fields_complete(self, session_id: str) -> bool:
        """Check whether all required lead fields are currently present."""
        validate_session_id(session_id)
        state = self.load_state(session_id)
        return _lead_fields_complete(state)

    def restore_recent_context(self, session_id: str, max_turns: int = 6) -> List[Dict[str, str]]:
        """Return the most recent chat context as role/content dictionaries."""
        validate_session_id(session_id)
        state = self.load_state(session_id)
        max_messages = max(1, max_turns) * 2
        recent_messages = state.messages[-max_messages:]
        return [asdict(message) for message in recent_messages]

    def inspect_state(self, session_id: str, *, include_pii: bool = False) -> Dict[str, Any]:
        """Return full state payload for logging/debug inspection."""
        validate_session_id(session_id)
        state = self.load_state(session_id)
        state.memory_snapshot = _build_memory_snapshot(state)
        snapshot = state.to_state_dict()
        if include_pii:
            return snapshot
        return _mask_snapshot(snapshot)

    def _save_state(self, state: SessionState, all_sessions: Optional[Dict[str, Any]] = None) -> None:
        sessions = all_sessions if all_sessions is not None else self._load_all_sessions()
        state.memory_snapshot = _build_memory_snapshot(state)
        sessions[state.session_id] = state.to_state_dict()
        self._write_all_sessions(sessions)

    def _load_all_sessions(self) -> Dict[str, Any]:
        if not self._storage_path.exists():
            return {}

        with self._storage_path.open("r", encoding="utf-8") as file_handle:
            raw = json.load(file_handle)

        if not isinstance(raw, dict):
            raise ValueError("SimpleMem storage format must be a JSON object")

        if "sessions" in raw:
            sessions = raw.get("sessions")
            if not isinstance(sessions, dict):
                raise ValueError("SimpleMem sessions payload must be a JSON object")
            if self._enable_checksum:
                self._verify_checksum(raw)
            return sessions

        return raw

    def _write_all_sessions(self, payload: Dict[str, Any]) -> None:
        # Write atomically to avoid partial writes when persisting turn-by-turn state.
        to_write: Dict[str, Any] = payload
        if self._enable_checksum:
            serialized = _stable_json(payload)
            digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
            to_write = {
                "_meta": {"payload_sha256": digest},
                "sessions": payload,
            }
        with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=self._storage_path.parent) as temp_file:
            json.dump(to_write, temp_file, indent=2)
            temp_file.write("\n")
            temp_path = Path(temp_file.name)

        temp_path.replace(self._storage_path)

    def _verify_checksum(self, wrapped_payload: Dict[str, Any]) -> None:
        meta = wrapped_payload.get("_meta")
        if not isinstance(meta, dict):
            raise ValueError("Missing _meta for checksummed SimpleMem payload")
        expected = meta.get("payload_sha256")
        sessions = wrapped_payload.get("sessions")
        if not isinstance(expected, str) or not isinstance(sessions, dict):
            raise ValueError("Invalid checksum metadata in SimpleMem payload")
        actual = hashlib.sha256(_stable_json(sessions).encode("utf-8")).hexdigest()
        if actual != expected:
            raise ValueError("SimpleMem checksum mismatch: possible session tampering detected")


def _new_session_state(session_id: str) -> SessionState:
    state = SessionState(session_id=session_id)
    state.memory_snapshot = _build_memory_snapshot(state)
    return state


def _deserialize_state(session_id: str, raw_state: Dict[str, Any]) -> SessionState:
    raw_messages = raw_state.get("messages", [])
    if not isinstance(raw_messages, list):
        raw_messages = []
    raw_messages = raw_messages[-MAX_STORED_MESSAGES:]

    messages: List[ConversationMessage] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "assistant"))[:64]
        content = clamp_persisted_text(str(item.get("content", "")))
        messages.append(ConversationMessage(role=role, content=content))

    def _clamp_optional_str(value: object) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str):
            return None
        return clamp_persisted_text(value, max_chars=MAX_LEAD_FIELD_CHARS)

    turn_raw = raw_state.get("turn_count", 0)
    try:
        turn_count = int(turn_raw)
    except (TypeError, ValueError):
        turn_count = 0
    turn_count = max(0, min(turn_count, 1_000_000))

    kb_raw = raw_state.get("kb_context", [])
    kb_context: List[str] = []
    if isinstance(kb_raw, list):
        for ref in kb_raw[:256]:
            if isinstance(ref, str):
                kb_context.append(clamp_persisted_text(ref, max_chars=512))

    state = SessionState(
        session_id=session_id,
        messages=messages,
        intent=raw_state.get("intent") if isinstance(raw_state.get("intent"), str) else None,
        lead_status=str(raw_state.get("lead_status", "new"))[:32],
        lead_name=_clamp_optional_str(raw_state.get("lead_name")),
        lead_email=_clamp_optional_str(raw_state.get("lead_email")),
        lead_platform=_clamp_optional_str(raw_state.get("lead_platform")),
        email_retry_count=int(raw_state.get("email_retry_count", 0)) if isinstance(raw_state.get("email_retry_count", 0), int) else 0,
        platform_retry_count=int(raw_state.get("platform_retry_count", 0)) if isinstance(raw_state.get("platform_retry_count", 0), int) else 0,
        lead_collection_paused=bool(raw_state.get("lead_collection_paused", False)),
        lead_capture_fingerprint=raw_state.get("lead_capture_fingerprint")
        if isinstance(raw_state.get("lead_capture_fingerprint"), str)
        else None,
        kb_context=kb_context,
        turn_count=turn_count,
        memory_snapshot=dict(raw_state.get("memory_snapshot", {}))
        if isinstance(raw_state.get("memory_snapshot"), dict)
        else {},
    )
    state.memory_snapshot = _build_memory_snapshot(state)
    return state


def _build_memory_snapshot(state: SessionState) -> Dict[str, Any]:
    return {
        "session_id": state.session_id,
        "messages_stored": len(state.messages),
        "turn_count": state.turn_count,
        "intent": state.intent,
        "lead_status": state.lead_status,
        "lead_fields_complete": _lead_fields_complete(state),
        "email_retry_count": state.email_retry_count,
        "platform_retry_count": state.platform_retry_count,
        "lead_collection_paused": state.lead_collection_paused,
        "lead_capture_fingerprint": state.lead_capture_fingerprint,
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


def _dedupe_preserve_order(values: tuple[str, ...]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def _stable_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _mask_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    masked = dict(snapshot)
    if isinstance(masked.get("lead_email"), str):
        masked["lead_email"] = mask_email_in_text(masked["lead_email"])
    messages = masked.get("messages")
    if isinstance(messages, list):
        masked_messages: List[Dict[str, Any]] = []
        for message in messages:
            if isinstance(message, dict):
                message_copy = dict(message)
                content = message_copy.get("content")
                if isinstance(content, str):
                    message_copy["content"] = mask_email_in_text(content)
                masked_messages.append(message_copy)
        masked["messages"] = masked_messages
    memory_snapshot = masked.get("memory_snapshot")
    if isinstance(memory_snapshot, dict):
        memory_snapshot_copy = dict(memory_snapshot)
        preview = memory_snapshot_copy.get("recent_context_preview")
        if isinstance(preview, list):
            memory_snapshot_copy["recent_context_preview"] = [
                mask_email_in_text(item) if isinstance(item, str) else item
                for item in preview
            ]
        masked["memory_snapshot"] = memory_snapshot_copy
    return masked
