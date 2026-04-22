"""AutoStream conversational agent orchestration.

Phase 5 scope:
- Intent routing.
- Lead qualification slot filling.
- Strict tool execution gating.
- Session memory checkpointing per turn.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.intents import IntentClassifier, IntentLabel
from src.memory import SessionState, SimpleMemStore
from src.rag import RAGResult, answer_from_kb
from src.tools import (
    LeadCapturePayload,
    extract_email,
    extract_name,
    extract_platform,
    mock_lead_capture,
    normalize_platform,
    supported_platforms,
)


LeadCaptureTool = Callable[[str, str, str], str]


@dataclass(frozen=True)
class AgentResponse:
    """Structured response payload from one user turn."""

    text: str
    intent: str
    lead_status: str
    lead_fields: dict[str, str | None]
    tool_called: bool


class AutoStreamAgent:
    """Deterministic production-oriented conversational controller."""

    def __init__(
        self,
        *,
        memory_store: SimpleMemStore | None = None,
        intent_classifier: IntentClassifier | None = None,
        lead_capture_tool: LeadCaptureTool = mock_lead_capture,
    ) -> None:
        self._memory = memory_store or SimpleMemStore()
        self._intent_classifier = intent_classifier or IntentClassifier()
        self._lead_capture_tool = lead_capture_tool

    def handle_message(self, session_id: str, user_message: str) -> AgentResponse:
        """Process one inbound message and return the agent response."""
        prior_state = self._memory.load_state(session_id)
        intent_result = self._intent_classifier.classify(user_message)
        self._memory.update_intent(session_id, intent_result.label.value)

        tool_called = False
        kb_context_refs: tuple[str, ...] = ()

        if self._should_use_lead_flow(prior_state=prior_state, intent=intent_result.label):
            response_text, tool_called = self._handle_lead_flow(
                session_id=session_id,
                user_message=user_message,
                intent=intent_result.label,
            )
        elif intent_result.label == IntentLabel.PRODUCT_OR_PRICING_INQUIRY:
            rag_result = answer_from_kb(user_message)
            response_text = rag_result.answer
            kb_context_refs = rag_result.kb_context
        else:
            response_text = _greeting_response()

        updated_state = self._memory.record_turn(
            session_id=session_id,
            user_message=user_message,
            assistant_message=response_text,
            intent=intent_result.label.value,
            kb_context_refs=kb_context_refs,
        )

        return AgentResponse(
            text=response_text,
            intent=intent_result.label.value,
            lead_status=updated_state.lead_status,
            lead_fields={
                "name": updated_state.lead_name,
                "email": updated_state.lead_email,
                "platform": updated_state.lead_platform,
            },
            tool_called=tool_called,
        )

    def inspect_session_state(self, session_id: str) -> dict[str, object]:
        """Expose internal persisted state for diagnostics and testing."""
        return self._memory.inspect_state(session_id)

    def _should_use_lead_flow(self, prior_state: SessionState, intent: IntentLabel) -> bool:
        if prior_state.lead_status in {"collecting", "ready"}:
            return True
        if intent == IntentLabel.HIGH_INTENT_LEAD:
            return True
        return False

    def _handle_lead_flow(self, session_id: str, user_message: str, intent: IntentLabel) -> tuple[str, bool]:
        state = self._memory.load_state(session_id)

        if intent == IntentLabel.HIGH_INTENT_LEAD and state.lead_status == "new":
            self._memory.begin_lead_collection(session_id)
            state = self._memory.load_state(session_id)

        if state.lead_status == "captured":
            return (
                "Your lead details are already captured for this session. "
                "If you want to submit another lead, please start a new session.",
                False,
            )

        parsed_name = extract_name(user_message)
        parsed_email = extract_email(user_message)
        parsed_platform = extract_platform(user_message)

        if parsed_name or parsed_email or parsed_platform:
            self._memory.update_lead_fields(
                session_id=session_id,
                name=parsed_name,
                email=parsed_email,
                platform=parsed_platform,
            )
            state = self._memory.load_state(session_id)

        if self._memory.lead_fields_complete(session_id):
            payload = _payload_from_state(state)
            capture_message = self._lead_capture_tool(payload.name, payload.email, payload.platform)
            self._memory.mark_lead_captured(session_id)
            return (
                f"Thanks, all required details are collected. {capture_message}",
                True,
            )

        next_field = _next_missing_field(state)
        summary = _slot_summary(state)

        if next_field == "name":
            prompt = "What is your name?"
        elif next_field == "email":
            if "@" in user_message and parsed_email is None:
                prompt = "That email looks invalid. Please share a valid email address."
            else:
                prompt = "What is the best email to reach you?"
        else:
            requested_platforms = ", ".join(supported_platforms())
            prompt = f"Which platform do you primarily create on? ({requested_platforms})"

        return f"{summary} {prompt}", False


def build_default_agent(storage_path: str | Path | None = None) -> AutoStreamAgent:
    """Factory helper for standard local execution."""
    memory = SimpleMemStore(storage_path=storage_path) if storage_path else SimpleMemStore()
    return AutoStreamAgent(memory_store=memory)


def _greeting_response() -> str:
    return "Hi! I can help with AutoStream plans, features, and policies. What would you like to know?"


def _next_missing_field(state: SessionState) -> str:
    if not state.lead_name:
        return "name"
    if not state.lead_email:
        return "email"
    if not state.lead_platform:
        return "platform"
    return ""


def _slot_summary(state: SessionState) -> str:
    name = state.lead_name or "missing"
    email = state.lead_email or "missing"
    platform = state.lead_platform or "missing"
    return f"Collected so far - name: {name}, email: {email}, platform: {platform}."


def _payload_from_state(state: SessionState) -> LeadCapturePayload:
    platform = normalize_platform(state.lead_platform or "")
    if not state.lead_name or not state.lead_email or not platform:
        raise ValueError("Lead payload requested before all fields were complete")

    return LeadCapturePayload(
        name=state.lead_name,
        email=state.lead_email,
        platform=platform,
    )
