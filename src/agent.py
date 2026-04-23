"""AutoStream conversational agent orchestration.

Phase 5 scope:
- Intent routing.
- Lead qualification slot filling.
- Strict tool execution gating.
- Session memory checkpointing per turn.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import time
from typing import Callable

from src.input_limits import clamp_user_message, validate_session_id
from src.intents import IntentClassifier, IntentLabel
from src.lead_flow import (
    is_pause_or_uncertain_message,
    is_resume_request,
    lead_capture_fingerprint,
    next_missing_field,
    payload_from_state,
    slot_resume_prompt,
    slot_retry_count,
    slot_summary,
)
from src.memory import SessionState, SimpleMemStore
from src.rag import RAGResult, answer_from_kb
from src.tools import (
    LeadCapturePayload,
    extract_email,
    extract_name,
    extract_platform,
    mock_lead_capture,
    supported_platforms,
    validate_lead_payload,
)


LeadCaptureTool = Callable[[str, str, str], str]
RateLimitHook = Callable[[str], bool]
MAX_SLOT_RETRIES = 2


class RouteState(str, Enum):
    GREETING = "greeting"
    INQUIRY = "inquiry"
    LEAD_FLOW = "lead_flow"


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
        rate_limit_hook: RateLimitHook | None = None,
        tool_timeout_seconds: float = 2.0,
        circuit_fail_threshold: int = 3,
        circuit_cooldown_seconds: float = 30.0,
    ) -> None:
        self._memory = memory_store or SimpleMemStore()
        self._intent_classifier = intent_classifier or IntentClassifier()
        self._lead_capture_tool = lead_capture_tool
        self._rate_limit_hook = rate_limit_hook
        self._tool_timeout_seconds = tool_timeout_seconds
        self._circuit_fail_threshold = circuit_fail_threshold
        self._circuit_cooldown_seconds = circuit_cooldown_seconds
        self._tool_failure_count = 0
        self._circuit_open_until = 0.0

    def handle_message(self, session_id: str, user_message: str) -> AgentResponse:
        """Process one inbound message and return the agent response."""
        validate_session_id(session_id)
        user_message = clamp_user_message(user_message)
        if self._rate_limit_hook is not None and not self._rate_limit_hook(session_id):
            state = self._memory.load_state(session_id)
            return AgentResponse(
                text=(
                    "You are sending messages too quickly. Please wait a moment and try again."
                ),
                intent="rate_limited",
                lead_status=state.lead_status,
                lead_fields={
                    "name": state.lead_name,
                    "email": state.lead_email,
                    "platform": state.lead_platform,
                },
                tool_called=False,
            )
        prior_state = self._memory.load_state(session_id)
        intent_result = self._intent_classifier.classify(user_message)
        self._memory.update_intent(session_id, intent_result.label.value)

        tool_called = False
        kb_context_refs: tuple[str, ...] = ()

        route_state = self._resolve_route_state(prior_state, intent_result.label, user_message)
        if route_state == RouteState.LEAD_FLOW:
            effective_intent = intent_result.label
            if prior_state.lead_collection_paused and is_resume_request(user_message):
                effective_intent = IntentLabel.HIGH_INTENT_LEAD
            response_text, tool_called = self._handle_lead_flow(
                session_id=session_id,
                user_message=user_message,
                intent=effective_intent,
            )
        elif route_state == RouteState.INQUIRY:
            rag_result = answer_from_kb(user_message)
            response_text = rag_result.answer
            if prior_state.lead_status in {"collecting", "ready"}:
                lead_state = self._memory.load_state(session_id)
                next_field = next_missing_field(lead_state)
                if next_field:
                    response_text = f"{response_text} {slot_resume_prompt(next_field)}"
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

    def _resolve_route_state(
        self,
        prior_state: SessionState,
        intent: IntentLabel,
        user_message: str,
    ) -> RouteState:
        if self._should_use_lead_flow(prior_state=prior_state, intent=intent):
            return RouteState.LEAD_FLOW
        if prior_state.lead_collection_paused and is_resume_request(user_message):
            return RouteState.LEAD_FLOW
        if intent == IntentLabel.PRODUCT_OR_PRICING_INQUIRY:
            return RouteState.INQUIRY
        return RouteState.GREETING

    def inspect_session_state(self, session_id: str, *, include_pii: bool = False) -> dict[str, object]:
        """Expose internal persisted state for diagnostics and testing."""
        return self._memory.inspect_state(session_id, include_pii=include_pii)

    def _should_use_lead_flow(self, prior_state: SessionState, intent: IntentLabel) -> bool:
        if prior_state.lead_status in {"collecting", "ready"}:
            # Product questions are answered from the KB without abandoning slot collection.
            if intent == IntentLabel.PRODUCT_OR_PRICING_INQUIRY:
                return False
            return True
        if prior_state.lead_collection_paused and intent == IntentLabel.HIGH_INTENT_LEAD:
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

        if is_pause_or_uncertain_message(user_message):
            self._memory.pause_lead_collection(session_id)
            return (
                "No problem, we can continue later. I will pause lead collection for now. "
                "Say 'resume signup' or 'sign me up' whenever you want to continue.",
                False,
            )

        next_field = next_missing_field(state)
        # Accept at most one new slot per user turn (first missing field in order).
        parsed_name = extract_name(user_message) if next_field == "name" else None
        parsed_email = extract_email(user_message) if next_field == "email" else None
        parsed_platform = extract_platform(user_message) if next_field == "platform" else None

        if parsed_name or parsed_email or parsed_platform:
            self._memory.update_lead_fields(
                session_id=session_id,
                name=parsed_name,
                email=parsed_email,
                platform=parsed_platform,
            )
            state = self._memory.load_state(session_id)
        elif next_field in {"email", "platform"}:
            self._memory.increment_slot_retry(session_id, next_field)
            state = self._memory.load_state(session_id)
            if slot_retry_count(state, next_field) >= MAX_SLOT_RETRIES:
                return (
                    f"{slot_summary(state)} I can keep helping with product questions. "
                    "Reply 'skip lead' to continue without lead capture for now, "
                    "or share the missing detail to continue signup.",
                    False,
                )

        if self._memory.lead_fields_complete(session_id):
            payload = payload_from_state(state)
            fingerprint = lead_capture_fingerprint(payload)
            if state.lead_capture_fingerprint == fingerprint:
                self._memory.mark_lead_captured(session_id)
                return (
                    "Lead details were already captured for this exact payload in this session.",
                    False,
                )
            valid, failed_field = validate_lead_payload(
                payload.name,
                payload.email,
                payload.platform,
            )
            if not valid:
                summary = slot_summary(state)
                return (
                    f"{summary} Please provide a valid {failed_field} so we can complete your signup.",
                    False,
                )
            try:
                capture_message = self._run_lead_capture_safely(payload)
            except ValueError as exc:
                return (
                    f"{slot_summary(state)} {exc}",
                    False,
                )
            self._memory.mark_lead_captured(session_id)
            self._memory.set_lead_capture_fingerprint(session_id, fingerprint)
            return (
                f"Thanks, all required details are collected. {capture_message}",
                True,
            )

        next_field = next_missing_field(state)
        summary = slot_summary(state)

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

    def _run_lead_capture_safely(self, payload: LeadCapturePayload) -> str:
        now = time.monotonic()
        if now < self._circuit_open_until:
            raise ValueError("Lead capture is temporarily unavailable. Please try again shortly.")

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                self._lead_capture_tool,
                payload.name,
                payload.email,
                payload.platform,
            )
            try:
                result = future.result(timeout=self._tool_timeout_seconds)
                self._tool_failure_count = 0
                return result
            except Exception as exc:
                self._tool_failure_count += 1
                if self._tool_failure_count >= self._circuit_fail_threshold:
                    self._circuit_open_until = now + self._circuit_cooldown_seconds
                raise ValueError("Lead capture failed. Please try again.") from exc


def build_default_agent(storage_path: str | Path | None = None) -> AutoStreamAgent:
    """Factory helper for standard local execution."""
    memory = SimpleMemStore(storage_path=storage_path) if storage_path else SimpleMemStore()
    return AutoStreamAgent(memory_store=memory)


def _greeting_response() -> str:
    return "Hi! I can help with AutoStream plans, features, and policies. What would you like to know?"
