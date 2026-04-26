"""LangGraph wrapper around deterministic AutoStream agent logic.

This module keeps existing deterministic routing, RAG, and tool gating intact.
LangGraph is used as an orchestration layer, while GPT-4o-mini is optional and
only used to lightly polish greeting text.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Literal, Optional, TypedDict, Union

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from src.agent import AgentResponse, AutoStreamAgent, build_default_agent


class GraphState(TypedDict, total=False):
    """State payload passed between LangGraph nodes."""

    session_id: str
    user_message: str
    response: AgentResponse
    route: str


class LangGraphAutoStreamAgent:
    """LangGraph orchestrator that delegates business logic to AutoStreamAgent."""

    def __init__(
        self,
        core_agent: Optional[AutoStreamAgent] = None,
        *,
        model_name: str = "gpt-4o-mini",
        enable_llm_polish: bool = True,
    ) -> None:
        self._core_agent = core_agent or build_default_agent()
        self._model_name = model_name
        self._llm = self._build_llm(enable_llm_polish=enable_llm_polish, model_name=model_name)
        self._graph = self._build_graph()

    def handle_message(self, session_id: str, user_message: str) -> AgentResponse:
        """Run one LangGraph-orchestrated agent turn."""
        result = self._graph.invoke({"session_id": session_id, "user_message": user_message})
        return result["response"]

    def inspect_session_state(self, session_id: str, *, include_pii: bool = False) -> Dict[str, object]:
        """Expose deterministic persisted state from the core agent."""
        return self._core_agent.inspect_session_state(session_id, include_pii=include_pii)

    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("core_turn", self._node_core_turn)
        graph.add_node("greeting_route", self._node_greeting_route)
        graph.add_node("inquiry_route", self._node_inquiry_route)
        graph.add_node("lead_route", self._node_lead_route)
        graph.add_node("maybe_polish", self._node_maybe_polish)

        graph.add_edge(START, "core_turn")
        graph.add_conditional_edges(
            "core_turn",
            self._route_after_core,
            {
                "greeting_route": "greeting_route",
                "inquiry_route": "inquiry_route",
                "lead_route": "lead_route",
            },
        )
        graph.add_edge("greeting_route", "maybe_polish")
        graph.add_edge("inquiry_route", "maybe_polish")
        graph.add_edge("lead_route", "maybe_polish")
        graph.add_edge("maybe_polish", END)
        return graph.compile()

    def _node_core_turn(self, state: GraphState) -> GraphState:
        response = self._core_agent.handle_message(
            session_id=state["session_id"],
            user_message=state["user_message"],
        )
        return {"response": response}

    def _route_after_core(self, state: GraphState) -> Literal["greeting_route", "inquiry_route", "lead_route"]:
        intent = state["response"].intent
        if intent == "product_or_pricing_inquiry":
            return "inquiry_route"
        if intent == "high_intent_lead":
            return "lead_route"
        return "greeting_route"

    def _node_greeting_route(self, state: GraphState) -> GraphState:
        return {"route": "greeting"}

    def _node_inquiry_route(self, state: GraphState) -> GraphState:
        return {"route": "inquiry"}

    def _node_lead_route(self, state: GraphState) -> GraphState:
        return {"route": "lead"}

    def _node_maybe_polish(self, state: GraphState) -> GraphState:
        response = state["response"]
        if state.get("route") != "greeting" or self._llm is None:
            return {"response": response}

        try:
            raw_content = self._llm.invoke(
                [
                    (
                        "system",
                        "You are a concise assistant. Rewrite the greeting in one sentence. "
                        "Do not add any new product facts or pricing.",
                    ),
                    ("human", response.text),
                ]
            ).content
            polished_text = raw_content.strip() if isinstance(raw_content, str) else str(raw_content).strip()
        except Exception:
            return {"response": response}

        if not polished_text:
            return {"response": response}

        return {
            "response": AgentResponse(
                text=polished_text,
                intent=response.intent,
                lead_status=response.lead_status,
                lead_fields=response.lead_fields,
                tool_called=response.tool_called,
            )
        }

    @staticmethod
    def _build_llm(*, enable_llm_polish: bool, model_name: str) -> Optional[ChatOpenAI]:
        if not enable_llm_polish:
            return None
        if not os.getenv("OPENAI_API_KEY"):
            return None
        return ChatOpenAI(model=model_name, temperature=0)


def build_langgraph_agent(
    storage_path: Optional[Union[str, Path]] = None,
    *,
    model_name: str = "gpt-4o-mini",
    enable_llm_polish: bool = True,
) -> LangGraphAutoStreamAgent:
    """Factory helper for LangGraph-based orchestration."""
    core_agent = build_default_agent(storage_path=storage_path)
    return LangGraphAutoStreamAgent(
        core_agent=core_agent,
        model_name=model_name,
        enable_llm_polish=enable_llm_polish,
    )
