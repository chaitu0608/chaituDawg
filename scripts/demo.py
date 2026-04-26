"""Run a deterministic demo conversation for AutoStream."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.graph_agent import build_langgraph_agent


def main() -> int:
    storage_path = REPO_ROOT / "data" / "demo_session_memory.json"
    if storage_path.exists():
        storage_path.unlink()

    agent = build_langgraph_agent(storage_path=storage_path)
    session_id = "demo-session-1"

    scripted_user_messages = (
        "Hey there",
        "What are your pricing plans?",
        "I want to try the Pro plan",
        "My name is Maya",
        "maya@example.com",
        "YouTube",
        "Sign me up again",
    )

    print("=== AutoStream Demo ===")
    print(f"Session: {session_id}")
    print("")

    for turn_index, user_message in enumerate(scripted_user_messages, start=1):
        result = agent.handle_message(session_id=session_id, user_message=user_message)

        print(f"Turn {turn_index}")
        print(f"User: {user_message}")
        print(f"Agent: {result.text}")
        print(f"Intent: {result.intent}")
        print(f"Lead Status: {result.lead_status}")
        print(f"Lead Fields: {result.lead_fields}")
        print(f"Tool Called: {result.tool_called}")
        print("-" * 72)

    state = agent.inspect_session_state(session_id)
    print("Final Persisted State Snapshot")
    print(f"Turn Count: {state['turn_count']}")
    print(f"Lead Status: {state['lead_status']}")
    print(f"Intent: {state['intent']}")
    print(f"Memory Snapshot: {state['memory_snapshot']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
