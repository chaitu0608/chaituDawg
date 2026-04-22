# Goal

Build a production-grade conversational AI agent for AutoStream in controlled, testable phases.

## Execution Rules

1. Implement exactly one phase at a time.
2. Keep each phase runnable before moving forward.
3. Preserve existing behavior after each change.
4. Prefer deterministic logic over probabilistic behavior.
5. Never answer product facts outside the local KB.

# Phase 1 to Phase 7 Tasks

## Phase 1: Project scaffold and data design

- Create folders: src/, tests/, data/, docs/.
- Create data/knowledge_base.json with canonical pricing, features, and policies.
- Create requirements.txt with pinned dependencies.
- Create this iteration guide.

## Phase 2: Intent classifier

- Implement hybrid intent detection in src/intents.py.
- Add deterministic keyword/rule routing first.
- Add optional LLM fallback only for ambiguous inputs.
- Add tests/test_intents.py with greeting, inquiry, and high-intent coverage.

## Phase 3: Local RAG layer

- Implement KB loading and retrieval in src/rag.py.
- Route pricing/features/policy questions to local KB only.
- Enforce concise structured answers from retrieved context.
- Add tests/test_rag_accuracy.py for exact fact assertions.

## Phase 4: SimpleMem integration

- Implement memory wrapper in src/memory.py.
- Persist conversation messages, detected intent, and lead slots.
- Restore state across at least 5-6 turns.
- Add memory state inspection helpers for debug logging.

## Phase 5: Lead qualification and tool gating

- Implement graph/orchestration flow in src/agent.py.
- Implement mock_lead_capture and validators in src/tools.py.
- Collect fields in order: name, email, platform.
- Ask one missing field per turn and summarize collected slots.
- Enforce one-time tool execution only when all slots are present.

## Phase 6: Testing and metrics

- Add or expand tests for intent, RAG, memory retention, and tool gating.
- Add evaluation script for lightweight metrics reporting.
- Verify no premature tool execution with partial slot inputs.

## Phase 7: README and demo readiness

- Create README.md with setup, architecture, and run instructions.
- Document practical WhatsApp webhook integration path.
- Provide demo script showing Q&A, intent shift, slot filling, and successful capture.

# Definition of Done by Phase

## Phase 1

- Repository structure exists.
- Local KB contains exact canonical facts.
- Dependencies are pinned.
- Iteration plan is documented.

## Phase 2

- Three intent classes are implemented reliably.
- Rule-first classifier works and tests pass.

## Phase 3

- Pricing and policy answers come only from local KB.
- Tests confirm factual outputs with exact assertions.

## Phase 4

- State persists for 5-6 turns.
- Lead slots and conversation context are retained and restorable.

## Phase 5

- High intent triggers slot filling.
- Tool cannot run before all required fields exist.
- Tool runs exactly once after completion.

## Phase 6

- Automated checks report classification, factual accuracy, tool gating, and memory retention.
- Core test suite passes.

## Phase 7

- New user can run the project end-to-end locally.
- Architecture and integration notes are clear and practical.
- Demo flow is repeatable.

# Testing Checklist

- Run pytest after each implementation phase.
- Keep tests deterministic and independent of remote services.
- Verify intent routing for greeting, inquiry, and high intent.
- Verify exact KB-backed answers for pricing and policies.
- Verify slot-filling asks only one missing field at a time.
- Verify mock_lead_capture does not execute early.
- Verify state retention across at least six turns.

# Final Submission Checklist

- All required files exist and are organized by responsibility.
- Local KB is the single source of truth for product facts.
- Lead capture gating is safe and deterministic.
- Memory behavior is testable and persistent.
- Test suite passes and demonstrates guardrails.
- Documentation supports local run and demo execution.
