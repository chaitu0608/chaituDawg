# AutoStream Social-to-Lead Agentic Workflow

Production-quality conversational AI workflow for a fictional SaaS product (AutoStream) with:

- deterministic intent routing
- local KB-backed Q&A (no runtime web search)
- durable multi-turn memory
- strict lead tool gating (name, email, platform required)

## What Is Implemented

- Phase 1: scaffold, pinned dependencies, local knowledge base
- Phase 2: hybrid intent classifier (rule-first, optional fallback)
- Phase 3: deterministic local RAG answers from JSON KB
- Phase 4: SimpleMem-style persistent state store
- Phase 5: lead qualification flow and one-time gated tool execution
- Phase 6: benchmark metrics + report generation
- Phase 7: README, demo script, practical webhook integration notes

## Project Structure

- `src/intents.py`: intent labels and rule-first classifier
- `src/rag.py`: local KB loading, retrieval, deterministic response formatting
- `src/memory.py`: durable session and slot state persistence
- `src/tools.py`: extraction/validation helpers and mock lead tool
- `src/agent.py`: orchestration and routing across greeting, inquiry, and lead flow
- `src/evaluation.py`: metric computations for phase 6
- `scripts/demo.py`: demo conversation runner
- `scripts/evaluate.py`: metrics report generator
- `data/knowledge_base.json`: canonical product and policy facts
- `tests/`: unit tests for intents, RAG, memory, tool gating, and phase-6 thresholds
- `reports/`: generated metric reports

## Architecture

The runtime flow is intentionally deterministic:

1. User input is classified by `IntentClassifier`.
2. `AutoStreamAgent` routes by intent:
   - `casual_greeting` -> short greeting response
   - `product_or_pricing_inquiry` -> `answer_from_kb` in `src/rag.py`
   - `high_intent_lead` or active collecting session -> lead slot flow
3. Lead slot flow updates only required slots:
   - `name`
   - `email`
   - `platform`
4. Tool execution gate checks all slots are complete before calling `mock_lead_capture`.
5. `SimpleMemStore` checkpoints state every turn.

### State Model

Durable state fields include:

- `messages`
- `intent`
- `lead_status`
- `lead_name`
- `lead_email`
- `lead_platform`
- `kb_context`
- `turn_count`
- `memory_snapshot`

## Local Setup

### Requirements

- Python 3.10+

### Install

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

### Run Tests

```bash
./.venv/bin/python -m pytest -q
```

### Run Demo

```bash
./.venv/bin/python scripts/demo.py
```

### Run Evaluation + Report

```bash
./.venv/bin/python scripts/evaluate.py --output-dir reports
```

Artifacts will be written to:

- `reports/phase6_metrics.json`
- `reports/phase6_metrics.md`

## Behavior Guarantees

- No runtime web search.
- Pricing/features/policies answered only from local KB.
- No premature lead tool call.
- One missing lead field asked at a time.
- Tool runs exactly once per session after all fields are present.
- Conversation state survives multi-turn flows.

## Demo Flow Covered

The demo script shows:

1. Pricing question answered from KB.
2. Intent shift from inquiry to high-intent lead.
3. Stepwise slot collection (name -> email -> platform).
4. Successful one-time mock lead capture.
5. Follow-up behavior after lead is already captured.

## Practical WhatsApp Webhook Integration

Use a thin API wrapper that forwards inbound WhatsApp messages to `AutoStreamAgent`.

### Recommended endpoint contract

- `POST /webhooks/whatsapp`
- Parse provider payload for:
  - `session_id` (phone number or mapped conversation id)
  - `text` (incoming message)
- Call:
  - `agent.handle_message(session_id=session_id, user_message=text)`
- Return/respond with `AgentResponse.text` via your WhatsApp provider SDK/API.

### Minimal integration shape (FastAPI-style pseudo-code)

```python
from fastapi import FastAPI, Request
from src.agent import build_default_agent

app = FastAPI()
agent = build_default_agent()

@app.post("/webhooks/whatsapp")
async def whatsapp_webhook(request: Request):
    payload = await request.json()
    session_id = payload["from"]
    text = payload["text"]

    result = agent.handle_message(session_id=session_id, user_message=text)

    # send result.text through provider API here
    return {"reply": result.text}
```

### Production checklist

- Verify webhook signatures from provider.
- Normalize and validate incoming text.
- Rate-limit and add idempotency keys for retries.
- Persist session id mapping strategy for multi-device users.
- Add structured logs using `inspect_session_state` for debugging.
- Keep product facts in KB file, not hardcoded in webhook layer.

## Notes

- The repository uses deterministic logic first and keeps LLM fallback optional.
- For demonstration, `mock_lead_capture` prints and returns confirmation text.
- `src/memory.py` intentionally stores conversational state only, not product facts.
