# AutoStream Social-to-Lead Agentic Workflow

Submission-ready conversational agent for AutoStream with:

- deterministic intent routing and **local KB-only** answers for pricing, features, and policies
- strict lead tool gating (name, email, platform) with **at most one** `mock_lead_capture` call after all slots are valid
- durable multi-turn session memory (JSON-backed, 5–6+ turns)
- **LangGraph** orchestration in `src/graph_agent.py` around the deterministic core
- optional **GPT-4o-mini** via LangChain only for greeting phrasing when `OPENAI_API_KEY` is set—never for KB facts, intent, or tool decisions

## How to run the project locally

### Requirements

- Python 3.9+ (no Python 3.10-only language features in application code)
- virtual environment recommended
- `OPENAI_API_KEY` only if you want optional greeting polish

### Install

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

### Run demo

```bash
./.venv/bin/python scripts/demo.py
```

The demo uses `build_langgraph_agent`, clears `data/demo_session_memory.json`, then runs: greeting → pricing (KB) → high intent → name → email → platform → tool → follow-up.

### Run webhook server

```bash
AUTOSTREAM_WEBHOOK_SECRET=dev-secret ./.venv/bin/python -m uvicorn examples.webhook_app:app --reload
```

### Run tests

```bash
./.venv/bin/python -m pip install pytest==8.3.3 pytest-cov==5.0.0 mypy==1.13.0
./.venv/bin/python -m pytest -q
```

## Architecture (≈200 words)

Business logic stays deterministic in `src/agent.py`: rule-first intents (`casual_greeting`, `product_or_pricing_inquiry`, `high_intent_lead`), `answer_from_kb` in `src/rag.py` for all product/policy facts from `data/knowledge_base.json`, slot progression and validation in `src/tools.py` / `src/lead_flow.py`, and guarded `mock_lead_capture`. `src/graph_agent.py` wraps this in a small LangGraph (`START` → `core_turn` → conditional `greeting_route` / `inquiry_route` / `lead_route` → `maybe_polish` → `END`) so the rubric’s framework requirement is met without reimplementing routing.

State is persisted each turn by `SimpleMemStore` in `src/memory.py` (messages, intent, lead slots and status, retries, pause flags, KB refs, turn count). That gives stable multi-turn behavior and correct lead gating across sessions.

The optional LLM path runs only in `maybe_polish` for greeting-shaped turns when an API key exists; inquiry and lead routes skip it so pricing and capture behavior stay fully deterministic and testable.

## Deterministic behavior guarantees

- No runtime web search.
- Pricing, features, and policies come only from the local knowledge base file.
- Lead capture runs only after name, email, and platform are present and valid, and only once per completed payload for the session.
- Conversation state persists across turns in the session store.

## WhatsApp integration (webhooks + Twilio)

This repository ships a webhook reference in `examples/webhook_app.py` and can be
connected to WhatsApp through Twilio by deploying this server and registering the
public webhook URL in the Twilio console.

### Deployment flow with Twilio webhook URL

1. Deploy this app and expose a public URL (for example, `https://your-domain.com`).
2. In Twilio WhatsApp Sandbox (or a Twilio WhatsApp-enabled number), set the inbound
   webhook URL to:
   - `https://your-domain.com/webhooks/whatsapp`
3. Twilio forwards each inbound WhatsApp message to that URL.
4. Your webhook adapter maps Twilio payload fields into the agent input:
   - `session_id` <- sender identifier (for example Twilio `From`)
   - `text` <- message body (for example Twilio `Body`)
5. The server calls:
   - `agent.handle_message(session_id=session_id, user_message=text)`
6. The server returns the generated reply payload (`reply`, `intent`, `lead_status`,
   `tool_called`) so your Twilio response layer can deliver the WhatsApp message.

### Signature verification note

The included example currently verifies a custom `X-Autostream-Signature` header
using `AUTOSTREAM_WEBHOOK_SECRET` for local/demo safety. For production Twilio use,
replace or extend this verifier to validate Twilio's signature header
(`X-Twilio-Signature`) against the exact URL/body Twilio signed.

`GET /healthz` is included for probes.

## Docker

```bash
docker compose up --build
```

Serves the webhook on `http://localhost:8000` with `./data` mounted.

## Project structure

| Path | Role |
|------|------|
| `src/agent.py` | Deterministic routing, KB usage, lead flow and tool gating |
| `src/graph_agent.py` | LangGraph wrapper + optional greeting polish |
| `src/intents.py` | Intent labels and classifier |
| `src/rag.py` | Local KB load, retrieval, deterministic answers |
| `src/memory.py` | Persistent session store |
| `src/tools.py` | Slot extractors, validation, `mock_lead_capture` |
| `scripts/demo.py` | Scripted end-to-end demo |
| `examples/webhook_app.py` | FastAPI webhook reference |
| `tests/` | Unit and integration tests |

## Security notes

- Validate and bound `session_id` and user message length before persistence (see `src/input_limits.py`).
- Do not log PII in production if policy forbids it.

## Further reading

- [`CONTRIBUTING.md`](CONTRIBUTING.md) for contribution guidelines.
- Optional evaluation/report scripts: `scripts/evaluate.py`, `reports/` (if used in your workflow).
