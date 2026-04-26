"""Reference FastAPI webhook with signature verification and idempotent routing."""

from __future__ import annotations

from hashlib import sha256
import hmac
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, Header, HTTPException, Request

from src.graph_agent import build_langgraph_agent

WEBHOOK_SECRET = os.getenv("AUTOSTREAM_WEBHOOK_SECRET", "dev-secret")
agent = build_langgraph_agent()
app = FastAPI(title="AutoStream Webhook Example")


def _verify_signature(raw_body: bytes, signature: Optional[str]) -> None:
    if signature is None:
        raise HTTPException(status_code=401, detail="Missing webhook signature")
    expected = hmac.new(WEBHOOK_SECRET.encode("utf-8"), raw_body, sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/webhooks/whatsapp")
async def whatsapp_webhook(
    request: Request,
    x_autostream_signature: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    raw_body = await request.body()
    _verify_signature(raw_body, x_autostream_signature)

    payload = await request.json()
    session_id = str(payload.get("from", "")).strip()
    text = str(payload.get("text", "")).strip()
    if not session_id or not text:
        raise HTTPException(status_code=400, detail="Payload must include from and text")

    result = agent.handle_message(session_id=session_id, user_message=text)
    return {
        "reply": result.text,
        "intent": result.intent,
        "lead_status": result.lead_status,
        "tool_called": result.tool_called,
    }
