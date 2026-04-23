"""Golden snapshot checks for canonical knowledge-base facts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.rag import DEFAULT_KB_PATH


def test_kb_file_matches_expected_golden_hash() -> None:
    kb_payload = json.loads(Path(DEFAULT_KB_PATH).read_text(encoding="utf-8"))
    canonical_json = json.dumps(kb_payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    # If this fails, canonical KB changed. Review and intentionally update hash.
    assert digest == "bd3d4876cee65250266935e6eda494e6c448d81d591ad1351410305f1336f04b"


def test_kb_core_fact_snapshot() -> None:
    kb_payload = json.loads(Path(DEFAULT_KB_PATH).read_text(encoding="utf-8"))
    assert kb_payload["plans"][0]["price"] == "$29/month"
    assert kb_payload["plans"][1]["price"] == "$79/month"
    assert kb_payload["policies"]["refund_policy"]["text"] == "No refunds after 7 days"
