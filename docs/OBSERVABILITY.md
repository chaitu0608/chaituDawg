# Observability and Incident Checklist

## Suggested Structured Log Keys

- `session_id`: stable chat/session identifier
- `intent`: classified intent label
- `route_state`: greeting / inquiry / lead_flow
- `lead_status`: new / collecting / ready / captured
- `tool_called`: whether lead capture tool executed this turn
- `rate_limited`: true when hook blocked processing
- `circuit_open`: true when lead-capture circuit breaker is active
- `kb_query_type`: plan_details / plan_comparison / refund_policy / support_policy / unknown

## Recommended Metrics

- `autostream_messages_total{intent=...}`
- `autostream_rate_limited_total`
- `autostream_lead_capture_attempts_total`
- `autostream_lead_capture_success_total`
- `autostream_lead_capture_fail_total`
- `autostream_circuit_open_total`
- `autostream_kb_unknown_query_total`

## Incident Debug Flow

1. Confirm webhook signature failures vs application failures.
2. Check rate-limited sessions (`intent=rate_limited` responses).
3. Inspect `lead_status` progression and retry counters (`email_retry_count`, `platform_retry_count`).
4. If captures fail repeatedly, check circuit state and downstream tool availability.
5. Verify KB schema version and canonical facts (`schema_version` in `data/knowledge_base.json`).
6. Re-run validation suite:
   - `python -m pytest -q`
   - `python scripts/evaluate.py --output-dir reports`
