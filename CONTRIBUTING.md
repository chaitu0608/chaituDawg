# Contributing

## Development setup

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements-dev.txt
```

## Local checks before PR

```bash
make test
make lint
make eval
```

## Conventions

- Keep routing deterministic first; avoid adding probabilistic behavior in core flows.
- Do not add product facts outside `data/knowledge_base.json`.
- Add tests for every behavior change:
  - intent routing
  - RAG factual outputs
  - memory persistence
  - lead tool gating
- Prefer explicit, typed state transitions over implicit branching.
- Preserve backward compatibility for persisted session data when changing `src/memory.py`.

## Adding new intents

1. Add deterministic rules in `src/intents.py`.
2. Add tests in `tests/test_intents.py`.
3. Ensure new intent does not bypass KB and tool-gating constraints.

## Updating KB facts

1. Edit `data/knowledge_base.json`.
2. Keep `schema_version` valid and metadata fields (`source`, `last_updated`) complete.
3. Update golden snapshot expectations if canonical facts intentionally changed.
4. Run `make test`.
