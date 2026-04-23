PYTHON ?= ./.venv/bin/python

.PHONY: test demo eval lint typecheck coverage

test:
	$(PYTHON) -m pytest -q

demo:
	$(PYTHON) scripts/demo.py

eval:
	$(PYTHON) scripts/evaluate.py --output-dir reports

lint:
	$(PYTHON) -m mypy src

typecheck: lint

coverage:
	$(PYTHON) -m pytest -q --cov=src.agent --cov=src.memory --cov-branch --cov-fail-under=90
