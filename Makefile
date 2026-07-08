.PHONY: install test run offline
install:
	pip install -e ".[dev,real]"
test:
	pytest -q
run:
	agent-memory-eval --json results.json
offline:
	agent-memory-eval --offline
