.PHONY: proto engine mcp eval test demo clean

# ── Proto generation ───────────────────────────────────────────────
proto:
	python scripts/generate_proto.py

# ── Run individual services ────────────────────────────────────────
engine:
	python -m engine.server

mcp:
	python -m mcp_bridge.server

# ── Evaluation harness ─────────────────────────────────────────────
eval:
	python -m eval.runner

eval-json:
	python -m eval.runner --json eval_results.json

eval-category:
	python -m eval.runner --category $(CATEGORY)

# ── Interactive demo (starts engine + REPL in one process) ────────
demo:
	python scripts/demo.py

# ── Unit tests (no LLM, no gRPC required) ─────────────────────────
test:
	pytest tests/test_clob.py tests/test_guardrails.py -v

# ── Integration tests (requires gRPC engine) ──────────────────────
test-integration:
	pytest tests/test_integration.py -v -s

# ── Full test suite ────────────────────────────────────────────────
test-all:
	pytest -v

# ── Cleanup ───────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -f eval_results.json
