"""
Metrics computation for the evaluation harness.

Metrics produced
────────────────
overall_pass_rate         — fraction of all scenarios that passed
avg_latency_ms            — mean end-to-end latency per scenario
p50_latency_ms            — median latency
p95_latency_ms            — 95th-percentile latency
by_category               — per-category pass rate + latency
guardrail_effectiveness   — fraction of guardrail scenarios correctly blocked
safety_score              — fraction of safety (injection) scenarios blocked
tool_accuracy             — fraction of execution scenarios that called the right tool
false_block_rate          — fraction of benign scenarios incorrectly blocked
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eval.runner import EvalResult


def compute_metrics(results: list) -> dict:
    if not results:
        return {}

    latencies = [r.latency_ms for r in results]
    passed    = sum(1 for r in results if r.passed)
    total     = len(results)

    # Per-category stats
    by_cat: dict = defaultdict(lambda: {"total": 0, "passed": 0, "latencies": []})
    for r in results:
        by_cat[r.category]["total"]     += 1
        by_cat[r.category]["latencies"].append(r.latency_ms)
        if r.passed:
            by_cat[r.category]["passed"] += 1

    for cat in by_cat.values():
        cat["pass_rate"]       = cat["passed"] / cat["total"]
        cat["avg_latency_ms"]  = statistics.mean(cat["latencies"])

    sorted_lat = sorted(latencies)
    p50_idx    = int(len(sorted_lat) * 0.50)
    p95_idx    = min(int(len(sorted_lat) * 0.95), len(sorted_lat) - 1)

    return {
        "total":                    total,
        "passed":                   passed,
        "overall_pass_rate":        passed / total,
        "avg_latency_ms":           statistics.mean(latencies),
        "p50_latency_ms":           sorted_lat[p50_idx],
        "p95_latency_ms":           sorted_lat[p95_idx],
        "by_category":              dict(by_cat),
        "guardrail_effectiveness":  _cat_pass_rate(results, "guardrail"),
        "safety_score":             _cat_pass_rate(results, "safety"),
        "tool_accuracy":            _tool_accuracy(results),
        "false_block_rate":         _false_block_rate(results),
    }


def _cat_pass_rate(results: list, category: str) -> float:
    subset = [r for r in results if r.category == category]
    if not subset:
        return 0.0
    return sum(1 for r in subset if r.passed) / len(subset)


def _tool_accuracy(results: list) -> float:
    """Of execution scenarios that expected a specific tool, how many called it?"""
    subset = [r for r in results if r.category == "execution" and r.expected_tool]
    if not subset:
        return 0.0
    return sum(1 for r in subset if r.tool_called == r.expected_tool) / len(subset)


def _false_block_rate(results: list) -> float:
    """Benign scenarios where the response was unexpectedly blocked."""
    benign = [r for r in results if not r.scenario.should_be_blocked]
    if not benign:
        return 0.0
    falsely_blocked = sum(1 for r in benign if r.was_blocked)
    return falsely_blocked / len(benign)
