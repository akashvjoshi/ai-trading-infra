"""
Evaluation runner.

Usage
─────
  python -m eval.runner                        # run all scenarios
  python -m eval.runner --category execution   # filter by category
  python -m eval.runner --id exec_001          # single scenario
  python -m eval.runner --json out.json        # save results
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Optional

from eval.metrics import compute_metrics
from eval.scenarios import SCENARIOS, Scenario
from llm_service.service import TradingService

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    scenario_id:   str
    category:      str
    description:   str
    passed:        bool
    latency_ms:    float
    tool_called:   Optional[str]
    was_blocked:   bool
    response:      str
    expected_tool: Optional[str]
    error:         Optional[str]    = None
    scenario:      Optional[object] = None   # back-reference, excluded from JSON


class EvalRunner:
    def __init__(self, grpc_addr: str = "localhost:50051") -> None:
        self.grpc_addr = grpc_addr
        self.results:  list[EvalResult] = []

    def run(
        self,
        scenarios: list[Scenario] | None = None,
        category:  str | None             = None,
        scenario_id: str | None           = None,
    ) -> list[EvalResult]:
        pool = scenarios or SCENARIOS
        if category:
            pool = [s for s in pool if s.category == category]
        if scenario_id:
            pool = [s for s in pool if s.id == scenario_id]

        self.results = []
        for s in pool:
            result = self._run_one(s)
            self.results.append(result)
            icon = "✓" if result.passed else "✗"
            logger.info("[%s] %-12s %s (%.0fms) tool=%s",
                        icon, s.id, s.description, result.latency_ms,
                        result.tool_called or "—")

        return self.results

    # ─────────────────── single scenario ──────────────────────────

    def _run_one(self, scenario: Scenario) -> EvalResult:
        svc = TradingService(grpc_addr=self.grpc_addr)
        tool_called  = None
        was_blocked  = False
        response_txt = ""
        error        = None

        t0 = time.perf_counter()
        try:
            response_txt = svc.chat(scenario.user_input)
            # Inspect tool calls made during this turn
            if svc.last_tool_calls:
                tool_called = svc.last_tool_calls[0]["name"]
            was_blocked = (
                response_txt.startswith("[BLOCKED]")
                or "blocked" in response_txt[:50].lower()
            )
        except Exception as exc:
            error       = str(exc)
            was_blocked = False

        latency_ms = (time.perf_counter() - t0) * 1000

        passed = self._evaluate(scenario, tool_called, was_blocked)

        return EvalResult(
            scenario_id   = scenario.id,
            category      = scenario.category,
            description   = scenario.description,
            passed        = passed,
            latency_ms    = latency_ms,
            tool_called   = tool_called,
            was_blocked   = was_blocked,
            response      = response_txt[:200],
            expected_tool = scenario.expected_tool,
            error         = error,
            scenario      = scenario,
        )

    def _evaluate(
        self,
        scenario:    Scenario,
        tool_called: Optional[str],
        was_blocked: bool,
    ) -> bool:
        if scenario.should_be_blocked:
            return was_blocked

        if scenario.expected_tool:
            return tool_called == scenario.expected_tool

        # Robustness / open-ended: pass if no crash and not incorrectly blocked
        return not was_blocked

    # ─────────────────────── Reporting ────────────────────────────

    def print_report(self) -> None:
        metrics = compute_metrics(self.results)
        print()
        print("=" * 65)
        print("  AI TRADING INFRASTRUCTURE — EVALUATION REPORT")
        print("=" * 65)

        for cat, stats in sorted(metrics["by_category"].items()):
            print(f"\n  {cat.upper()}  ({stats['pass_rate']:.0%} pass | "
                  f"avg {stats['avg_latency_ms']:.0f}ms)")
            for r in self.results:
                if r.category != cat:
                    continue
                icon    = "✓" if r.passed else "✗"
                blocked = " [blocked]" if r.was_blocked else ""
                tool    = f" → {r.tool_called}" if r.tool_called else ""
                print(f"    {icon} [{r.scenario_id}] {r.description}{tool}{blocked}  "
                      f"({r.latency_ms:.0f}ms)")

        print()
        print(f"  OVERALL PASS RATE  : {metrics['overall_pass_rate']:.0%}  "
              f"({metrics['passed']}/{metrics['total']})")
        print(f"  LATENCY            : avg {metrics['avg_latency_ms']:.0f}ms  "
              f"p50 {metrics['p50_latency_ms']:.0f}ms  "
              f"p95 {metrics['p95_latency_ms']:.0f}ms")
        print(f"  TOOL ACCURACY      : {metrics['tool_accuracy']:.0%}")
        print(f"  GUARDRAIL EFFECT.  : {metrics['guardrail_effectiveness']:.0%}")
        print(f"  SAFETY SCORE       : {metrics['safety_score']:.0%}")
        print(f"  FALSE BLOCK RATE   : {metrics['false_block_rate']:.0%}")
        print("=" * 65)

    def save_json(self, path: str) -> None:
        data = []
        for r in self.results:
            d = asdict(r)
            d.pop("scenario", None)
            data.append(d)
        with open(path, "w") as fh:
            json.dump({"results": data, "metrics": compute_metrics(self.results)}, fh, indent=2)
        logger.info("Results saved to %s", path)


# ───────────────────────────── CLI ────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run LLM trading evaluation")
    parser.add_argument("--grpc",     default="localhost:50051")
    parser.add_argument("--category", default=None, help="Filter by category")
    parser.add_argument("--id",       default=None, help="Run a single scenario by ID")
    parser.add_argument("--json",     default=None, metavar="FILE", help="Save JSON results")
    args = parser.parse_args()

    runner = EvalRunner(grpc_addr=args.grpc)
    runner.run(category=args.category, scenario_id=args.id)
    runner.print_report()
    if args.json:
        runner.save_json(args.json)


if __name__ == "__main__":
    main()
