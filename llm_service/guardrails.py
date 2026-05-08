"""
Guardrails — input validation, risk checks, and prompt-injection detection.

Three layers:
  1. Input guard  — reject malicious or oversized user text before it reaches the LLM
  2. Tool guard   — validate LLM-generated tool arguments before execution
  3. Rate guard   — enforce per-session order limits
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Optional

# ──────────────────────────── Limits ──────────────────────────────
MAX_ORDER_QTY          = Decimal("100")
MIN_ORDER_QTY          = Decimal("0.001")
MAX_PRICE              = Decimal("1_000_000")
MIN_PRICE              = Decimal("0.01")
MAX_PRICE_DEVIATION    = Decimal("30")   # % deviation from mid price
MAX_ORDERS_PER_SESSION = 50
MAX_INPUT_LENGTH       = 2_000

# ─────────────────────── Injection Patterns ───────────────────────
_INJECTION_PATTERNS: list[str] = [
    r"ignore\s+(previous|prior|all)\s+instructions?",
    r"disregard\s+(your|the|all)\s+instructions?",
    r"you\s+are\s+now\s+",
    r"pretend\s+(that\s+)?you",
    r"act\s+as\s+(an?\s+)?unrestricted",
    r"\bDAN\b",
    r"\bjailbreak\b",
    r"</?(system|assistant|user)>",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"\[INST\]",
    r"###\s*System",
    r"bypass\s+(safety|guardrail|restriction)",
    r"unlimited\s+order",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


@dataclass
class GuardResult:
    allowed: bool
    reason:  Optional[str] = None


class Guardrails:
    def __init__(
        self,
        max_orders:       int              = MAX_ORDERS_PER_SESSION,
        reference_price:  Optional[Decimal] = None,
    ) -> None:
        self.max_orders      = max_orders
        self.reference_price = reference_price   # e.g. current mid price from orderbook
        self._orders_placed  = 0

    # ────────────────────── Public checks ─────────────────────────

    def check_input(self, text: str) -> GuardResult:
        """Run before sending user text to the LLM."""
        if len(text) > MAX_INPUT_LENGTH:
            return GuardResult(False, f"Input too long ({len(text)} chars, max {MAX_INPUT_LENGTH})")
        for pattern in _COMPILED:
            if pattern.search(text):
                return GuardResult(False, "Potential prompt injection detected — request rejected")
        return GuardResult(True)

    def check_tool_call(self, tool_name: str, args: dict) -> GuardResult:
        """Run after the LLM emits a tool_use block, before executing it."""
        if tool_name == "place_order":
            return self._check_place_order(args)
        return GuardResult(True)

    def update_reference_price(self, price: Decimal) -> None:
        self.reference_price = price

    # ────────────────────── Private helpers ───────────────────────

    def _check_place_order(self, args: dict) -> GuardResult:
        # Rate limit
        if self._orders_placed >= self.max_orders:
            return GuardResult(False, f"Session order limit reached ({self.max_orders} orders)")

        # Quantity
        try:
            qty = Decimal(str(args.get("quantity", "")))
        except InvalidOperation:
            return GuardResult(False, "Invalid quantity — must be a decimal number")
        if qty < MIN_ORDER_QTY:
            return GuardResult(False, f"Quantity {qty} below minimum ({MIN_ORDER_QTY} ETH)")
        if qty > MAX_ORDER_QTY:
            return GuardResult(False, f"Quantity {qty} exceeds maximum ({MAX_ORDER_QTY} ETH)")

        # Price
        try:
            price = Decimal(str(args.get("price", "")))
        except InvalidOperation:
            return GuardResult(False, "Invalid price — must be a decimal number")
        if price < MIN_PRICE:
            return GuardResult(False, f"Price ${price} below minimum (${MIN_PRICE})")
        if price > MAX_PRICE:
            return GuardResult(False, f"Price ${price} exceeds maximum (${MAX_PRICE})")

        # Price deviation check vs reference mid
        if self.reference_price and self.reference_price > 0:
            deviation_pct = abs(price - self.reference_price) / self.reference_price * 100
            if deviation_pct > MAX_PRICE_DEVIATION:
                return GuardResult(
                    False,
                    f"Price ${price} deviates {deviation_pct:.1f}% from mid "
                    f"(${self.reference_price}). Max allowed: {MAX_PRICE_DEVIATION}%.",
                )

        self._orders_placed += 1
        return GuardResult(True)
