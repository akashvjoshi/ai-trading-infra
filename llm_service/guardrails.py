"""
Guardrails — input validation, risk checks, and prompt-injection detection.

Six layers:
  1. Input guard     — reject malicious or oversized user text before it reaches the LLM
  2. Tool guard      — validate LLM-generated tool arguments before execution
  3. Rate guard      — enforce per-session order limits
  4. Liquidity guard — ensure sufficient market depth for large orders
  5. Time guard      — prevent rapid-fire trading patterns
  6. Behavioral guard — detect suspicious trading patterns
"""
from __future__ import annotations

import re
import time
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

# ─────────────────────── New Guardrail Limits ─────────────────────
MAX_LIQUIDITY_RATIO    = Decimal("0.5")   # Order can't exceed 50% of available depth
MIN_ORDER_INTERVAL     = 2.0             # Minimum seconds between orders
MAX_POSITION_SIZE      = Decimal("500")   # Max total ETH position
WASH_TRADE_WINDOW      = 60.0            # Seconds to check for wash trading

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
        available_depth:  Optional[dict]   = None,  # {'bids': Decimal, 'asks': Decimal}
    ) -> None:
        self.max_orders      = max_orders
        self.reference_price = reference_price   # e.g. current mid price from orderbook
        self.available_depth = available_depth or {'bids': Decimal('0'), 'asks': Decimal('0')}
        self._orders_placed  = 0
        self._last_order_time = 0.0
        self._position_size   = Decimal('0')  # Net ETH position (positive = long ETH)
        self._recent_orders   = []  # List of (timestamp, side, price, qty) for wash trade detection

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

    def update_available_depth(self, bids: Decimal, asks: Decimal) -> None:
        """Update available market depth for liquidity checks."""
        self.available_depth = {'bids': bids, 'asks': asks}

    # ────────────────────── New Guardrail Methods ──────────────────

    def _check_liquidity_guard(self, side: str, quantity: Decimal) -> Optional[GuardResult]:
        """Check if order size exceeds available market depth."""
        if side.upper() == "BUY":
            available = self.available_depth.get('asks', Decimal('0'))
        else:
            available = self.available_depth.get('bids', Decimal('0'))

        if available > 0 and quantity > available * MAX_LIQUIDITY_RATIO:
            return GuardResult(
                False,
                f"Order size {quantity} ETH exceeds {MAX_LIQUIDITY_RATIO*100}% of available "
                f"{'ask' if side.upper() == 'BUY' else 'bid'} depth ({available} ETH). "
                f"Consider smaller orders or use quote() to check fill impact."
            )
        return None

    def _check_time_guard(self) -> Optional[GuardResult]:
        """Prevent orders placed too quickly in succession."""
        current_time = time.time()
        if current_time - self._last_order_time < MIN_ORDER_INTERVAL:
            return GuardResult(
                False,
                f"Orders must be at least {MIN_ORDER_INTERVAL} seconds apart. "
                f"Please wait before placing another order."
            )
        return None

    def _check_behavioral_guard(self, side: str, price: Decimal, quantity: Decimal) -> Optional[GuardResult]:
        """Detect suspicious trading patterns like potential wash trading."""
        current_time = time.time()

        # Clean old orders outside the wash trade window
        self._recent_orders = [
            order for order in self._recent_orders
            if current_time - order[0] <= WASH_TRADE_WINDOW
        ]

        # Check for round-trip trading (buy then immediate sell at similar price)
        opposing_recent_orders = [
            order for order in self._recent_orders
            if order[1] != side.upper()  # Opposite side
        ]

        for timestamp, opp_side, opp_price, opp_qty in opposing_recent_orders:
            price_diff_pct = abs(price - opp_price) / opp_price * 100
            if price_diff_pct < 5.0:  # Within 5% price range
                return GuardResult(
                    False,
                    "Potential wash trading detected. Orders on both sides at similar prices "
                    "within the last minute are not allowed."
                )

        return None

    def _update_position_and_history(self, side: str, price: Decimal, quantity: Decimal) -> None:
        """Update position tracking and order history after successful order."""
        # Update position (simplified - doesn't account for fills, just intent)
        if side.upper() == "BUY":
            self._position_size += quantity
        else:
            self._position_size -= quantity

        # Cap position size
        if abs(self._position_size) > MAX_POSITION_SIZE:
            self._position_size = MAX_POSITION_SIZE if self._position_size > 0 else -MAX_POSITION_SIZE

        # Update order history for behavioral analysis
        current_time = time.time()
        self._recent_orders.append((current_time, side.upper(), price, quantity))
        self._last_order_time = current_time

    def _check_place_order(self, args: dict) -> GuardResult:
        # Rate limit
        if self._orders_placed >= self.max_orders:
            return GuardResult(False, f"Session order limit reached ({self.max_orders} orders)")

        # Parse and validate quantity
        try:
            qty = Decimal(str(args.get("quantity", "")))
        except InvalidOperation:
            return GuardResult(False, "Invalid quantity — must be a decimal number")
        if qty < MIN_ORDER_QTY:
            return GuardResult(False, f"Quantity {qty} below minimum ({MIN_ORDER_QTY} ETH)")
        if qty > MAX_ORDER_QTY:
            return GuardResult(False, f"Quantity {qty} exceeds maximum ({MAX_ORDER_QTY} ETH)")

        # Parse and validate price
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

        side = args.get("side", "").upper()
        if side not in ["BUY", "SELL"]:
            return GuardResult(False, "Invalid side — must be BUY or SELL")

        # ── New Guardrails ───────────────────────────────────────

        # Liquidity guard
        liquidity_check = self._check_liquidity_guard(side, qty)
        if liquidity_check:
            return liquidity_check

        # # Time guard
        # time_check = self._check_time_guard()
        # if time_check:
        #     return time_check

        # Behavioral guard
        behavioral_check = self._check_behavioral_guard(side, price, qty)
        if behavioral_check:
            return behavioral_check

        # Position size check
        projected_position = self._position_size + qty if side == "BUY" else self._position_size - qty
        if abs(projected_position) > MAX_POSITION_SIZE:
            return GuardResult(
                False,
                f"Projected position ({projected_position} ETH) would exceed maximum "
                f"position size ({MAX_POSITION_SIZE} ETH). Current position: {self._position_size} ETH."
            )

        # If all checks pass, update tracking
        self._orders_placed += 1
        self._update_position_and_history(side, price, qty)

        return GuardResult(True)
