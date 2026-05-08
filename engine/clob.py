"""
Core Central Limit Order Book — price-time priority, thread-safe.

Primary functions
─────────────────
  1. Spot Limit Buy    — place_order(OrderSide.BUY,  price, qty)
  2. Spot Limit Sell   — place_order(OrderSide.SELL, price, qty)
  3. Order Cancellation— cancel_order(order_id)
  4. Fill Quote        — quote_order(side, qty)  [read-only, no state mutation]

Matching rule
─────────────
Price-time priority:
  • Best price executes first (lowest ask for buys, highest bid for sells).
  • Equal-price orders execute in FIFO order (time priority).

Trade price is always the resting order's price (maker sets price).

Thread-safety model
───────────────────
A ReadWriteLock separates reads from writes:
  • Writes (place_order, cancel_order) hold the lock exclusively.
  • Reads  (get_snapshot, get_order, get_trades) share the lock and run
    concurrently with each other.

Efficiency
──────────
  • SortedDict keeps price levels in order at all times — O(log n) insert
    and delete, O(k) range iteration during matching (k = levels consumed).
  • cancel_order removes the order from its deque immediately — O(m) where
    m is the number of orders at that price level, which is typically small.
    This keeps get_snapshot and the matching loop free of any filtered-set
    bookkeeping, making both simpler and provably correct.
  • The trade log is a bounded deque (default 10 000 entries).
"""
from __future__ import annotations

import threading
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from time import time_ns
from typing import Iterator, Optional

from sortedcontainers import SortedDict

MAX_TRADES = 10_000


# ──────────────────────────────────────────────────────────────────
# ReadWriteLock
# ──────────────────────────────────────────────────────────────────

class _RWLock:
    """
    Fair readers-writer lock backed by threading.Condition.

    Many readers may hold the lock simultaneously.
    A writer waits for all current readers to finish, then holds exclusively.
    notify_all() on every release ensures no starvation.
    """

    def __init__(self) -> None:
        self._cond    = threading.Condition(threading.Lock())
        self._readers = 0
        self._writing = False

    @contextmanager
    def read_locked(self) -> Iterator[None]:
        with self._cond:
            while self._writing:
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                if self._readers == 0:
                    self._cond.notify_all()

    @contextmanager
    def write_locked(self) -> Iterator[None]:
        with self._cond:
            while self._writing or self._readers > 0:
                self._cond.wait()
            self._writing = True
        try:
            yield
        finally:
            with self._cond:
                self._writing = False
                self._cond.notify_all()


# ──────────────────────────────────────────────────────────────────
# Domain models
# ──────────────────────────────────────────────────────────────────

class OrderSide(Enum):
    BUY  = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    OPEN             = "OPEN"
    FILLED           = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED        = "CANCELLED"


@dataclass
class Order:
    order_id:        str
    side:            OrderSide
    price:           Decimal
    quantity:        Decimal
    filled_quantity: Decimal     = field(default_factory=Decimal)
    status:          OrderStatus = OrderStatus.OPEN
    created_at:      int         = field(default_factory=time_ns)

    @property
    def remaining_quantity(self) -> Decimal:
        return self.quantity - self.filled_quantity

    def is_active(self) -> bool:
        return self.status in (OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED)


@dataclass
class Trade:
    trade_id:      str
    buy_order_id:  str
    sell_order_id: str
    price:         Decimal
    quantity:      Decimal
    timestamp:     int = field(default_factory=time_ns)


# ──────────────────────────────────────────────────────────────────
# Order Book
# ──────────────────────────────────────────────────────────────────

class OrderBook:
    """
    In-memory CLOB for a single trading pair (e.g. ETH/USDC).

    _bids : SortedDict (ascending keys) — best bid = highest key = last entry
    _asks : SortedDict (ascending keys) — best ask = lowest  key = first entry

    Each key maps to a deque[Order] — FIFO queue enforcing time priority
    within a price level.
    """

    def __init__(self, max_trades: int = MAX_TRADES) -> None:
        self._rwlock = _RWLock()
        self._bids:   SortedDict       = SortedDict()
        self._asks:   SortedDict       = SortedDict()
        self._orders: dict[str, Order] = {}
        self._trades: deque[Trade]     = deque(maxlen=max_trades)

    # ─────────────────── Primary functions ────────────────────────

    def place_order(
        self,
        side:     OrderSide,
        price:    Decimal,
        quantity: Decimal,
    ) -> tuple[Order, list[Trade]]:
        """
        Place a Spot Limit Buy or Spot Limit Sell order.

        Steps:
          1. Create the order and register it in _orders.
          2. Run the matching engine against the opposing side.
          3. If any quantity remains unfilled, rest it in the book.

        Returns (order, list_of_trades_generated).
        The order object reflects the final status after matching.
        """
        with self._rwlock.write_locked():
            order = Order(
                order_id        = str(uuid.uuid4()),
                side            = side,
                price           = price,
                quantity        = quantity,
                filled_quantity = Decimal("0"),
            )
            self._orders[order.order_id] = order
            trades = self._match(order)
            if order.remaining_quantity > 0:
                self._add_to_book(order)
            return order, trades

    def cancel_order(self, order_id: str) -> tuple[bool, str, Optional[Order]]:
        """
        Cancel an open or partially-filled order.

        The order is removed from its price-level deque immediately so that
        get_snapshot and matching are never exposed to stale entries.
        Returns (success, message, order).
        """
        with self._rwlock.write_locked():
            order = self._orders.get(order_id)
            if not order:
                return False, "Order not found", None
            if not order.is_active():
                return False, f"Order already {order.status.value}", order
            self._remove_from_book(order)
            order.status = OrderStatus.CANCELLED
            return True, "Order cancelled", order

    # ─────────────────── Read operations ──────────────────────────

    def get_snapshot(self, depth: int = 0) -> dict:
        """
        Return a point-in-time snapshot of the book.

        depth=0 returns all levels; depth=N returns the top N levels per side.
        Runs concurrently with other reads.
        """
        with self._rwlock.read_locked():
            # Bids: highest price first (reverse the ascending SortedDict)
            bid_prices = list(reversed(self._bids.keys()))
            ask_prices = list(self._asks.keys())

            def build_levels(book: SortedDict, prices: list) -> list[dict]:
                levels = []
                for p in prices:
                    q = book.get(p)
                    if q:
                        levels.append({
                            "price":       str(p),
                            "quantity":    str(sum(o.remaining_quantity for o in q)),
                            "order_count": len(q),
                        })
                return levels

            # Build the full filtered list first, then apply depth.
            # Applying depth before building would count phantom empty levels.
            bids = build_levels(self._bids, bid_prices)
            asks = build_levels(self._asks, ask_prices)

            if depth > 0:
                bids = bids[:depth]
                asks = asks[:depth]

            # Derive best prices from the filtered results, not the raw key list.
            # Using raw keys would give a wrong price if the top level is empty.
            best_bid  = bids[0]["price"] if bids else ""
            best_ask  = asks[0]["price"] if asks else ""
            spread    = ""
            mid_price = ""
            if best_bid and best_ask:
                spread    = str(Decimal(best_ask) - Decimal(best_bid))
                mid_price = str((Decimal(best_bid) + Decimal(best_ask)) / 2)

            return {
                "bids":      bids,
                "asks":      asks,
                "best_bid":  best_bid,
                "best_ask":  best_ask,
                "spread":    spread,
                "mid_price": mid_price,
            }

    def quote_order(self, side: OrderSide, quantity: Decimal) -> dict:
        """
        Simulate filling `quantity` against the live book without mutating state.

        BUY  taker: walks asks ascending (cheapest first).
        SELL taker: walks bids descending (highest first).

        Returns executable_price (VWAP), fills, slippage_from_mid,
        recommendation, fully_fillable, and fillable_quantity.
        Read-only — holds only the read lock.
        """
        with self._rwlock.read_locked():
            bid_keys = list(reversed(self._bids.keys()))
            ask_keys = list(self._asks.keys())

            best_bid = bid_keys[0] if bid_keys else None
            best_ask = ask_keys[0] if ask_keys else None
            mid = (best_bid + best_ask) / 2 if (best_bid is not None and best_ask is not None) else None

            walk_keys = ask_keys if side == OrderSide.BUY else list(reversed(self._bids.keys()))
            book      = self._asks if side == OrderSide.BUY else self._bids

            fills:      list[dict] = []
            remaining  = quantity
            total_cost = Decimal("0")

            for price in walk_keys:
                if remaining <= 0:
                    break
                level_qty = sum(
                    o.remaining_quantity for o in book.get(price, []) if o.is_active()
                )
                if level_qty <= 0:
                    continue
                fill_qty = min(remaining, level_qty)
                fills.append({"price": str(price), "qty": str(fill_qty)})
                total_cost += price * fill_qty
                remaining  -= fill_qty

            fillable_qty   = quantity - remaining
            fully_fillable = remaining <= 0

            if fillable_qty > 0:
                vwap     = (total_cost / fillable_qty).quantize(Decimal("0.01"))
                vwap_str = str(vwap)
            else:
                vwap_str = ""

            slippage_str = ""
            if mid is not None and vwap_str:
                vwap_dec = Decimal(vwap_str)
                raw      = (vwap_dec - mid) / mid * 100 if side == OrderSide.BUY else (mid - vwap_dec) / mid * 100
                slippage_str = f"{abs(raw).quantize(Decimal('0.0001'))}%"

            if not fills:
                side_label    = "ask" if side == OrderSide.BUY else "bid"
                recommendation = f"No liquidity on the {side_label} side. Cannot fill this order."
            elif not fully_fillable:
                worst_price    = fills[-1]["price"]
                recommendation = (
                    f"Only {fillable_qty} ETH fillable (requested {quantity}). "
                    f"place_order({side.value}, {worst_price}, {fillable_qty}) for a partial fill."
                )
            else:
                worst_price    = fills[-1]["price"]
                recommendation = (
                    f"place_order({side.value}, {worst_price}, {quantity}) "
                    f"to guarantee a full fill at VWAP {vwap_str}"
                )

            return {
                "executable_price":  vwap_str,
                "fills":             fills,
                "slippage_from_mid": slippage_str,
                "recommendation":    recommendation,
                "fully_fillable":    fully_fillable,
                "fillable_quantity": str(fillable_qty),
            }

    def get_order(self, order_id: str) -> Optional[Order]:
        """O(1) lookup. Concurrent with other reads."""
        with self._rwlock.read_locked():
            return self._orders.get(order_id)

    def get_trades(
        self,
        limit:    int = 50,
        order_id: Optional[str] = None,
    ) -> list[Trade]:
        """Return recent trades, newest first. Concurrent with other reads."""
        with self._rwlock.read_locked():
            if order_id:
                return [
                    t for t in reversed(self._trades)
                    if t.buy_order_id == order_id or t.sell_order_id == order_id
                ][:limit]
            return list(reversed(self._trades))[:limit]

    # ─────────────────── Matching internals ───────────────────────
    # All private methods below are called while the write lock is held.

    def _match(self, incoming: Order) -> list[Trade]:
        """
        Price-time priority matching using SortedDict.irange_key().

        BUY  taker: iterate asks ascending (cheapest first), stop when ask > limit
        SELL taker: iterate bids descending (highest first), stop when bid < limit

        irange_key() returns only keys within the crossable range — O(k) where
        k is the number of price levels actually consumed.
        """
        trades: list[Trade] = []

        if incoming.side == OrderSide.BUY:
            # Match against asks with price ≤ incoming limit price
            for price in list(self._asks.irange_key(None, incoming.price)):
                if incoming.remaining_quantity <= 0:
                    break
                trades.extend(self._fill_at_level(incoming, self._asks[price], price))
            self._prune(self._asks)
        else:
            # Match against bids with price ≥ incoming limit price (highest first)
            for price in reversed(list(self._bids.irange_key(incoming.price, None))):
                if incoming.remaining_quantity <= 0:
                    break
                trades.extend(self._fill_at_level(incoming, self._bids[price], price))
            self._prune(self._bids)

        return trades

    def _fill_at_level(
        self,
        incoming: Order,
        queue:    deque[Order],
        price:    Decimal,
    ) -> list[Trade]:
        """
        Consume resting orders at a single price level (FIFO).
        Trade price is the resting order's price (maker price).
        """
        trades: list[Trade] = []
        while queue and incoming.remaining_quantity > 0:
            resting = queue[0]

            # Guard: skip any order that was partially filled then cancelled.
            # Under normal operation this should not occur because cancel_order
            # removes orders from the deque immediately, but the check is kept
            # as a safety net.
            if not resting.is_active():
                queue.popleft()
                continue

            trade_qty = min(incoming.remaining_quantity, resting.remaining_quantity)
            trade = Trade(
                trade_id      = str(uuid.uuid4()),
                buy_order_id  = incoming.order_id if incoming.side == OrderSide.BUY else resting.order_id,
                sell_order_id = resting.order_id  if incoming.side == OrderSide.BUY else incoming.order_id,
                price         = price,
                quantity      = trade_qty,
            )
            incoming.filled_quantity += trade_qty
            resting.filled_quantity  += trade_qty

            # Update statuses after adjusting filled quantities
            resting.status  = OrderStatus.FILLED          if resting.remaining_quantity  <= 0 else OrderStatus.PARTIALLY_FILLED
            incoming.status = OrderStatus.FILLED          if incoming.remaining_quantity <= 0 else OrderStatus.PARTIALLY_FILLED

            if resting.remaining_quantity <= 0:
                queue.popleft()

            self._trades.append(trade)
            trades.append(trade)

        return trades

    def _add_to_book(self, order: Order) -> None:
        """Add a resting order to its price level (creating the level if needed)."""
        book = self._bids if order.side == OrderSide.BUY else self._asks
        if order.price not in book:
            book[order.price] = deque()
        book[order.price].append(order)

    def _remove_from_book(self, order: Order) -> None:
        """
        Immediately remove a specific order from its price-level deque.

        O(m) where m = orders at that price level (typically very small).
        Immediate removal is used instead of a lazy-cancelled set so that
        get_snapshot and the matching loop need no filtered-set bookkeeping.
        If the level becomes empty it is deleted from the SortedDict.
        """
        book = self._bids if order.side == OrderSide.BUY else self._asks
        if order.price in book:
            book[order.price] = deque(
                o for o in book[order.price] if o.order_id != order.order_id
            )
            if not book[order.price]:
                del book[order.price]

    @staticmethod
    def _prune(book: SortedDict) -> None:
        """Delete price levels whose deque was emptied during matching."""
        for price in [p for p, q in book.items() if not q]:
            del book[price]
