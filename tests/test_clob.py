"""Unit tests for the core CLOB engine — no gRPC, no LLM."""
import threading
from decimal import Decimal

import pytest

from engine.clob import OrderBook, OrderSide, OrderStatus


@pytest.fixture
def book() -> OrderBook:
    return OrderBook()


# ───────────────────────── Basic placement ────────────────────────

def test_buy_rests_when_no_match(book):
    order, trades = book.place_order(OrderSide.BUY, Decimal("3200"), Decimal("1"))
    assert order.status == OrderStatus.OPEN
    assert trades == []
    snap = book.get_snapshot()
    assert snap["bids"][0]["price"] == "3200"
    assert snap["asks"] == []


def test_sell_rests_when_no_match(book):
    order, trades = book.place_order(OrderSide.SELL, Decimal("3300"), Decimal("1"))
    assert order.status == OrderStatus.OPEN
    assert trades == []
    snap = book.get_snapshot()
    assert snap["asks"][0]["price"] == "3300"
    assert snap["bids"] == []


# ───────────────────────── Matching ───────────────────────────────

def test_exact_fill(book):
    buy, _  = book.place_order(OrderSide.BUY,  Decimal("3200"), Decimal("1"))
    sell, trades = book.place_order(OrderSide.SELL, Decimal("3200"), Decimal("1"))

    assert len(trades) == 1
    assert trades[0].price    == Decimal("3200")
    assert trades[0].quantity == Decimal("1")
    assert buy.status  == OrderStatus.FILLED
    assert sell.status == OrderStatus.FILLED
    snap = book.get_snapshot()
    assert snap["bids"] == []
    assert snap["asks"] == []


def test_buy_partially_fills(book):
    buy, _      = book.place_order(OrderSide.BUY,  Decimal("3200"), Decimal("2"))
    sell, trades = book.place_order(OrderSide.SELL, Decimal("3200"), Decimal("1"))

    assert len(trades) == 1
    assert trades[0].quantity == Decimal("1")
    assert buy.status  == OrderStatus.PARTIALLY_FILLED
    assert sell.status == OrderStatus.FILLED
    # 1 ETH should remain on the bid side
    snap = book.get_snapshot()
    assert snap["bids"][0]["quantity"] == "1"


def test_sell_partially_fills(book):
    sell, _     = book.place_order(OrderSide.SELL, Decimal("3200"), Decimal("3"))
    buy, trades = book.place_order(OrderSide.BUY,  Decimal("3200"), Decimal("1"))

    assert len(trades) == 1
    assert sell.status == OrderStatus.PARTIALLY_FILLED
    assert buy.status  == OrderStatus.FILLED


def test_taker_fills_across_multiple_levels(book):
    book.place_order(OrderSide.SELL, Decimal("3200"), Decimal("1"))
    book.place_order(OrderSide.SELL, Decimal("3210"), Decimal("1"))
    buy, trades = book.place_order(OrderSide.BUY, Decimal("3210"), Decimal("2"))

    assert len(trades) == 2
    assert buy.status == OrderStatus.FILLED


def test_price_time_priority_within_level(book):
    b1, _ = book.place_order(OrderSide.BUY, Decimal("3200"), Decimal("1"))
    b2, _ = book.place_order(OrderSide.BUY, Decimal("3200"), Decimal("1"))

    # One sell fills only b1 (arrived first)
    sell, trades = book.place_order(OrderSide.SELL, Decimal("3200"), Decimal("1"))
    assert b1.status == OrderStatus.FILLED
    assert b2.status == OrderStatus.OPEN


def test_sell_does_not_match_below_limit(book):
    book.place_order(OrderSide.BUY, Decimal("3100"), Decimal("1"))
    sell, trades = book.place_order(OrderSide.SELL, Decimal("3200"), Decimal("1"))
    assert trades == []
    assert sell.status == OrderStatus.OPEN


# ───────────────────────── Cancellation ───────────────────────────

def test_cancel_open_order(book):
    order, _ = book.place_order(OrderSide.BUY, Decimal("3200"), Decimal("1"))
    success, msg, cancelled = book.cancel_order(order.order_id)
    assert success is True
    assert cancelled.status == OrderStatus.CANCELLED
    assert book.get_snapshot()["bids"] == []


def test_cancel_removes_from_book(book):
    o1, _ = book.place_order(OrderSide.BUY, Decimal("3200"), Decimal("1"))
    o2, _ = book.place_order(OrderSide.BUY, Decimal("3200"), Decimal("2"))
    book.cancel_order(o1.order_id)

    snap = book.get_snapshot()
    assert snap["bids"][0]["quantity"] == "2"


def test_cancel_nonexistent_order(book):
    success, msg, order = book.cancel_order("no-such-id")
    assert success is False
    assert order is None


def test_cancel_already_filled(book):
    buy, _  = book.place_order(OrderSide.BUY,  Decimal("3200"), Decimal("1"))
    book.place_order(OrderSide.SELL, Decimal("3200"), Decimal("1"))
    success, msg, _ = book.cancel_order(buy.order_id)
    assert success is False


# ───────────────────────── Snapshot ───────────────────────────────

def test_snapshot_depth_limit(book):
    for p in [3200, 3190, 3180, 3170, 3160]:
        book.place_order(OrderSide.BUY, Decimal(str(p)), Decimal("1"))
    snap = book.get_snapshot(depth=3)
    assert len(snap["bids"]) == 3
    assert snap["bids"][0]["price"] == "3200"  # best bid first


def test_spread_and_mid_price(book):
    book.place_order(OrderSide.BUY,  Decimal("3200"), Decimal("1"))
    book.place_order(OrderSide.SELL, Decimal("3220"), Decimal("1"))
    snap = book.get_snapshot()
    assert snap["best_bid"]  == "3200"
    assert snap["best_ask"]  == "3220"
    assert snap["spread"]    == "20"
    assert snap["mid_price"] == "3210"


# ───────────────────────── Trade history ──────────────────────────

def test_get_trades_returns_latest_first(book):
    for price in [3200, 3210, 3220]:
        book.place_order(OrderSide.BUY,  Decimal(str(price)), Decimal("1"))
        book.place_order(OrderSide.SELL, Decimal(str(price)), Decimal("1"))
    trades = book.get_trades(limit=10)
    assert len(trades) == 3
    # most recent first
    assert Decimal(trades[0].price) >= Decimal(trades[-1].price)


def test_get_trades_filtered_by_order(book):
    buy1, _ = book.place_order(OrderSide.BUY, Decimal("3200"), Decimal("1"))
    buy2, _ = book.place_order(OrderSide.BUY, Decimal("3200"), Decimal("1"))
    book.place_order(OrderSide.SELL, Decimal("3200"), Decimal("2"))

    trades1 = book.get_trades(order_id=buy1.order_id)
    trades2 = book.get_trades(order_id=buy2.order_id)
    assert len(trades1) == 1
    assert len(trades2) == 1


# ───────────────────────── Thread safety ──────────────────────────

def test_concurrent_placements_do_not_crash(book):
    errors: list[Exception] = []

    def place_many():
        try:
            for _ in range(200):
                book.place_order(OrderSide.BUY, Decimal("3200"), Decimal("0.01"))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=place_many) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [], f"Thread errors: {errors}"


def test_concurrent_place_and_cancel(book):
    orders = []
    for _ in range(50):
        o, _ = book.place_order(OrderSide.BUY, Decimal("3200"), Decimal("0.1"))
        orders.append(o)

    errors: list[Exception] = []

    def cancel_all():
        for o in orders:
            try:
                book.cancel_order(o.order_id)
            except Exception as exc:
                errors.append(exc)

    threads = [threading.Thread(target=cancel_all) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


# ─────────── ReadWriteLock: concurrent reads don't block each other ──

def test_concurrent_reads_do_not_deadlock(book):
    """Multiple threads reading the snapshot simultaneously must not block."""
    for p in [3190, 3200, 3210]:
        book.place_order(OrderSide.BUY, Decimal(str(p)), Decimal("1"))

    results = []
    errors: list[Exception] = []

    def read_snap():
        try:
            snap = book.get_snapshot()
            results.append(len(snap["bids"]))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=read_snap) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert all(r == 3 for r in results)


def test_reads_and_writes_concurrent(book):
    """Readers and writers running simultaneously must not corrupt state."""
    errors: list[Exception] = []

    def writer():
        for i in range(100):
            try:
                book.place_order(OrderSide.BUY, Decimal("3200"), Decimal("0.01"))
            except Exception as exc:
                errors.append(exc)

    def reader():
        for _ in range(200):
            try:
                book.get_snapshot(depth=5)
            except Exception as exc:
                errors.append(exc)

    threads = (
        [threading.Thread(target=writer) for _ in range(4)]
        + [threading.Thread(target=reader) for _ in range(4)]
    )
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [], f"Errors: {errors}"


# ─────────────── Cancellation correctness ────────────────────────
#
# These tests pin the three bugs that existed in a previous implementation
# which used a lazy _cancelled set instead of immediate deque removal:
#
#   Bug 1: best_bid/best_ask derived from raw SortedDict keys, not from the
#          filtered result list → reported a phantom price when the top level
#          held only cancelled orders.
#
#   Bug 2: depth applied before filtering → a depth-N request could return
#          fewer than N levels because cancelled-only slots were counted first.
#
#   Bug 3: _prune only removed empty deques; levels full of cancelled orders
#          were never cleaned up, leaving phantom keys in the SortedDict.

def test_cancel_removes_order_from_snapshot(book):
    """Cancelled order must be invisible to get_snapshot immediately."""
    o, _ = book.place_order(OrderSide.BUY, Decimal("3200"), Decimal("1"))
    book.cancel_order(o.order_id)
    snap = book.get_snapshot()
    assert snap["bids"] == []


def test_cancel_removes_price_level_when_last_order(book):
    """When the last order at a price level is cancelled, the level itself
    must be removed — no phantom price keys in the book (Bug 3)."""
    o, _ = book.place_order(OrderSide.BUY, Decimal("3200"), Decimal("1"))
    book.cancel_order(o.order_id)
    # Direct inspection: _bids must be empty, not contain a key with empty deque
    assert len(book._bids) == 0


def test_best_bid_reflects_next_level_after_top_cancelled(book):
    """best_bid must come from the highest *active* level, not the highest
    key in the SortedDict (Bug 1)."""
    top, _  = book.place_order(OrderSide.BUY, Decimal("3200"), Decimal("1"))  # top level
    book.place_order(OrderSide.BUY, Decimal("3190"), Decimal("1"))             # second level
    book.cancel_order(top.order_id)  # cancel the top level entirely

    snap = book.get_snapshot()
    # best_bid must skip the now-empty 3200 level and report 3190
    assert snap["best_bid"] == "3190"
    assert len(snap["bids"]) == 1


def test_depth_counts_only_active_levels(book):
    """With depth=2, the result must contain 2 *active* price levels even if
    higher phantom levels exist in the raw SortedDict (Bug 2)."""
    prices = [3200, 3190, 3180]
    orders = []
    for p in prices:
        o, _ = book.place_order(OrderSide.BUY, Decimal(str(p)), Decimal("1"))
        orders.append(o)

    # Cancel the top level (3200) — it should not consume a depth slot
    book.cancel_order(orders[0].order_id)

    snap = book.get_snapshot(depth=2)
    assert len(snap["bids"]) == 2
    assert snap["bids"][0]["price"] == "3190"
    assert snap["bids"][1]["price"] == "3180"


def test_cancelled_order_not_matched_by_subsequent_taker(book):
    """A cancelled bid must not be matched by a later sell order."""
    buy, _ = book.place_order(OrderSide.BUY, Decimal("3200"), Decimal("1"))
    book.cancel_order(buy.order_id)
    sell, trades = book.place_order(OrderSide.SELL, Decimal("3200"), Decimal("1"))
    assert trades == []
    assert sell.status == OrderStatus.OPEN


def test_cancel_partially_filled_order(book):
    """Cancelling a partially-filled order should succeed and the remaining
    quantity should vanish from the book."""
    buy, _  = book.place_order(OrderSide.BUY,  Decimal("3200"), Decimal("2"))
    book.place_order(OrderSide.SELL, Decimal("3200"), Decimal("1"))   # partial fill
    assert buy.status == OrderStatus.PARTIALLY_FILLED

    success, _, cancelled = book.cancel_order(buy.order_id)
    assert success is True
    assert cancelled.status == OrderStatus.CANCELLED
    assert book.get_snapshot()["bids"] == []


def test_cancel_returns_false_for_filled_order(book):
    """Cancelling an already-filled order must fail gracefully."""
    buy, _  = book.place_order(OrderSide.BUY,  Decimal("3200"), Decimal("1"))
    book.place_order(OrderSide.SELL, Decimal("3200"), Decimal("1"))
    assert buy.status == OrderStatus.FILLED

    success, msg, _ = book.cancel_order(buy.order_id)
    assert success is False
    assert "FILLED" in msg


def test_cancel_twice_fails_second_time(book):
    """Double-cancelling the same order must fail on the second attempt."""
    o, _ = book.place_order(OrderSide.BUY, Decimal("3200"), Decimal("1"))
    book.cancel_order(o.order_id)
    success, _, _ = book.cancel_order(o.order_id)
    assert success is False


# ─────────────── Bounded trade log ────────────────────────────────

def test_trades_bounded_by_max_trades():
    small_book = OrderBook(max_trades=5)
    for i in range(10):
        price = Decimal("3200")
        small_book.place_order(OrderSide.BUY,  price, Decimal("1"))
        small_book.place_order(OrderSide.SELL, price, Decimal("1"))
    trades = small_book.get_trades(limit=100)
    # deque(maxlen=5) keeps only the last 5
    assert len(trades) == 5
