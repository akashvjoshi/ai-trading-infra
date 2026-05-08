"""
gRPC server — the ONLY public interface to the OrderBook.

Callers must use the ClobService stub (engine.generated.clob_pb2_grpc).
Direct instantiation of OrderBook outside this module is intentionally
not part of the public API.

Concurrency
───────────
gRPC manages a ThreadPoolExecutor whose size defaults to
  min(32, os.cpu_count() + 4)
and can be overridden via MAX_GRPC_WORKERS env var.

Each RPC call gets its own thread from the pool.  The OrderBook's
ReadWriteLock then coordinates inside: concurrent reads proceed in
parallel; writes (PlaceOrder, CancelOrder) serialise only against
each other and active reads.
"""
import logging
import os
import time
from concurrent import futures
from decimal import Decimal

import grpc

from engine.clob import OrderBook, OrderSide, OrderStatus
from engine.generated import clob_pb2, clob_pb2_grpc

logger = logging.getLogger(__name__)

# ── worker sizing ──────────────────────────────────────────────────
_DEFAULT_WORKERS = min(32, (os.cpu_count() or 1) + 4)
_MAX_WORKERS     = int(os.getenv("MAX_GRPC_WORKERS", _DEFAULT_WORKERS))

# ── protobuf converters ───────────────────────────────────────────

_TO_PB_STATUS = {
    OrderStatus.OPEN:             clob_pb2.OPEN,
    OrderStatus.FILLED:           clob_pb2.FILLED,
    OrderStatus.PARTIALLY_FILLED: clob_pb2.PARTIALLY_FILLED,
    OrderStatus.CANCELLED:        clob_pb2.CANCELLED,
}


def _order_to_pb(order) -> clob_pb2.Order:
    return clob_pb2.Order(
        order_id        = order.order_id,
        side            = clob_pb2.BUY if order.side == OrderSide.BUY else clob_pb2.SELL,
        price           = str(order.price),
        quantity        = str(order.quantity),
        filled_quantity = str(order.filled_quantity),
        status          = _TO_PB_STATUS[order.status],
        created_at      = order.created_at,
    )


def _trade_to_pb(trade) -> clob_pb2.Trade:
    return clob_pb2.Trade(
        trade_id      = trade.trade_id,
        buy_order_id  = trade.buy_order_id,
        sell_order_id = trade.sell_order_id,
        price         = str(trade.price),
        quantity      = str(trade.quantity),
        timestamp     = trade.timestamp,
    )


# ── servicer ──────────────────────────────────────────────────────

class ClobServicer(clob_pb2_grpc.ClobServiceServicer):
    """
    One instance is shared across all gRPC worker threads.
    Thread-safety is enforced inside OrderBook via its ReadWriteLock.
    """

    def __init__(self) -> None:
        self._book = OrderBook()

    def PlaceOrder(self, request, context):
        try:
            price    = Decimal(request.price)
            quantity = Decimal(request.quantity)
        except Exception as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"Invalid price or quantity: {exc}")
            return clob_pb2.PlaceOrderResponse()

        if price <= 0 or quantity <= 0:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Price and quantity must be positive")
            return clob_pb2.PlaceOrderResponse()

        side = OrderSide.BUY if request.side == clob_pb2.BUY else OrderSide.SELL
        order, trades = self._book.place_order(side, price, quantity)
        logger.info(
            "PlaceOrder side=%s price=%s qty=%s → %s (%d trades)",
            side.value, price, quantity, order.status.value, len(trades),
        )
        return clob_pb2.PlaceOrderResponse(
            order_id = order.order_id,
            order    = _order_to_pb(order),
            trades   = [_trade_to_pb(t) for t in trades],
        )

    def CancelOrder(self, request, context):
        if not request.order_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("order_id is required")
            return clob_pb2.CancelOrderResponse()

        success, message, order = self._book.cancel_order(request.order_id)
        logger.info("CancelOrder %s → success=%s", request.order_id, success)
        resp = clob_pb2.CancelOrderResponse(success=success, message=message)
        if order:
            resp.order.CopyFrom(_order_to_pb(order))
        return resp

    def GetOrderBook(self, request, context):
        snap = self._book.get_snapshot(depth=request.depth)
        return clob_pb2.GetOrderBookResponse(
            bids      = [clob_pb2.PriceLevel(**lv) for lv in snap["bids"]],
            asks      = [clob_pb2.PriceLevel(**lv) for lv in snap["asks"]],
            best_bid  = snap["best_bid"],
            best_ask  = snap["best_ask"],
            spread    = snap["spread"],
            mid_price = snap["mid_price"],
            timestamp = int(time.time() * 1e9),
        )

    def GetOrder(self, request, context):
        if not request.order_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("order_id is required")
            return clob_pb2.GetOrderResponse()

        order = self._book.get_order(request.order_id)
        if not order:
            return clob_pb2.GetOrderResponse(found=False)
        return clob_pb2.GetOrderResponse(order=_order_to_pb(order), found=True)

    def GetTrades(self, request, context):
        trades = self._book.get_trades(
            limit    = request.limit or 50,
            order_id = request.order_id or None,
        )
        return clob_pb2.GetTradesResponse(trades=[_trade_to_pb(t) for t in trades])


# ── public entry point ────────────────────────────────────────────

def serve(host: str = "localhost", port: int = 50051) -> grpc.Server:
    """
    Start the ClobService gRPC server.

    Worker count: MAX_GRPC_WORKERS env var (default: min(32, cpu_count+4)).
    Returns the running grpc.Server; caller is responsible for .stop().
    """
    executor = futures.ThreadPoolExecutor(max_workers=_MAX_WORKERS)
    server   = grpc.server(executor)
    clob_pb2_grpc.add_ClobServiceServicer_to_server(ClobServicer(), server)
    addr = f"{host}:{port}"
    server.add_insecure_port(addr)
    server.start()
    logger.info(
        "CLOB gRPC engine listening on %s  (workers=%d)", addr, _MAX_WORKERS
    )
    return server


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    srv = serve()
    try:
        while True:
            time.sleep(86_400)
    except KeyboardInterrupt:
        srv.stop(grace=5)
        logger.info("Engine stopped.")
