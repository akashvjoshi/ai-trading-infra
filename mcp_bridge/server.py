"""
MCP Bridge — wraps the gRPC CLOB engine for LLM consumption.

Design rationale
────────────────

Resources vs Tools
  Resources are PASSIVE CONTEXT — the LLM (or host) reads them at the start
  of a session to understand current market state without spending a tool call.
  Tools are ACTIVE ACTIONS — the LLM calls them to get fresh data right before
  acting, or to mutate state.

  Overlap is intentional:
    • market://eth-usdc/orderbook  (resource) — background context at session start
    • get_orderbook                (tool)     — fresh snapshot immediately before placing
  The resource gives the LLM a baseline; the tool gives it certainty at act time.

Resource granularity
  Three resources at different cost levels let the LLM pick the cheapest one
  that answers its question:
    /stats     — 4 numbers (best bid, ask, spread, mid)  ← use for price check
    /orderbook — top-5 depth per side                    ← use for liquidity check
    /trades    — last 20 fills                           ← use for price history

Tool descriptions
  Every description answers three questions the LLM needs:
    1. WHAT does this tool do?
    2. WHEN should I call it? (sequencing)
    3. WHAT will I get back?

Context window discipline
  • Tool results use compact JSON (no indentation).  Resources use indented JSON
    because they may be rendered in a UI.
  • Depth is capped server-side at MAX_DEPTH to prevent the LLM from
    accidentally flooding its own context.
  • Trade IDs are truncated to 8 chars in results (full UUID only where needed).
  • place_order returns quantity + filled_quantity; the LLM can derive remaining
    itself — no redundant fields.
"""
import asyncio
import json
import logging
import os
from decimal import Decimal

import grpc
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from engine.generated import clob_pb2, clob_pb2_grpc

logger = logging.getLogger(__name__)

GRPC_ADDR = os.getenv("ENGINE_ADDR", "localhost:50051")
MAX_DEPTH  = 20   # server-side cap; prevents the LLM flooding its own context


def _stub() -> clob_pb2_grpc.ClobServiceStub:
    return clob_pb2_grpc.ClobServiceStub(grpc.insecure_channel(GRPC_ADDR))


def _status(code: int) -> str:
    return {0: "OPEN", 1: "FILLED", 2: "PARTIALLY_FILLED", 3: "CANCELLED"}.get(code, "UNKNOWN")


def _compact(data) -> str:
    """Compact JSON for tool results — minimises token count."""
    return json.dumps(data, separators=(",", ":"))


def _pretty(data) -> str:
    """Indented JSON for resource content — may be user-visible in MCP clients."""
    return json.dumps(data, indent=2)


# ─────────────────────────────────────────────────────────────────
app = Server("clob-mcp-bridge")
# ─────────────────────────────────────────────────────────────────


# ══════════════════════════ Resources ════════════════════════════
#
# Resources are injected as passive context — the LLM reads them to
# understand the current market without spending a tool-call turn.
# ─────────────────────────────────────────────────────────────────

@app.list_resources()
async def list_resources() -> list[types.Resource]:
    return [
        types.Resource(
            uri         = "market://eth-usdc/stats",
            name        = "Market Stats (minimal)",
            description = (
                "Four numbers: best_bid, best_ask, spread, mid_price. "
                "Read this first to orient yourself before deciding whether to trade. "
                "Cheapest context — prefer this over /orderbook when you only need the price."
            ),
            mimeType    = "application/json",
        ),
        types.Resource(
            uri         = "market://eth-usdc/orderbook",
            name        = "Order Book — top 5 levels",
            description = (
                "Top-5 bid and ask price levels with quantity and order count. "
                "Read this when you need to assess available liquidity before placing "
                "a larger order. More expensive than /stats — only fetch when needed."
            ),
            mimeType    = "application/json",
        ),
        types.Resource(
            uri         = "market://eth-usdc/trades",
            name        = "Recent Trades — last 20",
            description = (
                "The 20 most recently executed trades (price, quantity). "
                "Useful for understanding recent transaction prices and market activity."
            ),
            mimeType    = "application/json",
        ),
    ]


@app.read_resource()
async def read_resource(uri: str) -> types.ReadResourceResult:
    stub = _stub()

    if uri == "market://eth-usdc/stats":
        resp = stub.GetOrderBook(clob_pb2.GetOrderBookRequest(depth=1))
        data = {
            "best_bid":  resp.best_bid,
            "best_ask":  resp.best_ask,
            "spread":    resp.spread,
            "mid_price": resp.mid_price,
        }

    elif uri == "market://eth-usdc/orderbook":
        resp = stub.GetOrderBook(clob_pb2.GetOrderBookRequest(depth=5))
        data = {
            "bids": [
                {"price": lv.price, "qty": lv.quantity, "order_count": lv.order_count}
                for lv in resp.bids
            ],
            "asks": [
                {"price": lv.price, "qty": lv.quantity, "order_count": lv.order_count}
                for lv in resp.asks
            ],
            "best_bid":  resp.best_bid,
            "best_ask":  resp.best_ask,
            "spread":    resp.spread,
            "mid_price": resp.mid_price,
        }

    elif uri == "market://eth-usdc/trades":
        resp = stub.GetTrades(clob_pb2.GetTradesRequest(limit=20))
        data = [
            {"id": t.trade_id[:8], "price": t.price, "qty": t.quantity}
            for t in resp.trades
        ]

    else:
        raise ValueError(f"Unknown resource URI: {uri}")

    return types.ReadResourceResult(
        contents=[
            types.TextResourceContents(uri=uri, text=_pretty(data), mimeType="application/json")
        ]
    )


# ══════════════════════════ Tools ════════════════════════════════
#
# Tools are ACTIVE — the LLM calls them to get fresh state or mutate it.
#
# Descriptions answer three questions:
#   1. What does this do?
#   2. When should I call it?  (sequencing guidance)
#   3. What will I get back?
# ─────────────────────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name        = "get_orderbook",
            description = (
                "Fetch a fresh order book snapshot right now. "
                "Call this immediately before place_order to get current prices — "
                "the resource /orderbook may be stale if time has passed. "
                "Returns top-N bid and ask levels, best bid/ask, spread, and mid price."
            ),
            inputSchema = {
                "type": "object",
                "properties": {
                    "depth": {
                        "type":        "integer",
                        "default":     5,
                        "minimum":     1,
                        "maximum":     MAX_DEPTH,
                        "description": f"Price levels per side (1–{MAX_DEPTH}). Use 1 for a quick price check.",
                    },
                },
            },
        ),
        types.Tool(
            name        = "place_order",
            description = (
                "Place a Spot Limit Buy or Spot Limit Sell order on the ETH/USDC book. "
                "BEFORE calling this: call get_orderbook (or read market://eth-usdc/stats) "
                "to know the current price — never guess. "
                "A limit order fills at your price or better; any unfilled remainder rests "
                "in the book. "
                "Returns: order_id (save it for cancel_order), quantity, filled_quantity, "
                "status (OPEN / FILLED / PARTIALLY_FILLED), and a list of resulting trades."
            ),
            inputSchema = {
                "type": "object",
                "properties": {
                    "side": {
                        "type":        "string",
                        "enum":        ["BUY", "SELL"],
                        "description": "BUY to acquire ETH, SELL to dispose of ETH",
                    },
                    "price": {
                        "type":        "string",
                        "description": "Limit price in USDC as a decimal string, e.g. '3200.50'",
                    },
                    "quantity": {
                        "type":        "string",
                        "description": "ETH amount as a decimal string, e.g. '0.5'",
                    },
                },
                "required": ["side", "price", "quantity"],
            },
        ),
        types.Tool(
            name        = "cancel_order",
            description = (
                "Cancel an order that is OPEN or PARTIALLY_FILLED. "
                "Will fail if the order is already FILLED or CANCELLED — "
                "call get_order first to confirm the status before cancelling. "
                "Returns: success (bool) and a message."
            ),
            inputSchema = {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type":        "string",
                        "description": "The order_id returned by place_order",
                    },
                },
                "required": ["order_id"],
            },
        ),
        types.Tool(
            name        = "get_order",
            description = (
                "Look up the current status of a specific order. "
                "Call this after place_order to confirm it was accepted, "
                "or before cancel_order to verify the order is still open. "
                "Returns: order_id, side, price, quantity, filled_quantity, status."
            ),
            inputSchema = {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                },
                "required": ["order_id"],
            },
        ),
        types.Tool(
            name        = "get_trades",
            description = (
                "Retrieve recent trade history. "
                "Use without order_id for general market activity. "
                "Use with order_id to see fills for a specific order after placing it. "
                "Returns a list of {id, price, qty} records, newest first."
            ),
            inputSchema = {
                "type": "object",
                "properties": {
                    "limit": {
                        "type":        "integer",
                        "default":     20,
                        "minimum":     1,
                        "maximum":     100,
                        "description": "Number of trades to return (1–100)",
                    },
                    "order_id": {
                        "type":        "string",
                        "description": "Filter to fills involving this order (optional)",
                    },
                },
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    stub = _stub()
    try:
        result = _dispatch(stub, name, arguments)
    except grpc.RpcError as exc:
        result = {"error": exc.details()}
    except Exception as exc:
        result = {"error": str(exc)}
    # Compact JSON: tool results are consumed by the LLM, not rendered for humans
    return [types.TextContent(type="text", text=_compact(result))]


def _dispatch(stub, name: str, args: dict) -> dict:

    if name == "get_orderbook":
        depth = min(int(args.get("depth", 5)), MAX_DEPTH)
        resp  = stub.GetOrderBook(clob_pb2.GetOrderBookRequest(depth=depth))
        return {
            "bids": [
                {"price": lv.price, "qty": lv.quantity, "order_count": lv.order_count}
                for lv in resp.bids
            ],
            "asks": [
                {"price": lv.price, "qty": lv.quantity, "order_count": lv.order_count}
                for lv in resp.asks
            ],
            "best_bid":  resp.best_bid,
            "best_ask":  resp.best_ask,
            "spread":    resp.spread,
            "mid_price": resp.mid_price,
        }

    if name == "place_order":
        side = clob_pb2.BUY if args["side"].upper() == "BUY" else clob_pb2.SELL
        resp = stub.PlaceOrder(clob_pb2.PlaceOrderRequest(
            side=side, price=str(args["price"]), quantity=str(args["quantity"])
        ))
        o = resp.order
        return {
            "order_id":        resp.order_id,
            "status":          _status(o.status),
            "quantity":        o.quantity,
            "filled_quantity": o.filled_quantity,
            # No redundant 'remaining' field — LLM can compute: quantity - filled_quantity
            "trades": [
                {"id": t.trade_id[:8], "price": t.price, "qty": t.quantity}
                for t in resp.trades
            ],
        }

    if name == "cancel_order":
        resp = stub.CancelOrder(clob_pb2.CancelOrderRequest(order_id=args["order_id"]))
        return {"success": resp.success, "message": resp.message}

    if name == "get_order":
        resp = stub.GetOrder(clob_pb2.GetOrderRequest(order_id=args["order_id"]))
        if not resp.found:
            return {"error": "Order not found"}
        o = resp.order
        return {
            "order_id":        o.order_id,
            "side":            "BUY" if o.side == clob_pb2.BUY else "SELL",
            "price":           o.price,
            "quantity":        o.quantity,
            "filled_quantity": o.filled_quantity,
            "status":          _status(o.status),
        }

    if name == "get_trades":
        limit = min(int(args.get("limit", 20)), 100)
        resp  = stub.GetTrades(clob_pb2.GetTradesRequest(
            limit    = limit,
            order_id = args.get("order_id", ""),
        ))
        return [
            {"id": t.trade_id[:8], "price": t.price, "qty": t.quantity}
            for t in resp.trades
        ]

    return {"error": f"Unknown tool: {name}"}


# ══════════════════════════ Entry Point ══════════════════════════

async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
