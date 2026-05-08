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

Resource granularity — four resources, increasing cost/detail:
  /stats    — 4 numbers (best bid, ask, spread, mid)      ← price check
  /orderbook— top-5 depth per side                        ← liquidity check
  /trades   — last 20 fills                               ← price history
  /context  — synthesised prose briefing with microstructure signals [NEW]
              Prose format means zero parsing cost for the LLM.

Tool descriptions
  Every description answers: WHAT, WHEN, and WHAT BACK.

New tools
  quote — simulate a fill before committing.  Walks the book server-side and
          returns VWAP, per-level fill breakdown, slippage, and a ready-to-use
          place_order recommendation.  Eliminates the manual book-walk reasoning
          step the LLM would otherwise perform (and often get wrong).

Structured errors (Idea 3)
  Every failure returns {"error": CODE, "detail": "...", "suggestion": "..."}
  so the LLM can pattern-match on the code and act on the suggestion instead
  of retrying blindly or giving up.  Error codes are defined in _ERRORS below.

Context window discipline
  • Tool results: compact JSON (separators=(",",":")) — ~30% fewer tokens.
  • Resource content: indented JSON or plain text (may be rendered in a UI).
  • Depth capped server-side at MAX_DEPTH.
  • Trade IDs truncated to 8 chars.
  • place_order returns quantity + filled_quantity; no redundant derived fields.
"""
import asyncio
import json
import logging
import os
import time
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

import grpc
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from engine.generated import clob_pb2, clob_pb2_grpc

logger    = logging.getLogger(__name__)
GRPC_ADDR = os.getenv("ENGINE_ADDR", "localhost:50051")
MAX_DEPTH = 20      # server-side cap on orderbook depth
MAX_LIMIT = 100     # server-side cap on trade history


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _stub() -> clob_pb2_grpc.ClobServiceStub:
    return clob_pb2_grpc.ClobServiceStub(grpc.insecure_channel(GRPC_ADDR))


def _status(code: int) -> str:
    return {0: "OPEN", 1: "FILLED", 2: "PARTIALLY_FILLED", 3: "CANCELLED"}.get(code, "UNKNOWN")


def _compact(data) -> str:
    """Compact JSON for tool results — minimises LLM token count."""
    return json.dumps(data, separators=(",", ":"))


def _pretty(data) -> str:
    """Indented JSON for resource content — may be rendered in a UI."""
    return json.dumps(data, indent=2)


# ── Structured errors ─────────────────────────────────────────────
#
# Every error returned by _dispatch is a dict with at minimum:
#   {"error": CODE, "detail": "human-readable explanation"}
# and optionally:
#   {"suggestion": "concrete next action the LLM should take"}
#
# Error codes the LLM can pattern-match on:
_ERRORS = {
    "NO_LIQUIDITY":          "No resting orders on the opposing side.",
    "PARTIAL_LIQUIDITY":     "Insufficient depth to fill the full quantity.",
    "ORDER_NOT_FOUND":       "No order exists with that order_id.",
    "ORDER_NOT_CANCELLABLE": "Order is already FILLED or CANCELLED.",
    "PRICE_DEVIATION":       "Order price is far from the current market mid.",
    "ENGINE_ERROR":          "The gRPC engine returned an error.",
    "INVALID_INPUT":         "One or more arguments are invalid.",
    "UNKNOWN_TOOL":          "No tool with that name is registered.",
}


def _error(code: str, detail: str, suggestion: str = "") -> dict:
    e = {"error": code, "detail": detail}
    if suggestion:
        e["suggestion"] = suggestion
    return e


# ── Quote book-walk logic ─────────────────────────────────────────

def _walk_book(levels, quantity: Decimal, side: str) -> dict:
    """
    Simulate walking price levels to fill `quantity` without placing an order.

    For BUY  : levels must be asks in ascending  order (cheapest first).
    For SELL : levels must be bids in descending order (highest first).

    Returns a dict with keys:
      fills          — list of {price, qty} per level consumed
      filled_qty     — total quantity that can be filled
      remaining_qty  — quantity that could not be filled (no depth)
      vwap           — volume-weighted average fill price (or "" if none)
      total_notional — total USDC cost/proceeds
    """
    if side == "SELL":
        levels = list(reversed(levels))   # highest bid first

    remaining      = quantity
    fills          = []
    total_notional = Decimal("0")

    for lv in levels:
        if remaining <= 0:
            break
        lp       = Decimal(lv.price)
        lq       = Decimal(lv.quantity)
        fill_qty = min(remaining, lq)
        fills.append({"price": str(lp), "qty": str(fill_qty)})
        total_notional += lp * fill_qty
        remaining      -= fill_qty

    filled = quantity - remaining
    vwap   = (total_notional / filled).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP) \
             if filled > 0 else Decimal("0")

    return {
        "fills":          fills,
        "filled_qty":     filled,
        "remaining_qty":  remaining,
        "vwap":           vwap,
        "total_notional": total_notional,
    }


# ──────────────────────────────────────────────────────────────────
app = Server("clob-mcp-bridge")
# ──────────────────────────────────────────────────────────────────


# ══════════════════════════ Resources ════════════════════════════

@app.list_resources()
async def list_resources() -> list[types.Resource]:
    return [
        types.Resource(
            uri         = "market://eth-usdc/stats",
            name        = "Market Stats (minimal)",
            description = (
                "Four numbers: best_bid, best_ask, spread, mid_price. "
                "Read this first to orient yourself. Cheapest context — prefer "
                "this over /orderbook when you only need the current price."
            ),
            mimeType    = "application/json",
        ),
        types.Resource(
            uri         = "market://eth-usdc/orderbook",
            name        = "Order Book — top 5 levels",
            description = (
                "Top-5 bid and ask price levels with quantity and order count. "
                "Read this when you need to assess available liquidity before "
                "a larger order. More expensive than /stats."
            ),
            mimeType    = "application/json",
        ),
        types.Resource(
            uri         = "market://eth-usdc/trades",
            name        = "Recent Trades — last 20",
            description = (
                "The 20 most recently executed trades. "
                "Useful for understanding recent transaction prices."
            ),
            mimeType    = "application/json",
        ),
        types.Resource(
            uri         = "market://eth-usdc/context",
            name        = "Market Briefing (synthesised prose)",
            description = (
                "A pre-synthesised plain-English market briefing. "
                "Includes price, depth, book imbalance, VWAP, market condition, "
                "and an action hint — all in one paragraph. "
                "Read this at session start for a complete market picture "
                "at the lowest reasoning cost: no JSON parsing required."
            ),
            mimeType    = "text/plain",
        ),
    ]


@app.read_resource()
async def read_resource(uri: str) -> types.ReadResourceResult:
    stub = _stub()

    # ── /stats ────────────────────────────────────────────────────
    if uri == "market://eth-usdc/stats":
        resp = stub.GetOrderBook(clob_pb2.GetOrderBookRequest(depth=1))
        data = {
            "best_bid":  resp.best_bid,
            "best_ask":  resp.best_ask,
            "spread":    resp.spread,
            "mid_price": resp.mid_price,
        }
        return types.ReadResourceResult(contents=[
            types.TextResourceContents(uri=uri, text=_pretty(data), mimeType="application/json")
        ])

    # ── /orderbook ────────────────────────────────────────────────
    if uri == "market://eth-usdc/orderbook":
        resp = stub.GetOrderBook(clob_pb2.GetOrderBookRequest(depth=5))
        data = {
            "bids": [{"price": lv.price, "qty": lv.quantity, "order_count": lv.order_count}
                     for lv in resp.bids],
            "asks": [{"price": lv.price, "qty": lv.quantity, "order_count": lv.order_count}
                     for lv in resp.asks],
            "best_bid":  resp.best_bid,
            "best_ask":  resp.best_ask,
            "spread":    resp.spread,
            "mid_price": resp.mid_price,
        }
        return types.ReadResourceResult(contents=[
            types.TextResourceContents(uri=uri, text=_pretty(data), mimeType="application/json")
        ])

    # ── /trades ───────────────────────────────────────────────────
    if uri == "market://eth-usdc/trades":
        resp = stub.GetTrades(clob_pb2.GetTradesRequest(limit=20))
        data = [{"id": t.trade_id[:8], "price": t.price, "qty": t.quantity}
                for t in resp.trades]
        return types.ReadResourceResult(contents=[
            types.TextResourceContents(uri=uri, text=_pretty(data), mimeType="application/json")
        ])

    # ── /context  (Idea 4) ────────────────────────────────────────
    if uri == "market://eth-usdc/context":
        book   = stub.GetOrderBook(clob_pb2.GetOrderBookRequest(depth=5))
        trades = stub.GetTrades(clob_pb2.GetTradesRequest(limit=20))
        text   = _build_context_briefing(book, trades)
        return types.ReadResourceResult(contents=[
            types.TextResourceContents(uri=uri, text=text, mimeType="text/plain")
        ])

    raise ValueError(f"Unknown resource URI: {uri}")


def _build_context_briefing(book, trades) -> str:
    """
    Synthesise a plain-English market briefing from live orderbook + trade data.

    Computes:
      • Book imbalance  = bid_depth / total_depth  (>0.65 = buy pressure)
      • VWAP            = Σ(price × qty) / Σ(qty)  over recent trades
      • Spread %        = spread / mid × 100
      • Market condition label derived from spread % and total depth
    """
    ts = time.strftime("%H:%M:%S UTC", time.gmtime())

    # ── depth & imbalance ──────────────────────────────────────
    bid_depth   = sum(Decimal(lv.quantity) for lv in book.bids)
    ask_depth   = sum(Decimal(lv.quantity) for lv in book.asks)
    total_depth = bid_depth + ask_depth
    imbalance   = float(bid_depth / total_depth) if total_depth > 0 else 0.5

    if imbalance > 0.65:
        pressure      = "BUY-HEAVY"
        pressure_note = "more buyers than sellers — ask prices may firm up"
    elif imbalance < 0.35:
        pressure      = "SELL-HEAVY"
        pressure_note = "more sellers than buyers — bid prices may soften"
    else:
        pressure      = "BALANCED"
        pressure_note = "supply and demand roughly equal"

    # ── VWAP over recent trades ────────────────────────────────
    vwap_str   = "n/a (no recent trades)"
    last_price = "n/a"
    if trades.trades:
        total_val = sum(Decimal(t.price) * Decimal(t.quantity) for t in trades.trades)
        total_vol = sum(Decimal(t.quantity) for t in trades.trades)
        if total_vol > 0:
            vwap     = (total_val / total_vol).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            vwap_str = f"${vwap} over {len(trades.trades)} trades"
        last_price = trades.trades[0].price   # newest first

    # ── spread & condition ─────────────────────────────────────
    mid        = Decimal(book.mid_price)  if book.mid_price  else Decimal("0")
    spread     = Decimal(book.spread)     if book.spread      else Decimal("0")
    spread_pct = float(spread / mid * 100) if mid > 0 else 0.0

    if total_depth == 0:
        condition = "EMPTY BOOK — no resting orders on either side"
        guidance  = "Place resting orders to seed the book."
    elif spread_pct < 0.5:
        condition = "LIQUID · TIGHT SPREAD"
        guidance  = "Limit orders near mid are likely to fill quickly."
    elif spread_pct < 2.0:
        condition = "MODERATE · NORMAL SPREAD"
        guidance  = "Consider pricing between mid and best to improve fill odds."
    else:
        condition = "ILLIQUID · WIDE SPREAD"
        guidance  = "Wide spread detected — crossing orders will pay significant slippage."

    # ── compose briefing ──────────────────────────────────────
    lines = [
        f"ETH/USDC · {ts}",
        "",
        f"PRICE      mid ${book.mid_price or 'n/a'} | "
        f"bid ${book.best_bid or 'n/a'} | "
        f"ask ${book.best_ask or 'n/a'} | "
        f"spread ${book.spread or 'n/a'} ({spread_pct:.2f}%)",
        f"DEPTH      buys {bid_depth:.4f} ETH / {len(book.bids)} levels | "
        f"sells {ask_depth:.4f} ETH / {len(book.asks)} levels",
        f"IMBALANCE  {imbalance:.2f} → {pressure} — {pressure_note}",
        f"VWAP       {vwap_str} | last trade ${last_price}",
        f"CONDITION  {condition}",
        "",
        f"GUIDANCE   {guidance}",
        f"           Use quote(side, quantity) to preview fill cost before placing.",
    ]
    return "\n".join(lines)


# ══════════════════════════ Tools ════════════════════════════════

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        # ── Read tools ────────────────────────────────────────────
        types.Tool(
            name        = "get_orderbook",
            description = (
                "Fetch a fresh order book snapshot right now. "
                "Call this immediately before place_order to get current prices — "
                "the resource /orderbook may be stale if time has passed. "
                "Returns top-N bid and ask levels, best bid/ask, spread, mid price."
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
                        "maximum":     MAX_LIMIT,
                        "description": "Number of trades to return",
                    },
                    "order_id": {
                        "type":        "string",
                        "description": "Filter to fills involving this order (optional)",
                    },
                },
            },
        ),

        # ── Quote tool (Idea 2) ───────────────────────────────────
        types.Tool(
            name        = "quote",
            description = (
                "Simulate executing a trade and preview the expected fill BEFORE committing. "
                "Call this instead of manually reading get_orderbook and doing arithmetic — "
                "it walks the book server-side and returns: "
                "the volume-weighted average fill price (VWAP), "
                "a per-level breakdown of which orders will be consumed, "
                "total slippage from mid, "
                "whether the full quantity is fillable right now, "
                "and a ready-to-use place_order recommendation. "
                "No order is placed. "
                "Use this whenever the user asks about cost or price impact of a trade."
            ),
            inputSchema = {
                "type": "object",
                "properties": {
                    "side": {
                        "type":        "string",
                        "enum":        ["BUY", "SELL"],
                        "description": "Direction of the intended trade",
                    },
                    "quantity": {
                        "type":        "string",
                        "description": "ETH amount to simulate, e.g. '2.0'",
                    },
                },
                "required": ["side", "quantity"],
            },
        ),

        # ── Write tools ───────────────────────────────────────────
        types.Tool(
            name        = "place_order",
            description = (
                "Place a Spot Limit Buy or Spot Limit Sell order on the ETH/USDC book. "
                "BEFORE calling this: either call quote(side, quantity) to get the "
                "expected price, or call get_orderbook for the current market. "
                "A limit order fills at your price or better; any unfilled remainder "
                "rests in the book. "
                "Returns: order_id (save it for cancel_order), quantity, "
                "filled_quantity, status, and resulting trades."
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
                "Will fail with ORDER_NOT_CANCELLABLE if the order is FILLED or CANCELLED — "
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
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    stub = _stub()
    try:
        result = _dispatch(stub, name, arguments)
    except grpc.RpcError as exc:
        result = _error(
            "ENGINE_ERROR",
            exc.details() or str(exc),
            "Check that the gRPC engine is running on " + GRPC_ADDR,
        )
    except (InvalidOperation, ValueError) as exc:
        result = _error("INVALID_INPUT", str(exc))
    except Exception as exc:
        logger.exception("Unexpected error in tool %s", name)
        result = _error("ENGINE_ERROR", str(exc))
    return [types.TextContent(type="text", text=_compact(result))]


# ── Dispatch ──────────────────────────────────────────────────────

def _dispatch(stub, name: str, args: dict) -> dict:

    # ── get_orderbook ──────────────────────────────────────────
    if name == "get_orderbook":
        depth = min(int(args.get("depth", 5)), MAX_DEPTH)
        resp  = stub.GetOrderBook(clob_pb2.GetOrderBookRequest(depth=depth))
        return {
            "bids": [{"price": lv.price, "qty": lv.quantity, "order_count": lv.order_count}
                     for lv in resp.bids],
            "asks": [{"price": lv.price, "qty": lv.quantity, "order_count": lv.order_count}
                     for lv in resp.asks],
            "best_bid":  resp.best_bid,
            "best_ask":  resp.best_ask,
            "spread":    resp.spread,
            "mid_price": resp.mid_price,
        }

    # ── quote (Idea 2) ────────────────────────────────────────
    if name == "quote":
        side     = args["side"].upper()
        quantity = Decimal(str(args["quantity"]))

        # Fetch deepest possible book to maximise fill simulation accuracy
        resp   = stub.GetOrderBook(clob_pb2.GetOrderBookRequest(depth=MAX_DEPTH))
        levels = resp.asks if side == "BUY" else resp.bids

        if not levels:
            return _error(
                "NO_LIQUIDITY",
                f"No resting {'asks' if side == 'BUY' else 'bids'} in the book.",
                "Place a resting order and wait for a counterparty, or try the other side.",
            )

        walk = _walk_book(levels, quantity, side)

        # Slippage from mid
        slippage_str = "n/a"
        if resp.mid_price and walk["filled_qty"] > 0:
            mid          = Decimal(resp.mid_price)
            slippage_pct = abs(walk["vwap"] - mid) / mid * 100
            slippage_str = f"{slippage_pct:.4f}%"

        fully_fillable = walk["remaining_qty"] == 0

        # Build recommendation
        if fully_fillable:
            worst_price  = walk["fills"][-1]["price"]
            recommendation = (
                f"place_order(side=\"{side}\", price=\"{worst_price}\", "
                f"quantity=\"{quantity}\") — guarantees full fill at "
                f"VWAP ~{walk['vwap']}"
            )
        else:
            if walk["fills"]:
                recommendation = (
                    f"Only {walk['filled_qty']} ETH immediately available. "
                    f"place_order(side=\"{side}\", price=\"{walk['fills'][-1]['price']}\", "
                    f"quantity=\"{walk['filled_qty']}\") for an immediate partial fill."
                )
            else:
                recommendation = "No liquidity available for this side."

        result = {
            "side":              side,
            "requested_qty":     str(quantity),
            "executable_qty":    str(walk["filled_qty"]),
            "vwap":              str(walk["vwap"]),
            "total_notional":    str(walk["total_notional"].quantize(Decimal("0.01"),
                                     rounding=ROUND_HALF_UP)),
            "slippage_from_mid": slippage_str,
            "fills":             walk["fills"],
            "fully_fillable":    fully_fillable,
            "recommendation":    recommendation,
        }

        if not fully_fillable:
            result["warning"] = _error(
                "PARTIAL_LIQUIDITY",
                f"Only {walk['filled_qty']} of {quantity} ETH is available at current depth.",
                f"Increase the order size tolerance or place a resting order for the remainder.",
            )

        return result

    # ── place_order ───────────────────────────────────────────
    if name == "place_order":
        try:
            price    = Decimal(str(args["price"]))
            quantity = Decimal(str(args["quantity"]))
        except InvalidOperation as exc:
            return _error("INVALID_INPUT", f"price/quantity must be decimal numbers: {exc}")

        # Pre-flight: warn if price is dramatically far from current mid
        book = stub.GetOrderBook(clob_pb2.GetOrderBookRequest(depth=1))
        if book.mid_price:
            mid       = Decimal(book.mid_price)
            deviation = abs(price - mid) / mid * 100
            if deviation > 50:
                return _error(
                    "PRICE_DEVIATION",
                    f"Price {price} is {deviation:.1f}% from current mid {mid}.",
                    f"Did you intend {mid}? Call quote(\"{args['side']}\", \"{quantity}\") "
                    f"to preview the fill at current market price.",
                )

        side = clob_pb2.BUY if args["side"].upper() == "BUY" else clob_pb2.SELL
        resp = stub.PlaceOrder(clob_pb2.PlaceOrderRequest(
            side=side, price=str(price), quantity=str(quantity)
        ))
        o = resp.order
        return {
            "order_id":        resp.order_id,
            "status":          _status(o.status),
            "quantity":        o.quantity,
            "filled_quantity": o.filled_quantity,
            "trades":          [{"id": t.trade_id[:8], "price": t.price, "qty": t.quantity}
                                for t in resp.trades],
        }

    # ── cancel_order ──────────────────────────────────────────
    if name == "cancel_order":
        resp = stub.CancelOrder(clob_pb2.CancelOrderRequest(order_id=args["order_id"]))
        if not resp.success:
            # Distinguish "not found" from "not cancellable" so the LLM knows what to do
            if "not found" in resp.message.lower():
                return _error(
                    "ORDER_NOT_FOUND",
                    f"No order with id '{args['order_id']}' exists.",
                    "Verify the order_id from a previous place_order response.",
                )
            return _error(
                "ORDER_NOT_CANCELLABLE",
                f"Cannot cancel: {resp.message}",
                "Call get_order to check the current status before attempting to cancel.",
            )
        return {"success": True, "message": resp.message}

    # ── get_order ─────────────────────────────────────────────
    if name == "get_order":
        resp = stub.GetOrder(clob_pb2.GetOrderRequest(order_id=args["order_id"]))
        if not resp.found:
            return _error(
                "ORDER_NOT_FOUND",
                f"No order with id '{args['order_id']}' exists.",
                "Verify the order_id from a previous place_order response.",
            )
        o = resp.order
        return {
            "order_id":        o.order_id,
            "side":            "BUY" if o.side == clob_pb2.BUY else "SELL",
            "price":           o.price,
            "quantity":        o.quantity,
            "filled_quantity": o.filled_quantity,
            "status":          _status(o.status),
        }

    # ── get_trades ────────────────────────────────────────────
    if name == "get_trades":
        limit = min(int(args.get("limit", 20)), MAX_LIMIT)
        resp  = stub.GetTrades(clob_pb2.GetTradesRequest(
            limit    = limit,
            order_id = args.get("order_id", ""),
        ))
        return [{"id": t.trade_id[:8], "price": t.price, "qty": t.quantity}
                for t in resp.trades]

    return _error("UNKNOWN_TOOL", f"No tool named '{name}' is registered.")


# ══════════════════════════ Entry Point ══════════════════════════

async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
