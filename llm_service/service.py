"""
LLM Interaction Service — natural language interface to the CLOB.

Handles:
  • Multi-turn conversation with tool-calling agentic loop
  • Guardrail enforcement before and after LLM calls
  • Tool call tracking for evaluation introspection
"""
from __future__ import annotations

import json
import logging
import os
from decimal import Decimal
from typing import Optional

import anthropic
import grpc

from engine.generated import clob_pb2, clob_pb2_grpc
from llm_service.guardrails import Guardrails

logger = logging.getLogger(__name__)

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

SYSTEM_PROMPT = """You are a professional trading assistant for the ETH/USDC spot market.
You interact with a Central Limit Order Book (CLOB) on behalf of the user.

## Available actions
- place_order   — submit a limit buy or sell order
- cancel_order  — cancel an open order by ID
- get_orderbook — inspect current market depth
- get_order     — check the status of a specific order
- get_trades    — view recent trade history

## Rules you must follow
1. Always call get_orderbook before placing an order so you know the current price.
2. Confirm order details (side, price, quantity, total USDC value) with the user BEFORE placing.
3. If the request is ambiguous (missing price or quantity), ask for clarification. Do not guess.
4. Warn explicitly if an order is large relative to visible liquidity.
5. Never manufacture order IDs — only use IDs returned by previous tool calls.
6. Respond concisely; include only the fields the user asked about.

## Market parameters
- Trading pair : ETH / USDC
- Min order    : 0.001 ETH
- Max order    : 100 ETH
- Price unit   : USDC (2 decimal places)
- Qty unit     : ETH  (up to 6 decimal places)
"""

_TOOLS: list[dict] = [
    {
        "name": "place_order",
        "description": "Place a limit buy or sell order on the ETH/USDC order book.",
        "input_schema": {
            "type": "object",
            "properties": {
                "side":     {"type": "string", "enum": ["BUY", "SELL"]},
                "price":    {"type": "string", "description": "Limit price in USDC"},
                "quantity": {"type": "string", "description": "ETH amount"},
            },
            "required": ["side", "price", "quantity"],
        },
    },
    {
        "name": "cancel_order",
        "description": "Cancel an open order by UUID.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "get_orderbook",
        "description": "Fetch current order book depth (bids, asks, spread, mid price).",
        "input_schema": {
            "type": "object",
            "properties": {"depth": {"type": "integer", "default": 5}},
        },
    },
    {
        "name": "get_order",
        "description": "Look up a specific order by ID.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "get_trades",
        "description": "Retrieve recent trade history.",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit":    {"type": "integer", "default": 20},
                "order_id": {"type": "string"},
            },
        },
    },
]


class TradingService:
    def __init__(
        self,
        grpc_addr: str            = "localhost:50051",
        model:     str            = MODEL,
        guardrails: Optional[Guardrails] = None,
    ) -> None:
        self.client     = anthropic.Anthropic()
        self.model      = model
        self.stub       = clob_pb2_grpc.ClobServiceStub(grpc.insecure_channel(grpc_addr))
        self.guardrails = guardrails or Guardrails()
        self.history:   list[dict] = []
        # Introspection — populated during each chat() call
        self.last_tool_calls: list[dict] = []

    # ───────────────────────── Public ─────────────────────────────

    def chat(self, user_message: str) -> str:
        """Process a natural-language user message and return the LLM reply."""
        self.last_tool_calls = []

        guard = self.guardrails.check_input(user_message)
        if not guard.allowed:
            return f"[BLOCKED] {guard.reason}"

        self.history.append({"role": "user", "content": user_message})

        response = self._call_llm()

        # Agentic loop — resolve all tool_use blocks
        while response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = self._execute_tool(block.name, block.input)
                    self.last_tool_calls.append({"name": block.name, "input": block.input, "result": result})
                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     json.dumps(result),
                    })

            self.history.append({"role": "assistant", "content": response.content})
            self.history.append({"role": "user",      "content": tool_results})
            response = self._call_llm()

        reply = "".join(b.text for b in response.content if hasattr(b, "text"))
        self.history.append({"role": "assistant", "content": response.content})
        return reply

    def reset(self) -> None:
        self.history = []
        self.last_tool_calls = []

    # ───────────────────────── Private ────────────────────────────

    def _call_llm(self):
        return self.client.messages.create(
            model      = self.model,
            max_tokens = 4_096,
            system     = SYSTEM_PROMPT,
            tools      = _TOOLS,
            messages   = self.history,
        )

    def _execute_tool(self, name: str, args: dict) -> dict:
        guard = self.guardrails.check_tool_call(name, args)
        if not guard.allowed:
            logger.warning("Tool call blocked: %s(%s) — %s", name, args, guard.reason)
            return {"error": f"Blocked: {guard.reason}"}

        try:
            return self._dispatch(name, args)
        except grpc.RpcError as exc:
            logger.error("gRPC error executing %s: %s", name, exc.details())
            return {"error": exc.details()}
        except Exception as exc:
            logger.exception("Unexpected error executing %s", name)
            return {"error": str(exc)}

    def _dispatch(self, name: str, args: dict) -> dict:
        if name == "place_order":
            side = clob_pb2.BUY if args["side"].upper() == "BUY" else clob_pb2.SELL
            resp = self.stub.PlaceOrder(clob_pb2.PlaceOrderRequest(
                side=side, price=str(args["price"]), quantity=str(args["quantity"])
            ))
            # Update reference price for future guardrail checks
            book = self.stub.GetOrderBook(clob_pb2.GetOrderBookRequest(depth=1))
            if book.mid_price:
                self.guardrails.update_reference_price(Decimal(book.mid_price))
            return {
                "order_id":  resp.order_id,
                "status":    _status(resp.order.status),
                "filled":    resp.order.filled_quantity,
                "remaining": str(float(resp.order.quantity) - float(resp.order.filled_quantity)),
                "trades":    len(resp.trades),
            }

        if name == "cancel_order":
            resp = self.stub.CancelOrder(clob_pb2.CancelOrderRequest(order_id=args["order_id"]))
            return {"success": resp.success, "message": resp.message}

        if name == "get_orderbook":
            resp = self.stub.GetOrderBook(clob_pb2.GetOrderBookRequest(depth=args.get("depth", 5)))
            if resp.mid_price:
                self.guardrails.update_reference_price(Decimal(resp.mid_price))
            return {
                "bids":      [{"price": lv.price, "qty": lv.quantity} for lv in resp.bids],
                "asks":      [{"price": lv.price, "qty": lv.quantity} for lv in resp.asks],
                "best_bid":  resp.best_bid,
                "best_ask":  resp.best_ask,
                "spread":    resp.spread,
                "mid_price": resp.mid_price,
            }

        if name == "get_order":
            resp = self.stub.GetOrder(clob_pb2.GetOrderRequest(order_id=args["order_id"]))
            if not resp.found:
                return {"error": "Order not found"}
            o = resp.order
            return {
                "order_id": o.order_id,
                "side":     "BUY" if o.side == clob_pb2.BUY else "SELL",
                "price":    o.price,
                "quantity": o.quantity,
                "filled":   o.filled_quantity,
                "status":   _status(o.status),
            }

        if name == "get_trades":
            resp = self.stub.GetTrades(clob_pb2.GetTradesRequest(
                limit    = args.get("limit", 20),
                order_id = args.get("order_id", ""),
            ))
            return [{"price": t.price, "qty": t.quantity} for t in resp.trades]

        return {"error": f"Unknown tool: {name}"}


def _status(code: int) -> str:
    return {0: "OPEN", 1: "FILLED", 2: "PARTIALLY_FILLED", 3: "CANCELLED"}.get(code, "UNKNOWN")
