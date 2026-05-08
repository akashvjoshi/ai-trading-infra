"""
Interactive demo — starts the engine in-process, seeds some orders,
then opens a REPL for natural-language trading via the LLM service.

Usage: python scripts/demo.py
"""
import logging
import sys
import time
from decimal import Decimal
import os

# Add the parent directory to sys.path to enable imports from engine and llm_service
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── bootstrap: start the engine in a background thread ────────────
import threading
from engine.server import serve

logging.basicConfig(level=logging.WARNING)
srv = serve(port=50051)
time.sleep(0.5)   # let gRPC bind

# ── seed the order book so there's something to look at ───────────
import grpc
from engine.generated import clob_pb2, clob_pb2_grpc

stub = clob_pb2_grpc.ClobServiceStub(grpc.insecure_channel("localhost:50051"))

# Seed asks (sell orders)
for price, qty in [("3220", "2"), ("3230", "1.5"), ("3240", "3")]:
    stub.PlaceOrder(clob_pb2.PlaceOrderRequest(side=clob_pb2.SELL, price=price, quantity=qty))

# Seed bids (buy orders)
for price, qty in [("3200", "1"), ("3190", "2"), ("3180", "0.5")]:
    stub.PlaceOrder(clob_pb2.PlaceOrderRequest(side=clob_pb2.BUY, price=price, quantity=qty))

print("\n ETH/USDC order book seeded.")
print(" Mid price: ~3210 USDC | Spread: 20 USDC")
print(" Type 'quit' to exit.\n")

# ── REPL ───────────────────────────────────────────────────────────
from llm_service.service import TradingService

svc = TradingService(grpc_addr="localhost:50051")

while True:
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye.")
        break

    if user_input.lower() in ("quit", "exit", "q"):
        print("Bye.")
        break
    if not user_input:
        continue

    response = svc.chat(user_input)
    print(f"\nAssistant: {response}\n")

srv.stop(0)
