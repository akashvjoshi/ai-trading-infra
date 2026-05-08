"""
engine package — CLOB trading engine.

Public surface
──────────────
The ONLY supported way to interact with this engine is through the
gRPC service defined in proto/clob.proto.

  • To start the server : from engine.server import serve
  • To call the server  : use engine.generated.clob_pb2_grpc.ClobServiceStub

OrderBook is intentionally NOT re-exported here.  Any code outside this
package that imports engine.clob directly is bypassing the gRPC contract
and will break if the internal implementation changes.
"""
from engine.server import serve  # noqa: F401 — the only intended public export

__all__ = ["serve"]
