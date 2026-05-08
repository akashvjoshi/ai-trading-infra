#!/usr/bin/env bash
# Regenerate gRPC stubs from proto/clob.proto
# Requires: pip install grpcio-tools
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."

python -m grpc_tools.protoc \
  -I"$ROOT/proto" \
  --python_out="$ROOT/engine/generated" \
  --grpc_python_out="$ROOT/engine/generated" \
  "$ROOT/proto/clob.proto"

# Fix relative imports in generated files (grpc_tools quirk)
sed -i 's/^import clob_pb2/from engine.generated import clob_pb2/' \
  "$ROOT/engine/generated/clob_pb2_grpc.py"

echo "Proto stubs regenerated in engine/generated/"
