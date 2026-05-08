"""
Cross-platform proto generation script.
Run: python scripts/generate_proto.py
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "engine" / "generated"

def main():
    cmd = [
        sys.executable, "-m", "grpc_tools.protoc",
        f"-I{ROOT / 'proto'}",
        f"--python_out={OUT}",
        f"--grpc_python_out={OUT}",
        str(ROOT / "proto" / "clob.proto"),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Fix absolute import in generated grpc file → package-relative
    grpc_file = OUT / "clob_pb2_grpc.py"
    text = grpc_file.read_text()
    text = re.sub(
        r"^import clob_pb2",
        "from engine.generated import clob_pb2",
        text,
        flags=re.MULTILINE,
    )
    grpc_file.write_text(text)
    print(f"Stubs written to {OUT}")

if __name__ == "__main__":
    main()
