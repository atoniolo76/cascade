"""Stands in for post-traces-processing: same argv shape, writes kernelslist.g and .tracez."""

import sys
from pathlib import Path

traces = Path(sys.argv[1])
assert sys.argv[2] == "-j"
if (traces / "FAIL").exists():
    print("simulated post-processing failure")
    sys.exit(3)
out = []
for raw in sorted(traces.glob("kernelslist_ctx_*")):
    for line in raw.read_text().splitlines():
        if line.startswith("kernel"):
            name = line.removesuffix(".xz") + "z"
            (traces / name).write_bytes(b"z")
            out.append(name)
        elif line:
            out.append(line)
(traces / "kernelslist.g").write_text("\n".join(out) + "\n")
calls = traces.parent / "postprocess-calls"
calls.write_text(str(int(calls.read_text() or 0) + 1) if calls.exists() else "1")
