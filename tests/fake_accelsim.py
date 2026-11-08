#!/usr/bin/env python3
"""Offline stand-in for the simulator binary and trace postprocessor."""

import json
import os
import sys
import time
from pathlib import Path

if Path(sys.argv[0]).name == "post-traces-processing":
    directory = Path(sys.argv[1])
    names = []
    for raw in directory.glob("*.trace.xz"):
        processed = raw.with_name(raw.name.removesuffix(".xz") + "z")
        processed.write_text(raw.read_text())
        names.append(processed.name)
    (directory / "kernelslist.g").write_text("\n".join(names) + "\n")
else:
    assert os.environ.get("CASCADE_TEST_RUNTIME") == "ready"
    listing = Path(sys.argv[sys.argv.index("-trace") + 1])
    lines = listing.read_text().splitlines()
    assert len(lines) == 1 and lines[0].startswith("kernel-")
    spec = json.loads((listing.parent / lines[0]).read_bytes().split(b"\0", 1)[-1])
    if spec.get("mode") == "fail":
        sys.exit(2)
    if spec.get("mode") == "timeout":
        time.sleep(2)
    print("kernel_name =", spec["name"])
    print("kernel_launch_uid = 1")
    print("gpu_sim_cycle =", spec["cycles"])
    print("gpu_tot_sim_cycle =", spec["cycles"])
