from __future__ import annotations

import csv
import json
from pathlib import Path

from cascade.sim.timeline import numeric, parse_durations
from cascade.stats import LogParser


def import_durations(stats_path: Path, mapping_path: Path, clock_mhz: float) -> dict:
    numeric(clock_mhz, "clock_mhz", positive=True)
    if stats_path.suffix == ".csv":
        stats = list(csv.DictReader(stats_path.open()))
        records = [
            {
                "uid": int(s["uid"]),
                "name": s["name"],
                "sim_cycles": numeric(float(s["gpu_sim_cycle"]), "gpu_sim_cycle"),
            }
            for s in stats
        ]
    else:
        text = stats_path.read_text()
        if "*** exit detected ***" not in text:
            raise ValueError("simulator log lacks completion marker")
        parser = LogParser("offline-import", clock_mhz)
        stats = parser.feed(text.splitlines()) + parser.finish()
        if any(
            len(str(s.stats.get("kernel_name", "")).split()) != 1
            or len(str(s.stats.get("kernel_launch_uid", "")).split()) != 1
            for s in stats
        ):
            raise ValueError("aggregate/concurrent stats cannot be mapped to individual kernels")
        records = [
            {"uid": s.kernel_launch_uid, "name": s.kernel_name, "sim_cycles": s.gpu_sim_cycle}
            for s in stats
        ]
    lookup = {r["uid"]: r for r in records}
    if not records or len(lookup) != len(records) or None in lookup:
        raise ValueError("simulator stats need unique, nonempty kernel UIDs")
    mapping = json.loads(mapping_path.read_text())
    if not isinstance(mapping, list) or not mapping:
        raise ValueError("mapping must be a nonempty list of uid/name/seq entries")
    output = []
    for entry in mapping:
        source = lookup.get(entry["uid"])
        if source is None or entry["name"].strip() != source["name"].strip():
            raise ValueError("kernel UID/name mapping mismatch")
        output.append(
            {k: entry[k] for k in ("repeat", "step", "seq") if k in entry}
            | {"sim_cycles": numeric(source["sim_cycles"], "sim_cycles"), "clock_mhz": clock_mhz}
        )
    result = {"kernels": output}
    parse_durations(result)
    return result
