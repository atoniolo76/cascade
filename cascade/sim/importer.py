from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median

from cascade.sim.timeline import numeric, parse_durations
from cascade.stats import LogParser


def read_stats(stats_path: Path) -> list[dict]:
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
        parser = LogParser("offline-import", 1)
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
    for record in records:
        cycles = numeric(record["sim_cycles"], "sim_cycles")
        record["sim_cycles"] = int(cycles) if cycles == int(cycles) else cycles
    return records


def map_records(records: list[dict], mapping: list[dict], clock_mhz: float) -> dict:
    numeric(clock_mhz, "clock_mhz", positive=True)
    clock_mhz = int(clock_mhz) if clock_mhz == int(clock_mhz) else clock_mhz
    lookup = {r["uid"]: r for r in records}
    if not isinstance(mapping, list) or not mapping:
        raise ValueError("mapping must be a nonempty list of uid/name/seq entries")
    output = []
    for entry in mapping:
        method = entry.get("method", "identity")
        if method not in ("identity", "shape_transfer"):
            raise ValueError("unknown kernel mapping method")
        uids = entry.get("source_uids", [entry["uid"]])
        if (
            not isinstance(uids, list)
            or not uids
            or any(type(uid) is not int or uid <= 0 for uid in uids)
            or len(set(uids)) != len(uids)
            or entry["uid"] not in uids
            or (method == "identity" and len(uids) != 1)
        ):
            raise ValueError("invalid mapping source_uids")
        sources = [lookup.get(uid) for uid in uids]
        if any(s is None or entry["name"].strip() != s["name"].strip() for s in sources):
            raise ValueError("kernel UID/name mapping mismatch")
        output.append(
            {k: entry[k] for k in ("repeat", "step", "seq") if k in entry}
            | {"sim_cycles": median(s["sim_cycles"] for s in sources), "clock_mhz": clock_mhz}
            | ({"source_group": entry["source_group"]} if "source_group" in entry else {})
        )
    result = {"kernels": output}
    parse_durations(result)
    return result


def import_durations(stats_path: Path, mapping_path: Path, clock_mhz: float) -> dict:
    mapping = json.loads(mapping_path.read_text())
    if isinstance(mapping, dict) and isinstance(mapping.get("kernels"), list):
        mapping = [
            {"uid": entry["tracer_uid"], **{k: entry[k] for k in ("name", "repeat", "step", "seq")}}
            for entry in mapping["kernels"]
        ]
    return map_records(read_stats(stats_path), mapping, clock_mhz)
