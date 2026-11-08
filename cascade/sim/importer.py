from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from statistics import median

from cascade.sim.accelsim import duration_records
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
    return map_records(read_stats(stats_path), json.loads(mapping_path.read_text()), clock_mhz)


def aligned_durations(
    records: list[dict], timeline: dict, alignment: dict, kernelslist: Path, clock_mhz: float
) -> dict:
    """Translate whole-run launch positions to tracer UIDs before propagation."""
    numeric(clock_mhz, "clock_mhz", positive=True)
    uids = []
    for line in kernelslist.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("Memcpy"):
            continue
        match = re.fullmatch(r"kernel-(\d+)(?:-ctx_0x[0-9a-fA-F]+)?\.trace(?:z|g|\.xz)?", line)
        if match is None:
            raise ValueError(f"unsupported kernelslist.g entry: {line!r}")
        uids.append(int(match[1]))
    lookup = {r["uid"]: r for r in records}
    if (
        not uids
        or len(set(uids)) != len(uids)
        or len(lookup) != len(records)
        or set(lookup) != set(range(1, len(uids) + 1))
    ):
        raise ValueError("simulator launch UIDs must match kernelslist.g positions 1..N")
    mapping = {k["tracer_uid"]: k for k in alignment["kernels"]}
    if not mapping or len(mapping) != len(alignment["kernels"]) or not set(mapping) <= set(uids):
        raise ValueError("alignment tracer UIDs must be unique and present in kernelslist.g")
    results = []
    for position, uid in enumerate(uids, 1):
        record = lookup[position]
        if uid not in mapping:
            continue
        if record["name"].strip() != mapping[uid]["name"].strip():
            raise ValueError(
                f"kernel UID/name mapping mismatch at launch {position}, tracer UID {uid}"
            )
        results.append(
            {"uid": uid, "cycles": numeric(record["sim_cycles"], "sim_cycles"), "status": "ok"}
        )
    durations, _ = duration_records(timeline, alignment, results, clock_mhz, propagate="interp")
    result = {"kernels": durations}
    parse_durations(result)
    return result
