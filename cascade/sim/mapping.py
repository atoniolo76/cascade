from __future__ import annotations

import bz2
import gzip
import hashlib
import json
import lzma
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

from cascade.sim.importer import map_records, read_stats
from cascade.sim.timeline import load_timeline, parse_durations, simulate


def signature(kind, kernel):
    dims = []
    for field in ("grid", "block"):
        value = kernel[field]
        if isinstance(value, str) and re.fullmatch(r"\d+x\d+x\d+", value):
            value = [int(n) for n in value.split("x")]
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 3
            or any(type(n) is not int or n <= 0 for n in value)
        ):
            raise ValueError(f"{field} must contain three positive integers")
        dims.append(tuple(value))
    return kind, kernel["name"], *dims


def trace_headers(kernelslist: Path) -> list[dict]:
    headers = []
    for line in kernelslist.read_text().splitlines():
        filename = line.strip()
        if not filename or filename.startswith(("#", "Memcpy")):
            continue
        path = kernelslist.parent / filename
        opener = {".xz": lzma.open, ".gz": gzip.open, ".bz2": bz2.open}.get(path.suffix, open)
        fields = {}
        with opener(path, "rb") as stream:
            for _ in range(128):
                raw = stream.readline(8192)
                if not raw:
                    break
                try:
                    text = raw.decode("utf-8").strip()
                except UnicodeDecodeError as error:
                    raise ValueError(f"missing trace header in {filename}") from error
                if text.startswith("-") and "=" in text:
                    field, value = text[1:].split("=", 1)
                    fields[field.strip()] = value.strip()
                if all(k in fields for k in ("kernel name", "kernel id", "grid dim", "block dim")):
                    break
        if not all(k in fields for k in ("kernel name", "kernel id", "grid dim", "block dim")):
            raise ValueError(f"missing trace header in {filename}")
        kernel = {"name": fields["kernel name"], "trace_id": int(fields["kernel id"])}
        for field in ("grid", "block"):
            value = fields[f"{field} dim"]
            if not re.fullmatch(r"\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)", value):
                raise ValueError(f"invalid {field} dim in {filename}")
            kernel[field] = [int(n) for n in value.strip("()").split(",")]
        signature("", kernel)
        headers.append(kernel)
    if not headers or len({h["trace_id"] for h in headers}) != len(headers):
        raise ValueError("trace list needs nonempty, unique kernel IDs")
    return headers


def build_mapping(
    stats_path: Path,
    kernelslist: Path,
    timeline_path: Path,
    clock_mhz: float,
    *,
    identity_path: Path | None = None,
    phases_path: Path | None = None,
    allow_shape_transfer: bool = False,
) -> dict:
    """Identity requires capture IDs/sidecar; geometry alone never establishes identity."""
    data = load_timeline(timeline_path)
    records = read_stats(stats_path)
    headers = trace_headers(kernelslist)
    lookup = {r["uid"]: r for r in records}
    if set(lookup) != set(range(1, len(headers) + 1)):
        raise ValueError("stats must contain one launch UID per trace, starting at 1")
    sources = {}
    for uid, header in enumerate(headers, 1):
        if header["name"] != lookup[uid]["name"]:
            raise ValueError("trace/stats UID/name mismatch")
        sources[uid] = header | {"sim_cycles": lookup[uid]["sim_cycles"]}
    targets = {}
    capture_ids = defaultdict(list)
    for step in data["steps"]:
        if step["warmup"]:
            continue
        for kernel in step["kernels"]:
            target = step["repeat"], step["step"], kernel["seq"]
            targets[target] = (step["kind"], kernel)
            if "trace_kernel_id" in kernel:
                if type(kernel["trace_kernel_id"]) is not int or kernel["trace_kernel_id"] < 0:
                    raise ValueError("trace_kernel_id must be a nonnegative integer")
                capture_ids[kernel["trace_kernel_id"]].append(target)
    identities = {}
    for uid, source in sources.items():
        matches = capture_ids.get(source["trace_id"], [])
        if len(matches) > 1:
            raise ValueError("ambiguous capture trace_kernel_id")
        if matches:
            identities[matches[0]] = uid
    if identity_path:
        sidecar = json.loads(identity_path.read_text())
        if not isinstance(sidecar, list) or not sidecar:
            raise ValueError("identity sidecar must be a nonempty mapping list")
        for entry in sidecar:
            target = tuple(entry[k] for k in ("repeat", "step", "seq"))
            uid = entry["uid"]
            if (
                any(type(n) is not int for n in (*target, uid))
                or target not in targets
                or uid not in sources
                or entry["name"] != sources[uid]["name"]
            ):
                raise ValueError("identity sidecar target/UID/name mismatch")
            if target in identities and identities[target] != uid:
                raise ValueError("conflicting capture identities")
            identities[target] = uid
    if len(set(identities.values())) != len(identities):
        raise ValueError("capture UID must identify one occurrence")
    phases = {}
    for target, uid in identities.items():
        kind, kernel = targets[target]
        if signature(kind, kernel) != signature(kind, sources[uid]):
            raise ValueError("identity geometry/name mismatch")
        if uid in phases and phases[uid] != kind:
            raise ValueError("conflicting source phases")
        phases[uid] = kind
    if phases_path:
        annotations = json.loads(phases_path.read_text())
        if not isinstance(annotations, list):
            raise ValueError("source phases must be a list")
        seen = set()
        for entry in annotations:
            uid, kind = entry["uid"], entry["kind"]
            if (
                type(uid) is not int
                or uid in seen
                or uid not in sources
                or kind not in ("prefill", "decode")
            ):
                raise ValueError("invalid source phase annotation")
            if uid in phases and phases[uid] != kind:
                raise ValueError("conflicting source phases")
            seen.add(uid)
            phases[uid] = kind
    groups, grouped = [], {}
    for uid, source in sources.items():
        if uid not in phases:
            continue
        sig = signature(phases[uid], source)
        if sig not in grouped:
            grouped[sig] = len(groups)
            groups.append(
                dict(
                    group=len(groups),
                    kind=phases[uid],
                    name=source["name"],
                    grid=source["grid"],
                    block=source["block"],
                    source_uids=[],
                    source_cycles=[],
                )
            )
        group = groups[grouped[sig]]
        group["source_uids"].append(uid)
        group["source_cycles"].append(source["sim_cycles"])
    for group in groups:
        group["sim_cycles"] = median(group["source_cycles"])
    mapping = []
    for target, (kind, kernel) in targets.items():
        if target in identities:
            uid = identities[target]
            entry = {"uid": uid, "name": kernel["name"], "method": "identity"}
        elif allow_shape_transfer and signature(kind, kernel) in grouped:
            group = groups[grouped[signature(kind, kernel)]]
            entry = dict(
                uid=group["source_uids"][0],
                name=kernel["name"],
                method="shape_transfer",
                source_uids=group["source_uids"],
                source_group=group["group"],
            )
        else:
            continue
        mapping.append(entry | dict(zip(("repeat", "step", "seq"), target, strict=True)))
    if not mapping:
        raise ValueError("no kernels matched; provide capture identity or explicit phased transfer")
    durations = map_records(records, mapping, clock_mhz)
    durations = {
        "provenance": {
            "real_run": data["meta"].get("run_name"),
            "input_sha256": hashlib.sha256(timeline_path.read_bytes()).hexdigest(),
            "stats_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
            "kernelslist_sha256": hashlib.sha256(kernelslist.read_bytes()).hexdigest(),
            "trace_headers_sha256": hashlib.sha256(
                json.dumps(headers, sort_keys=True).encode()
            ).hexdigest(),
            "identity_sha256": hashlib.sha256(identity_path.read_bytes()).hexdigest()
            if identity_path
            else None,
            "source_phases_sha256": hashlib.sha256(phases_path.read_bytes()).hexdigest()
            if phases_path
            else None,
            "method": (
                "Capture identity first; explicitly enabled "
                "phase/name/grid/block median transfer otherwise."
            ),
        },
        "source_groups": groups,
        **durations,
    }
    coverage = simulate(data, parse_durations(durations))["summary"]["simulation"]["coverage"]
    return {"mapping": mapping, "durations": durations, "coverage": coverage}
