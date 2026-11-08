"""Ordered, step-local instruction-trace to timing alignment (no GPU dependencies)."""

import csv
import json
import re
from difflib import SequenceMatcher
from pathlib import Path


def kernel_key(kernel: dict) -> tuple:
    def dims(value):
        return tuple(map(int, value.split("x") if isinstance(value, str) else value))
    return (kernel.get("mangled_name") or kernel["name"], dims(kernel["grid"]), dims(kernel["block"]))


def read_launches(traces: Path) -> list[dict]:
    launches = []
    for path in sorted(traces.glob("stats_ctx_*")):
        with path.open() as stream:
            for row in csv.reader(stream):
                if not row or not re.fullmatch(r"kernel-(\d+)-ctx_.*\.trace(?:\.xz)?", row[0].strip()):
                    continue
                if len(row) < 12:
                    raise ValueError(f"malformed tracer stats row in {path}: {row}")
                filename = row[0].strip()
                raw = traces / filename
                processed = traces / (filename.removesuffix(".xz") + "z")
                launches.append({"tracer_uid": int(re.match(r"kernel-(\d+)-", filename)[1]),
                                 "trace_file": filename, "name": row[1].strip(),
                                 "grid": list(map(int, row[2:5])), "block": list(map(int, row[6:9])),
                                 "traced": int(row[10]) > 0 or raw.is_file() or processed.is_file()})
    ids = [r["tracer_uid"] for r in launches]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate tracer UIDs across contexts; cannot identify kernels unambiguously")
    return sorted(launches, key=lambda r: r["tracer_uid"])


def indexed_steps(steps: list[dict]) -> dict:
    counts, result = {}, {}
    for step in sorted(steps, key=lambda s: s["step"]):
        repeat = step["repeat"]
        local = step.get("step_in_repeat", counts.get(repeat, 0))
        identity = repeat, local
        if identity in result:
            raise ValueError(f"duplicate repeat-local step {identity}")
        result[identity] = step
        counts[repeat] = local + 1
    return result


def align_step(tracer: list[dict], timeline: list[dict]) -> list[tuple[int, int]]:
    """Dimension-exact anchors first; name-only fallback within the remaining ordered gaps."""
    a, b = [kernel_key(k) for k in tracer], [kernel_key(k) for k in timeline]
    matches = []
    for tag, i, end_i, j, end_j in SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag == "equal":
            matches.extend(zip(range(i, end_i), range(j, end_j)))
        elif tag == "replace":
            fallback = SequenceMatcher(a=[k[0] for k in a[i:end_i]],
                                       b=[k[0] for k in b[j:end_j]], autojunk=False)
            for block in fallback.get_matching_blocks():
                matches.extend((i + block.a + n, j + block.b + n) for n in range(block.size))
    return matches


def build_alignment(timeline: dict, trace_steps: list[dict], launches: list[dict],
                    provenance: dict | None = None) -> dict:
    targets = indexed_steps(timeline["steps"])
    steps, kernels, missing_tracer, missing_timeline = [], [], [], []
    used = set()
    for identity, trace_step in indexed_steps(trace_steps).items():
        if not trace_step.get("traced"):
            continue
        first, end = trace_step.get("first_kernel_id"), trace_step.get("end_kernel_id")
        if first is None or end is None:
            raise ValueError(f"missing tracer kernel-id range for step {identity}")
        source = [k for k in launches if k["traced"] and first <= k["tracer_uid"] < end]
        target = targets.get(identity)
        expected = target["kernels"] if target else []
        pairs = align_step(source, expected)
        matched_a, matched_b = {a for a, _ in pairs}, {b for _, b in pairs}
        for a, b in pairs:
            launch, kernel = source[a], expected[b]
            if launch["tracer_uid"] in used:
                raise ValueError("overlapping traced step kernel-id ranges")
            used.add(launch["tracer_uid"])
            kernels.append({"tracer_uid": launch["tracer_uid"], "trace_file": launch["trace_file"],
                            "repeat": identity[0], "step": target["step"], "seq": kernel["seq"],
                            "grid": launch["grid"], "block": launch["block"],
                            "name": kernel_key(kernel)[0], "dims_match": kernel_key(launch) == kernel_key(kernel)})
        missing_tracer.extend({**k, "repeat": identity[0], "step_in_repeat": identity[1]}
                              for n, k in enumerate(source) if n not in matched_a)
        missing_timeline.extend({"repeat": identity[0], "step": target["step"], "seq": k["seq"], "name": k["name"]}
                                for n, k in enumerate(expected) if n not in matched_b)
        steps.append({"repeat": identity[0], "step": target["step"] if target else None,
                      "step_in_repeat": identity[1], "n_tracer": len(source), "n_timeline": len(expected),
                      "matched": len(pairs), "unmatched_tracer": len(source) - len(pairs),
                      "unmatched_timeline": len(expected) - len(pairs),
                      "exact": target is not None and [kernel_key(k) for k in source] ==
                      [kernel_key(k) for k in expected]})
    return {"provenance": {**(provenance or {}), "run_name": timeline["meta"].get("run_name"),
                           "traced_steps": [[s["repeat"], s["step_in_repeat"]] for s in steps],
                           "traced_timing_used": False}, "steps": steps, "kernels": kernels,
            "unmatched": {"n_tracer": len(missing_tracer), "n_timeline": len(missing_timeline),
                          "tracer": missing_tracer, "timeline": missing_timeline}}


def write_alignment(run_dir: str | Path) -> dict:
    run = Path(run_dir)
    manifest = json.loads((run / "run.json").read_text())
    result = build_alignment(json.loads((run / "corrected_timeline.json").read_text()),
                             json.loads((run / "traced/trace_steps.json").read_text()),
                             read_launches(run / "traced/traces"),
                             {"trace_repeat": manifest["trace_repeat"]})
    (run / "trace_alignment.json").write_text(json.dumps(result, indent=2))
    return result
