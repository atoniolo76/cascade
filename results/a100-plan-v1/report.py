"""Reproduce native OPT-125m A100 tables from saved profiling and simulation artifacts."""

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

HERE = Path(__file__).resolve().parent
GROUPS = (
    "QKV GEMM",
    "Attention",
    "KV write",
    "Out-proj",
    "MLP up GEMM",
    "MLP down GEMM",
    "Split-K reduction",
    "lm_head",
    "Small elementwise",
    "Layer norm",
    "Other setup/index/reduction",
)


def read_json(path):
    if path.is_file():
        return json.loads(path.read_text())
    with gzip.open(str(path) + ".gz", "rt") as stream:
        return json.load(stream)


def groups_for_step(kernels):
    """Infer the four matrix roles per OPT layer, then the final vocabulary projection."""
    matrices = [k for k in kernels if any(s in k["name"].lower() for s in ("gemm", "gemv"))]
    if not kernels:
        return {}
    if len(matrices) != 49:
        raise ValueError(f"expected 12 × 4 + 1 OPT matrix kernels, got {len(matrices)}")
    roles = ("QKV GEMM", "Out-proj", "MLP up GEMM", "MLP down GEMM") * 12 + ("lm_head",)
    result = {k["seq"]: role for k, role in zip(matrices, roles, strict=True)}
    for kernel in kernels:
        if kernel["seq"] in result:
            continue
        name = kernel["name"]
        if "reshape_and_cache" in name:
            group = "KV write"
        elif "flash_fwd" in name:
            group = "Attention"
        elif "layer_norm" in name:
            group = "Layer norm"
        elif "splitKreduce" in name:
            group = "Split-K reduction"
        elif "elementwise_kernel" in name:
            group = "Small elementwise"
        else:
            group = "Other setup/index/reduction"
        result[kernel["seq"]] = group
    return result


def table(headers, rows):
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    )


def summarize_group(group, records, simulated):
    if not records:
        return None
    real_us = mean(r[1] for r in records)
    predicted = [simulated[r[0]] for r in records if r[0] in simulated]
    if predicted and len(predicted) != len(records):
        raise ValueError(f"partial within-group coverage for {group}")
    sim_us = mean(r["sim_cycles"] / r["clock_mhz"] for r in predicted) if predicted else None
    return {
        "group": group,
        "n": len(records),
        "modeled_n": len(predicted),
        "real_us": real_us,
        "cycles": mean(r["sim_cycles"] for r in predicted) if predicted else None,
        "sim_us": sim_us,
        "signed_error_pct": (sim_us / real_us - 1) * 100 if predicted else None,
    }


def analyze(source, sim_dir):
    timeline = read_json(source / "corrected_timeline.json")
    alignment = read_json(source / "trace_alignment.json")
    run = read_json(source / "run.json")
    traced_steps = read_json(source / "traced/trace_steps.json")
    summary = read_json(sim_dir / "summary.json")
    durations = read_json(sim_dir / "sim_durations.json")
    replay = read_json(sim_dir / "simulated_timeline.json")
    compare = read_json(sim_dir / "compare.json")
    steps = [s for s in timeline["steps"] if not s["warmup"]]
    lookup = {}
    mixes = defaultdict(list)
    for step in steps:
        groups = groups_for_step(step["kernels"])
        for kernel in step["kernels"]:
            lookup[step["repeat"], step["step"], kernel["seq"]] = (kernel, groups[kernel["seq"]])
        for group in GROUPS:
            selected = [k for k in step["kernels"] if groups[k["seq"]] == group]
            mixes[step["kind"], group].append(
                (len(selected), sum(k["duration_ns"] for k in selected) / 1000)
            )
    real_steps = []
    for kind in ("prefill", "decode"):
        selected = [s for s in steps if s["kind"] == kind]
        real_steps.append(
            {
                "kind": kind,
                "n": len(selected),
                **{
                    field.replace("_ns", "_ms"): mean(s[field] for s in selected) / 1e6
                    for field in ("baseline_wall_ns", "gpu_busy_ns", "host_ns", "cuda_api_ns")
                },
                "wall_p50_ms": median(s["baseline_wall_ns"] for s in selected) / 1e6,
                "sim_gpu_busy_us": mean(
                    s["gpu_busy_ns"]
                    for s in replay["steps"]
                    if not s["warmup"] and s["kind"] == kind
                )
                / 1000,
            }
        )
    samples = defaultdict(list)
    by_uid = {}
    for entry in alignment["kernels"]:
        identity = entry["repeat"], entry["step"], entry["seq"]
        kernel, group = lookup[identity]
        by_uid[entry["tracer_uid"]] = group
        samples[group].append((identity, kernel["duration_ns"] / 1000))
    simulated = {(r["repeat"], r["step"], r["seq"]): r for r in durations["kernels"]}
    group_rows = []
    for group in GROUPS:
        row = summarize_group(group, samples[group], simulated)
        if row is not None:
            group_rows.append(row)
    verification = []
    for group in summary["dedup_verification"]["groups"]:
        verification.append(
            {
                key: group[key]
                for key in (
                    "representative",
                    "name",
                    "max_absolute_cycles",
                    "max_relative",
                    "comparisons",
                )
            }
            | {"group": " / ".join(dict.fromkeys(by_uid[uid] for uid in group["members"]))}
        )
    traces = []
    for step in traced_steps:
        if step["traced"]:
            traces.append(
                {
                    "step": step["step_in_repeat"],
                    "kind": step["kind"],
                    "wall_s": (step["monotonic_end_ns"] - step["monotonic_start_ns"]) / 1e9,
                    "kernels": step["end_kernel_id"] - step["first_kernel_id"],
                }
            )
    real_requests = [r for r in timeline["requests"] if not r["warmup"]]
    sim_requests = [r for r in replay["requests"] if not r["warmup"]]
    mix = [
        {
            "kind": kind,
            "group": group,
            "kernels_per_step": mean(r[0] for r in records),
            "gpu_us_per_step": mean(r[1] for r in records),
        }
        for (kind, group), records in mixes.items()
    ]
    return {
        "workload": run["workload"],
        "metadata": timeline["meta"],
        "stage_wall_s": run["stage_wall_s"],
        "trace_steps": traces,
        "real_steps": real_steps,
        "real_kernel_mix": mix,
        "groups": group_rows,
        "real_requests": real_requests,
        "sim_requests": sim_requests,
        "summary": summary,
        "verification": verification,
        "provenance": dict(Counter(r["sim"].split(":")[0] for r in durations["kernels"])),
        "coverage": replay["summary"]["simulation"]["coverage"],
        "compare": compare,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=HERE / "profile")
    parser.add_argument("--sim-dir", type=Path, default=HERE / "modal-complete")
    args = parser.parse_args()
    result = analyze(args.source, args.sim_dir)
    (args.sim_dir / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    rows = result["groups"]
    with (args.sim_dir / "kernel-groups.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    sections = [
        f"# Native A100 measured tables: {args.sim_dir.name}\n\n"
        f"Simulation counts: `{result['summary']['counts']}`. "
        "Generated by `report.py`; matrix roles are inferred "
        "from OPT execution order, not NVTX annotations. Signed group error uses the matched "
        "Nsight kernel mean; request/step comparisons use the real A100 unprofiled baseline.",
        "## Pipeline\n"
        + table(
            ["Stage", "Wall seconds"],
            [[name, f"{seconds:.6f}"] for name, seconds in result["stage_wall_s"].items()],
        ),
        "## Instrumented steps\n"
        + table(
            ["Local step", "Kind", "Kernels", "Seconds"],
            [
                [r["step"], r["kind"], r["kernels"], f"{r['wall_s']:.6f}"]
                for r in result["trace_steps"]
            ],
        ),
        "## Kernel groups\n"
        + table(
            ["Group", "Modeled / aligned", "Real µs", "Cycles", "Sim µs", "Signed error %"],
            [
                [
                    r["group"],
                    f"{r['modeled_n']} / {r['n']}",
                    *[
                        f"{r[k]:.6f}" if r[k] is not None else "timeout / unavailable"
                        for k in ("real_us", "cycles", "sim_us", "signed_error_pct")
                    ],
                ]
                for r in rows
            ],
        ),
        "## Real mix\n"
        + table(
            ["Kind", "Group", "Kernels/step", "GPU µs/step"],
            [
                [r["kind"], r["group"], r["kernels_per_step"], f"{r['gpu_us_per_step']:.6f}"]
                for r in result["real_kernel_mix"]
            ],
        ),
        "## Steady requests\n"
        + table(
            [
                "Repeat",
                "Real TTFT ms",
                "Sim TTFT ms",
                "Real ITL ms",
                "Sim ITL ms",
                "Real E2E ms",
                "Sim E2E ms",
            ],
            [
                [
                    real["repeat"],
                    *[
                        f"{r[k]:.6f}"
                        for k in ("ttft_ms", "itl_ms_mean", "e2e_ms")
                        for r in (real, sim)
                    ],
                ]
                for real, sim in zip(result["real_requests"], result["sim_requests"], strict=True)
            ],
        ),
        "## Verification\n"
        + table(
            ["Representative", "Group", "Max absolute cycles", "Max relative %"],
            [
                [
                    r["representative"],
                    r["group"],
                    r["max_absolute_cycles"],
                    f"{r['max_relative'] * 100:.6f}",
                ]
                for r in result["verification"]
                if r["max_absolute_cycles"]
            ],
        ),
    ]
    (args.sim_dir / "tables.md").write_text("\n\n".join(sections) + "\n")
    print(f"Generated {len(rows)} group rows from native A100 artifacts")


if __name__ == "__main__":
    main()
