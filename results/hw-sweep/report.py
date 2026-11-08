"""Generate the inventory and per-kernel/step/request tables from saved sweep artifacts."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

HERE = Path(__file__).resolve().parent


def group(name, grid):
    if "reshape_and_cache_flash_kernel" in name:
        return "KV write"
    if "FlashAttnFwdSm90" in name:
        return "attention"
    if name.startswith("nvjet_"):
        groups = {
            9: "QKV GEMM",
            3: "out-proj",
            12: "MLP up GEMM",
            6: "MLP down GEMM",
            66: "lm_head",
        }
        if len(grid) != 3 or grid[1] not in groups:
            raise ValueError(f"unclassified GEMM: {name} {grid}")
        return groups[grid[1]]
    if "splitKreduce_kernel" in name:
        return "MLP down reduction"
    if "elementwise_kernel" in name:
        return "small elementwise"
    if "layer_norm_kernel" in name:
        return "layer norm"
    return "other setup/index/reduction"


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def inventory(gpgpu, accel):
    rows = []
    for path in sorted((gpgpu / "configs").rglob("gpgpusim.config")):
        text = path.read_text()
        options = {}
        for raw in text.splitlines():
            parts = raw.split("#", 1)[0].split()
            if len(parts) >= 2:
                options[parts[0]] = parts[1]
        cap = options.get("-gpgpu_ptx_force_max_capability")
        major = options.get("-gpgpu_compute_capability_major")
        minor = options.get("-gpgpu_compute_capability_minor")
        clock = options["-gpgpu_clock_domains"]
        rows.append(
            {
                "config": str(path.parent.relative_to(gpgpu / "configs")),
                "sm": f"{major}.{minor}" if major else f"{cap[:-1]}.{cap[-1]}",
                "sm_source": "compute_capability_major/minor"
                if major
                else "ptx_force_max_capability",
                "clock_domains": clock,
                "clock_mhz": float(clock.split(":")[0]),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "trace_config": (
                    accel / "gpu-simulator/configs/tested-cfgs" / path.parent.name / "trace.config"
                ).is_file(),
            }
        )
    (HERE / "inventory.json").write_text(json.dumps(rows, indent=2) + "\n")
    write_csv(HERE / "inventory.csv", rows)
    tested = sum("tested-cfgs/" in row["config"] for row in rows)
    print(f"Inventoried {len(rows)} configs ({tested} tested)")


def tables(source):
    original = json.loads((source / "corrected_timeline.json").read_text())
    alignment = json.loads((source / "trace_alignment.json").read_text())
    aligned = {(k["repeat"], k["step"], k["seq"]): k["tracer_uid"] for k in alignment["kernels"]}
    report = [
        "# Measured sweep tables",
        "",
        "Cycles and microseconds are simulated, not hardware measurements.",
        "",
    ]
    for config in ("SM90_H100", "SM90_H200"):
        dest = HERE / config
        durations = json.loads((dest / "sim_durations.json").read_text())
        replay = json.loads((dest / "simulated_timeline.json").read_text())
        summary = json.loads((dest / "summary.json").read_text())
        by_key = {(k["repeat"], k["step"], k["seq"]): k for k in durations["kernels"]}
        rows = []
        for step in original["steps"]:
            for kernel in step["kernels"]:
                key = step["repeat"], step["step"], kernel["seq"]
                if key not in aligned:
                    continue
                duration = by_key[key]
                rows.append(
                    {
                        "config": config,
                        "repeat": step["repeat"],
                        "step": step["step"],
                        "step_in_repeat": step["step_in_repeat"],
                        "kind": step["kind"],
                        "context_length": step["context_length"],
                        "seq": kernel["seq"],
                        "uid": aligned[key],
                        "group": group(kernel["name"], kernel["grid"]),
                        "name": kernel["name"],
                        "grid": str(kernel["grid"]),
                        "cycles": duration["sim_cycles"],
                        "clock_mhz": duration["clock_mhz"],
                        "us": duration["sim_cycles"] / duration["clock_mhz"],
                        "source": duration["source"],
                        "provenance": duration["sim"],
                    }
                )
        write_csv(dest / "per-kernel.csv", rows)
        groups = defaultdict(list)
        for row in rows:
            groups[row["step_in_repeat"], row["kind"], row["group"]].append(row)
        grouped = [
            {
                "step_in_repeat": key[0],
                "kind": key[1],
                "group": key[2],
                "occurrences": len(values),
                "mean_cycles": mean(v["cycles"] for v in values),
                "mean_us": mean(v["us"] for v in values),
                "total_us": sum(v["us"] for v in values),
            }
            for key, values in sorted(groups.items())
        ]
        write_csv(dest / "kernel-groups.csv", grouped)
        steps = [
            {
                "repeat": s["repeat"],
                "step": s["step"],
                "step_in_repeat": s["step_in_repeat"],
                "kind": s["kind"],
                "gpu_busy_us": s["gpu_busy_ns"] / 1000,
                "step_wall_ms": s["baseline_wall_ns"] / 1e6,
            }
            for s in replay["steps"]
            if not s["warmup"]
        ]
        write_csv(dest / "step-metrics.csv", steps)
        requests = [
            {
                "repeat": r["repeat"],
                "e2e_ms": r["e2e_ms"],
                "ttft_ms": r["ttft_ms"],
                "itl_mean_ms": r["itl_ms_mean"],
            }
            for r in replay["requests"]
            if not r["warmup"]
        ]
        write_csv(dest / "request-metrics.csv", requests)
        report.extend(
            [
                f"## {config}",
                "",
                f"Accel-Sim wall seconds: {summary['wall_s']:.6f}",
                f"Counts: `{summary['counts']}`",
                "",
                "### Prefill/local step 0",
                "",
                "| Group | N | Mean cycles/kernel | Mean µs/kernel | Total µs |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in grouped:
            if row["step_in_repeat"] in (0, 2):
                if row["step_in_repeat"] == 2 and row == next(
                    r for r in grouped if r["step_in_repeat"] == 2
                ):
                    report.extend(
                        [
                            "",
                            "### Decode/local step 2 (context 8)",
                            "",
                            "| Group | N | Mean cycles/kernel | Mean µs/kernel | Total µs |",
                            "|---|---:|---:|---:|---:|",
                        ]
                    )
                report.append(
                    f"| {row['group']} | {row['occurrences']} | {row['mean_cycles']:.2f} | "
                    f"{row['mean_us']:.6f} | {row['total_us']:.6f} |"
                )
        report.extend(
            [
                "",
                "| Step kind | N | GPU busy µs mean | p50 | min | max |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for kind in ("prefill", "decode"):
            values = [s["gpu_busy_us"] for s in steps if s["kind"] == kind]
            report.append(
                f"| {kind} | {len(values)} | {mean(values):.6f} | {median(values):.6f} | "
                f"{min(values):.6f} | {max(values):.6f} |"
            )
        report.extend(["", "| Repeat | E2E ms | TTFT ms | ITL mean ms |", "|---|---:|---:|---:|"])
        for row in requests:
            report.append(
                f"| {row['repeat']} | {row['e2e_ms']:.6f} | {row['ttft_ms']:.6f} | "
                f"{row['itl_mean_ms']:.6f} |"
            )
        report.append("")
    (HERE / "tables.md").write_text("\n".join(report).rstrip() + "\n")
    print("Wrote per-kernel, group, step and request tables for H100 and H200")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpgpu", type=Path)
    parser.add_argument("--accel", type=Path)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.gpgpu and args.accel:
        inventory(args.gpgpu, args.accel)
    if args.source:
        tables(args.source)
