from __future__ import annotations

import argparse
import json
from pathlib import Path

from cascade.sim.compare import compare, perfetto, table
from cascade.sim.importer import import_durations
from cascade.sim.timeline import load_durations, load_timeline, simulate


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def run_simulate(args: argparse.Namespace) -> None:
    durations = load_durations(
        None if args.measured_only else args.durations, args.subtract_launch_cycles
    )
    result = simulate(load_timeline(args.input), durations, args.host_model)
    result["summary"]["simulation"]["subtract_launch_cycles"] = args.subtract_launch_cycles
    write_json(args.output, result)
    coverage = result["summary"]["simulation"]["coverage"]
    print(f"Wrote {args.output}")
    print(
        f"Simulated kernels: {coverage['simulated_kernels']}/{coverage['kernels']} "
        f"({coverage['kernel_fraction']:.2%}); measured GPU-time coverage: "
        f"{coverage['gpu_time_fraction']:.2%}"
    )


def run_compare(args: argparse.Namespace) -> None:
    real, simulated = load_timeline(args.real), load_timeline(args.simulated)
    report = compare(real, simulated)
    write_json(args.output, report)
    if args.perfetto:
        # Normalize the real track with measured durations and corrected host gaps too.
        write_json(args.perfetto, perfetto(simulate(real, {}), simulated))
    print(table(report))
    print(f"Wrote {args.output}")


def run_import(args: argparse.Namespace) -> None:
    result = import_durations(args.stats, args.mapping, args.clock_mhz)
    write_json(args.output, result)
    print(f"Wrote {args.output}: {len(result['kernels'])} mapped kernels")


def add_commands(sub: argparse._SubParsersAction) -> None:
    command = sub.add_parser("simulate", help="reconstruct steps using corrected host time")
    command.add_argument("input", type=Path, help="corrected_timeline.json")
    source = command.add_mutually_exclusive_group()
    source.add_argument("--durations", type=Path, help="mapped simulated kernel durations JSON")
    source.add_argument("--measured-only", action="store_true", help="use only measured durations")
    command.add_argument("--subtract-launch-cycles", type=float, default=0)
    command.add_argument(
        "--host-model", choices=["scaled-gaps", "parametric"], default="scaled-gaps"
    )
    command.add_argument("--output", type=Path, default=Path("simulated_timeline.json"))
    command.set_defaults(func=run_simulate)
    command = sub.add_parser("compare", help="compare reconstructed against unprofiled step timing")
    command.add_argument("real", type=Path)
    command.add_argument("simulated", type=Path)
    command.add_argument("--output", type=Path, default=Path("compare.json"))
    command.add_argument("--perfetto", type=Path, help="optional two-process Chrome JSON")
    command.set_defaults(func=run_compare)
    command = sub.add_parser(
        "import-sim-durations", help="map Accel-Sim stats to corrected kernels"
    )
    command.add_argument("--stats", required=True, type=Path, help="run.log or kernel-stats.csv")
    command.add_argument(
        "--mapping", required=True, type=Path, help="UID/name/seq JSON mapping list"
    )
    command.add_argument("--clock-mhz", required=True, type=float)
    command.add_argument("--output", type=Path, default=Path("sim_durations.json"))
    command.set_defaults(func=run_import)
