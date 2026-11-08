from __future__ import annotations

import argparse
import json
from pathlib import Path

from cascade.sim.compare import compare, perfetto, table
from cascade.sim.importer import import_durations
from cascade.sim.mapping import build_mapping
from cascade.sim.pipeline import run_pipeline, write_json
from cascade.sim.runner import config_clock, prepare_run
from cascade.sim.timeline import load_durations, load_timeline, simulate


def run_simulate(args: argparse.Namespace) -> None:
    durations = load_durations(
        None if args.measured_only else args.durations, args.subtract_launch_cycles
    )
    result = simulate(load_timeline(args.input), durations, args.host_model, args.gap_model)
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
    clock = config_clock(args.config) if args.config else args.clock_mhz
    if args.kernelslist:
        if not args.timeline:
            raise ValueError("--timeline is required with --kernelslist")
        generated = build_mapping(
            args.stats,
            args.kernelslist,
            args.timeline,
            clock,
            identity_path=args.identity,
            phases_path=args.source_phases,
            allow_shape_transfer=args.allow_shape_transfer,
        )
        result = generated["durations"]
        write_json(args.mapping_output, generated["mapping"])
        coverage = generated["coverage"]
        print(
            f"Coverage: {coverage['simulated_kernels']}/{coverage['kernels']} kernels; "
            f"{coverage['gpu_time_fraction']:.2%} measured GPU time"
        )
    else:
        result = import_durations(args.stats, args.mapping, clock)
    write_json(args.output, result)
    print(f"Wrote {args.output}: {len(result['kernels'])} mapped kernels")


def run_batch(args: argparse.Namespace) -> None:
    sanity = run_pipeline(args.input, args.durations, args.output, args.launch_latency_cycles)
    print(json.dumps(sanity, indent=2))
    print(f"Wrote six configurations to {args.output}")


def run_prepare(args: argparse.Namespace) -> None:
    plan = prepare_run(args.framework, args.config_name, args.kernelslist, args.run_dir)
    if args.output:
        write_json(args.output, plan)
    print(json.dumps(plan, indent=2))


def add_commands(sub: argparse._SubParsersAction) -> None:
    command = sub.add_parser("simulate", help="reconstruct steps using corrected host time")
    command.add_argument("input", type=Path, help="corrected_timeline.json")
    source = command.add_mutually_exclusive_group()
    source.add_argument("--durations", type=Path, help="mapped simulated kernel durations JSON")
    source.add_argument("--measured-only", action="store_true", help="use only measured durations")
    command.add_argument(
        "--launch-latency-cycles",
        "--subtract-launch-cycles",
        dest="subtract_launch_cycles",
        type=float,
        default=0,
        help="subtract configured launch latency from each simulated kernel",
    )
    command.add_argument("--gap-model", choices=["measured", "mean", "linear"], default="measured")
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
    source = command.add_mutually_exclusive_group(required=True)
    source.add_argument("--mapping", type=Path, help="UID/name/seq JSON mapping list")
    source.add_argument("--kernelslist", type=Path, help="generate mapping from trace headers")
    command.add_argument("--timeline", type=Path, help="corrected_timeline.json for generation")
    command.add_argument("--identity", type=Path, help="same-capture UID/name/repeat/step/seq list")
    command.add_argument("--source-phases", type=Path, help="explicit UID/kind phase annotations")
    command.add_argument("--allow-shape-transfer", action="store_true")
    command.add_argument("--mapping-output", type=Path, default=Path("kernel_mapping.json"))
    clock = command.add_mutually_exclusive_group(required=True)
    clock.add_argument("--clock-mhz", type=float)
    clock.add_argument("--config", type=Path, help="gpgpusim.config (reads core clock)")
    command.add_argument("--output", type=Path, default=Path("sim_durations.json"))
    command.set_defaults(func=run_import)
    command = sub.add_parser("prepare-sim-run", help="prepare configs and return worker argv/cwd")
    command.add_argument("--framework", required=True, type=Path)
    command.add_argument("--config-name", required=True)
    command.add_argument("--kernelslist", required=True, type=Path)
    command.add_argument("--run-dir", required=True, type=Path)
    command.add_argument("--output", type=Path)
    command.set_defaults(func=run_prepare)
    command = sub.add_parser("simulate-batch", help="write all six timing/gap configurations")
    command.add_argument("input", type=Path)
    command.add_argument("--durations", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    command.add_argument("--launch-latency-cycles", type=float, default=0)
    command.set_defaults(func=run_batch)
