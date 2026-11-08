from __future__ import annotations

import json
import shlex
from pathlib import Path

from cascade.sim.compare import compare, table
from cascade.sim.timeline import load_durations, load_timeline, numeric, simulate


def write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def request_sanity(real: dict, simulated: dict, report: dict, target: float = 1) -> dict:
    numeric(target, "sanity target", positive=True)
    requests = {r["repeat"]: r for r in real["requests"] if not r["warmup"]}
    predicted = {r["repeat"]: r for r in simulated["requests"] if not r["warmup"]}
    rows = []
    for metrics in report["requests"]:
        repeat = metrics["repeat"]
        request, prediction = requests[repeat], predicted[repeat]
        steps = [s for s in real["steps"] if not s["warmup"] and s["repeat"] == repeat]
        prefill = [s for s in steps if s["kind"] == "prefill"][-1]
        end, token = prefill.get("baseline_wall_end_ns"), request.get("first_token_ns")
        modeled = prediction.get("first_token_ns")

        def value(field, key, record=metrics):
            return record[field][key] if record[field] else None

        rows.append(
            dict(
                repeat=repeat,
                e2e_error_ms=value("e2e_ms", "error"),
                e2e_error_pct=value("e2e_ms", "error_pct"),
                itl_mean_error_pct=value("itl_ms_mean", "error_pct"),
                ttft_error_ms=value("ttft_ms", "error"),
                ttft_error_pct=value("ttft_ms", "error_pct"),
                prefill_end_minus_first_token_ms=(end - token) / 1e6
                if end is not None and token is not None
                else None,
                prefill_scalar_span_discrepancy_ms=(modeled - end) / 1e6
                if modeled is not None and end is not None
                else None,
                total_measured_gap_ms=sum(s.get("baseline_gap_before_ns") or 0 for s in steps)
                / 1e6,
            )
        )
    passes = {
        field: bool(rows)
        and all(
            r[field + "_error_pct"] is not None and abs(r[field + "_error_pct"]) <= target
            for r in rows
        )
        for field in ("e2e", "itl_mean", "ttft")
    }
    e2e_itl = passes["e2e"] and passes["itl_mean"]
    outcome = f"E2e and mean ITL {'pass' if e2e_itl else 'do not both pass'} {target:g}%. "
    in_step = bool(rows) and all(
        r["prefill_end_minus_first_token_ms"] is not None
        and r["prefill_end_minus_first_token_ms"] > 0
        for r in rows
    )
    cause = (
        "TTFT retains the modeled prefill end; observed first token is stamped inside prefill. "
        if in_step
        else "TTFT retains the modeled prefill end; inspect per-request errors. "
    )
    return dict(
        target_abs_error_pct=target,
        e2e_pass=passes["e2e"],
        itl_mean_pass=passes["itl_mean"],
        ttft_pass=passes["ttft"],
        e2e_itl_pass=e2e_itl,
        request_sanity_pass=all(passes.values()),
        rows=rows,
        explanation="Queued-reference reconstruction with signed last-token tail. "
        + outcome
        + cause
        + "No stamp correction, threshold tuning, or formula change.",
    )


def run_pipeline(
    input_path: Path, durations_path: Path, output: Path, subtract_cycles: float = 0
) -> dict:
    real = load_timeline(input_path)
    durations = load_durations(durations_path, subtract_cycles)
    output.mkdir(parents=True, exist_ok=True)
    for source, filename in (
        (input_path, "corrected_timeline.json"),
        (durations_path, "sim_durations.json"),
    ):
        if source.resolve() != (output / filename).resolve():
            (output / filename).write_bytes(source.read_bytes())
    sections, logs, sanity = [], [], {}
    for mode in ("measured-only", "accelsim"):
        for model in ("measured", "mean", "linear"):
            result = simulate(real, {} if mode == "measured-only" else durations, gap_model=model)
            result["summary"]["simulation"]["subtract_launch_cycles"] = (
                0 if mode == "measured-only" else subtract_cycles
            )
            report = compare(real, result)
            destination = output / mode / model
            write_json(destination / "simulated_timeline.json", result)
            write_json(destination / "compare.json", report)
            printed = table(report)
            sections.append(f"### {mode} / {model}\n\n```text\n{printed}\n```\n")
            logs.append(f"{mode} / {model}\n{printed}\n")
            if mode == "measured-only" and model == "measured":
                sanity = request_sanity(real, result, report)
    write_json(output / "sanity.json", sanity)
    command = shlex.join(
        [
            "cascade",
            "simulate-batch",
            str(input_path),
            "--durations",
            str(durations_path),
            "--output",
            str(output),
            "--launch-latency-cycles",
            str(subtract_cycles),
        ]
    )
    readme = (
        f"# Six-configuration replay: {real['meta'].get('run_name', 'unnamed')}\n\n"
        f"```bash\n{command}\n```\n\n"
        f"Launch-cycle subtraction: {subtract_cycles}; corrected host budgets are preserved.\n\n"
        f"Measured-only measured-gap e2e/ITL pass: {sanity['e2e_itl_pass']}; "
        f"TTFT pass: {sanity['ttft_pass']} (target {sanity['target_abs_error_pct']}%).\n"
        f"{sanity['explanation']}\n\n"
        "See sanity.json for each request's percentages and stamp offsets. "
        "Null/unavailable percentages do not count as passing. "
        "Mean/linear are in-sample fits, not scheduler predictions. "
        "The duration input's provenance records capture identity/transfer assumptions.\n\n"
        "Request errors are signed simulated-minus-reference; step/GPU errors are absolute.\n\n"
    )
    (output / "README.md").write_text(readme + "\n".join(sections))
    (output / "run.log").write_text("\n".join(logs))
    return sanity
