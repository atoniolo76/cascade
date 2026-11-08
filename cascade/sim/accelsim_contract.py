"""Read the profiler run-directory contract without importing the GPU package."""

import csv
from pathlib import Path


def kernel_key(kernel: dict) -> tuple:
    def dims(value):
        return tuple(map(int, value.split("x") if isinstance(value, str) else value))

    return (
        kernel.get("mangled_name") or kernel["name"],
        dims(kernel["grid"]),
        dims(kernel["block"]),
    )


def step_contexts(steps: list[dict]) -> dict:
    prompt, decode, result, counts, seen = {}, {}, {}, {}, set()
    for step in sorted(steps, key=lambda s: s["step"]):
        repeat = step["repeat"]
        local = step.get("step_in_repeat", counts.get(repeat, 0))
        if (repeat, local) in seen:
            raise ValueError(f"duplicate repeat-local step {(repeat, local)}")
        seen.add((repeat, local))
        counts[repeat] = local + 1
        if step.get("kind") == "prefill":
            count = step.get("num_prefill_reqs", step.get("num_reqs"))
            tokens = step.get("num_scheduled_tokens")
            per_request = list(step.get("tokens_per_req", {}).values())
            homogeneous = count == 1 or (per_request and len(set(per_request)) == 1)
            if count and tokens and tokens % count == 0 and homogeneous:
                prompt[repeat], decode[repeat] = tokens // count, 0
        elif step.get("kind") == "decode":
            decode[repeat] = decode.get(repeat, 0) + 1
        context = step.get("context_length")
        if context is None and repeat in prompt:
            context = prompt[repeat] + decode[repeat]
        if context is not None:
            result[repeat, step["step"]] = context
    return result


def post_process_command(accelsim_root: str, traces_dir: Path) -> list[str]:
    return [
        str(
            Path(accelsim_root)
            / "util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing"
        ),
        str(traces_dir),
        "-j",
        "8",
    ]


def _write_csv(path: Path, rows: list[dict], cols: list[str]):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
