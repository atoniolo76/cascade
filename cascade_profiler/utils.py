import json
import secrets
import subprocess
import sys
import threading
import time
from pathlib import Path


def _unique_run_name() -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"


def _workload(meta: dict, exec_mode: str | None = None, script: str | None = None) -> list[str]:
    """vllm_trace.py command line for a run's parameters (metadata.json).

    Keys missing from older metadata fall back to the original behaviour (layer scope, eager,
    vLLM's default async scheduling), so old trace runs are re-profiled with identical settings.
    """
    cmd = [sys.executable, script] if script else [sys.executable, "-m", "cascade_profiler.runner"]
    cmd += [
        "--model", meta["model"],
        "--layers", ",".join(meta.get("layers") or []),
        "--prompts", json.dumps(meta["prompts"]),
        "--max-tokens", str(meta["max_tokens"]),
        "--max-model-len", str(meta["max_model_len"]),
        "--trace-scope", meta.get("trace_scope", "layers"),
        "--trace-steps", ",".join(map(str, meta.get("trace_steps") or [])),
        "--exec-mode", exec_mode or meta.get("exec_mode", "eager"),
    ]
    if meta.get("async_scheduling") is False:
        cmd.append("--no-async-scheduling")
    return cmd


def _run_logged(name: str, cmd: list[str], env: dict, cwd: Path, deadline: float):
    """Run cmd, streaming output to stdout and <cwd>/<name>.log; kill it at `deadline`."""
    print(f"=== [{name}] Running: {' '.join(cmd)}", flush=True)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RuntimeError(f"[{name}] no time left before the function timeout")
    with open(cwd / f"{name}.log", "w") as log:
        proc = subprocess.Popen(
            cmd, env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        # Kill the workload before Modal's function timeout so uploads can still drain.
        killer = threading.Timer(remaining, proc.kill)
        killer.start()
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            proc.wait()
        finally:
            killer.cancel()
    if proc.returncode != 0:
        raise RuntimeError(f"[{name}] failed with exit code {proc.returncode}")


def _write_csv(path: Path, rows: list[dict], cols: list[str] | None = None):
    import csv
    cols = cols or (list(rows[0]) if rows else [])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
