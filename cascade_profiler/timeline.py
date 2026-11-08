import re

from cascade_profiler.nsys import STEP_RE
from cascade_profiler.planning import step_contexts

ITERATION_RE = re.compile(
    r"Iteration\((\d+)\): (\d+) context requests, (\d+) context tokens, "
    r"(\d+) generation requests, (\d+) generation tokens, iteration elapsed time: ([\d.]+) ms")


def _parse_iterations(log: str, step_log: list[dict]) -> tuple[list[dict], dict]:
    steps = {r["step"]: r for r in step_log}
    rows = []
    for match in ITERATION_RE.finditer(log):
        i, ctx, ctok, gen, gtok = map(int, match.groups()[:5])
        st = steps.get(i, {})
        start, end = st.get("wall_start_ns"), st.get("wall_end_ns")
        rows.append({"step": i, "repeat": st.get("repeat"), "warmup": st.get("warmup"),
                     "ctx_reqs": ctx, "ctx_tokens": ctok, "gen_reqs": gen, "gen_tokens": gtok,
                     "elapsed_ms": float(match[6]), "wall_start_ns": start, "wall_end_ns": end,
                     "wall_ns": end - start if start is not None and end is not None else None,
                     "kind": "prefill" if ctx and not gen else "decode" if gen and not ctx else "mixed"})
    ids = [r["step"] for r in rows]
    return rows, {"iteration_count": len(rows), "step_count": len(step_log),
                  "iteration_step_mismatch": len(rows) != len(step_log) or
                  set(ids) != set(steps) or len(ids) != len(set(ids))}


def _interval_union(intervals: list[tuple], start: int, end: int) -> int:
    total, cursor = 0, start
    for a, b in sorted(intervals):
        a, b = max(a, start, cursor), min(b, end)
        if b > a:
            total += b - a
            cursor = b
    return total


def _phase_nsys_stats(nvtx: list[dict], events: list[dict], apis: list[dict]) -> dict:
    out = {}
    for r in nvtx:
        m = STEP_RE.match(r["text"] or "")
        if not m or r["end"] is None:
            continue
        a, b = r["start"], r["end"]
        busy = _interval_union([(e["start_ns"], e["end_ns"]) for e in events
                                if e["event_type"] in {"kernel", "memcpy", "memset"}], a, b)
        api = _interval_union([(e["start_ns"], e["end_ns"]) for e in apis
                               if (e["pid"], e["tid"]) == (r["pid"], r["tid"])], a, b)
        out[int(m[1])] = {"gpu_busy_ns": busy, "cuda_api_ns": api,
                          "other_host_ns": b - a - api, "nsys_span_ns": b - a}
    return out


def _distribution(values: list) -> dict:
    import statistics

    v = sorted(values)
    def percentile(q):
        pos = (len(v) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(v) - 1)
        return v[lo] + (v[hi] - v[lo]) * (pos - lo)
    return {"n": len(v), "mean": statistics.fmean(v) if v else None,
            "p50": percentile(.5) if v else None, "p95": percentile(.95) if v else None}


def _reconcile(phases: dict, nsys_stats: dict) -> list[dict]:
    rows = []
    for kind in ("prefill", "decode"):
        row = {"kind": kind}
        for phase in ("baseline", "profiled"):
            selected = [r for r in phases.get(phase, [])
                        if r["warmup"] is False and r["kind"] == kind]
            for metric in ("wall_ns", "elapsed_ms", "gpu_busy_ns", "cuda_api_ns", "other_host_ns"):
                if phase == "baseline" and metric not in {"wall_ns", "elapsed_ms"}:
                    continue
                values = ([r[metric] for r in selected if r.get(metric) is not None]
                          if metric in {"wall_ns", "elapsed_ms"} else
                          [nsys_stats[r["step"]][metric] for r in selected if r["step"] in nsys_stats])
                for stat, value in _distribution(values).items():
                    row[f"{phase}_{metric}_{stat}"] = value
        for metric in ("wall_ns", "elapsed_ms"):
            for stat in ("mean", "p50"):
                a, b = (row[f"{p}_{metric}_{stat}"] for p in ("baseline", "profiled"))
                delta = b - a if a is not None and b is not None else None
                row[f"delta_{metric}_{stat}_ns"] = (delta * (1e6 if metric == "elapsed_ms" else 1)
                                                       if delta is not None else None)
                row[f"delta_{metric}_{stat}_pct"] = delta / a * 100 if delta is not None and a else None
        rows.append(row)
    return rows


def _corrected_timeline(meta: dict, baseline_steps: list[dict], profiled_steps: list[dict],
                        iterations: list[dict], metrics: dict, nsys_stats: dict,
                        nsys_steps: dict, kernels: list[dict], profiled_metrics: dict | None = None) -> dict:
    clock = metrics.get("clock", {})
    profiled_clock = (profiled_metrics or {}).get("clock", {})

    def timing(records, run_clock):
        previous, out = {}, {}
        offset = run_clock.get("wall_minus_monotonic_ns")
        for r in sorted(records, key=lambda r: r["step"]):
            start, end = r.get("monotonic_start_ns"), r.get("monotonic_end_ns")
            if offset is not None:
                if start is None and r.get("wall_start_ns") is not None:
                    start = r["wall_start_ns"] - offset
                if end is None and r.get("wall_end_ns") is not None:
                    end = r["wall_end_ns"] - offset
            prev = previous.get(r["repeat"])
            out[r["step"]] = (start, end, start - prev if start is not None and prev is not None else None)
            previous[r["repeat"]] = end
        return out

    baseline_times = timing(baseline_steps, clock)
    profiled_times = timing(profiled_steps, profiled_clock)
    def indexed(records):
        counts, out = {}, {}
        for record in sorted(records, key=lambda r: r["step"]):
            repeat = record["repeat"]
            pos = counts.get(repeat, 0)
            out[(repeat, pos)] = record
            counts[repeat] = pos + 1
        return out

    baseline, profiled = indexed(baseline_steps), indexed(profiled_steps)
    contexts = step_contexts(baseline_steps, iterations)
    iteration_by_step = {r["step"]: r for r in iterations}
    steps = []
    for key, record in baseline.items():
        it = iteration_by_step.get(record["step"], {})
        partner = profiled.get(key, {})
        pid = partner.get("step")
        stats, span = nsys_stats.get(pid), nsys_steps.get(pid)
        captured = stats is not None and span is not None
        wall = record["wall_end_ns"] - record["wall_start_ns"]
        busy = stats["gpu_busy_ns"] if captured else None
        step_kernels = []
        if captured:
            selected = sorted((k for k in kernels if span["start"] <= k["start"] < span["end"]),
                              key=lambda k: k["start"])
            for seq, k in enumerate(selected):
                step_kernels.append({"seq": seq, "name": k["mangled"],
                                     "grid": [int(x) for x in k["grid"].split("x")],
                                     "block": [int(x) for x in k["block"].split("x")],
                                     "duration_ns": k["end"] - k["start"],
                                     "start_offset_ns": k["start"] - span["start"]})
        steps.append({"step": record["step"], "repeat": record["repeat"], "step_in_repeat": key[1],
                      "context_length": contexts.get((record["repeat"], record["step"])),
                      "warmup": record["warmup"], "kind": it.get("kind", record.get("kind", "unknown")),
                      "num_reqs": it["ctx_reqs"] + it["gen_reqs"] if it else record.get("num_reqs"),
                      "num_scheduled_tokens": it["ctx_tokens"] + it["gen_tokens"] if it else
                      record.get("num_scheduled_tokens"), "baseline_wall_ns": wall,
                      "profiled_wall_ns": stats["nsys_span_ns"] if captured else None,
                      "gpu_busy_ns": busy, "cuda_api_ns": stats["cuda_api_ns"] if captured else None,
                      "host_ns": wall - busy if captured else None, "kernels": step_kernels})
        for phase_name, values in (("baseline", baseline_times[record["step"]]),
                                   ("profiled", profiled_times.get(pid, (None, None, None)))):
            for field, value in zip(("wall_start_ns", "wall_end_ns", "gap_before_ns"), values):
                steps[-1][f"{phase_name}_{field}"] = value
        for phase_name, source in (("baseline", record), ("profiled", partner)):
            for field in ("gc_before", "gc_after", "gc_pause_ns_in_step", "gc_pause_ns_in_gap_before"):
                steps[-1][f"{phase_name}_{field}"] = source.get(field)
    requests = []
    for record in metrics.get("requests", []):
        m = record.get("metrics") or {}
        def elapsed(end, start):
            return (m[end] - m[start]) * 1000 if end in m and start in m else None
        n = m.get("num_generation_tokens")
        decode_ms = elapsed("last_token_ts", "first_token_ts")
        repeat_steps = [s for s in steps if s["repeat"] == record["repeat"]]
        first_step = repeat_steps[0]["baseline_wall_start_ns"] if repeat_steps else None
        last_step = repeat_steps[-1]["baseline_wall_end_ns"] if repeat_steps else None
        offset = clock.get("wall_minus_monotonic_ns")
        arrival = round(m["arrival_time"] * 1e9) - offset if m.get("arrival_time") is not None and offset is not None else None
        finish = record.get("generate_return_monotonic_ns")
        queued = round(m["queued_ts"] * 1e9) if "queued_ts" in m else None
        last_token = round(m["last_token_ts"] * 1e9) if "last_token_ts" in m else None
        requests.append({"repeat": record["repeat"], "warmup": record["repeat"] == 0,
                         "queued_ms": elapsed("scheduled_ts", "queued_ts"),
                         "ttft_ms": elapsed("first_token_ts", "queued_ts"),
                         "itl_ms_mean": decode_ms / (n - 1) if n is not None and n > 1
                         and decode_ms is not None else None,
                         "itl_ms_p50": None, "e2e_ms": elapsed("last_token_ts", "queued_ts"),
                         "num_output_tokens": n,
                         "first_step_start_ns": first_step, "last_step_end_ns": last_step,
                         "arrival_ns": arrival,
                         "queued_ns": queued,
                         "first_token_ns": round(m["first_token_ts"] * 1e9) if "first_token_ts" in m else None,
                         "last_token_ns": last_token,
                         "pre_first_step_ns": first_step - queued if first_step is not None and queued is not None else None,
                         "post_last_step_ns": last_token - last_step if last_token is not None and last_step is not None else None,
                         "arrival_to_first_step_ns": first_step - arrival if first_step is not None and arrival is not None else None,
                         "last_step_end_to_return_ns": finish - last_step if finish is not None and last_step is not None else None})
    summary = {}
    for kind in sorted({s["kind"] for s in steps}):
        steady = [s for s in steps if s["kind"] == kind and not s["warmup"]]
        matched = [s for s in steady if s["profiled_wall_ns"] is not None]
        wall = _distribution([s["baseline_wall_ns"] for s in steady])
        a = _distribution([s["baseline_wall_ns"] for s in matched])["p50"]
        b = _distribution([s["profiled_wall_ns"] for s in matched])["p50"]
        summary[kind] = {"n": len(steady),
                         "baseline_wall_ns": {k: wall[k] for k in ("mean", "p50", "p95")},
                         "profiler_overhead_pct": (b - a) / a * 100 if a else None}
        for metric in ("gpu_busy_ns", "host_ns"):
            dist = _distribution([s[metric] for s in matched])
            summary[kind][metric] = {k: dist[k] for k in ("mean", "p50")}
        for metric in ("baseline_gap_before_ns", "profiled_gap_before_ns"):
            dist = _distribution([s[metric] for s in steady if s[metric] is not None])
            summary[kind][metric] = {k: dist[k] for k in ("mean", "p50", "p95")}
    return {"meta": {k: meta[k] for k in (
        "run_name", "model", "prompt", "max_tokens", "repeats", "exec_mode", "gpu_name",
        "sm_clock_mhz_median", "mem_clock_mhz", "vllm_version", "layerwise_nvtx")} | {
            **({"model_config": meta["model_config"]} if "model_config" in meta else {}),
            "clock": {"step_clock": "monotonic", "wall_minus_monotonic_ns": None,
                      "how": "unavailable in legacy metrics", **clock,
                      "profiled_wall_minus_monotonic_ns": profiled_clock.get("wall_minus_monotonic_ns")}},
        "steps": steps, "requests": requests, "summary": summary,
        "gc_summary": {"baseline": metrics.get("gc", {}).get("repeats", []),
                       "profiled": (profiled_metrics or {}).get("gc", {}).get("repeats", [])}}
