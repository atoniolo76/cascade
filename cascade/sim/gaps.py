import math
from statistics import mean, median, quantiles


def number(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{label} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def pre_step(request):
    if "pre_first_step_ns" in request:
        return number(request["pre_first_step_ns"], "pre_first_step_ns")
    return number(request.get("queued_ms", 0), "queued_ms") * 1e6


def grouped_steps(data):
    groups = {}
    for step in data["steps"]:
        if not step["warmup"]:
            groups.setdefault(step["repeat"], []).append(step)
    return groups


def validate_gaps(data):
    for steps in grouped_steps(data).values():
        if not any("baseline_gap_before_ns" in s for s in steps):
            continue
        for i, step in enumerate(steps):
            if "baseline_gap_before_ns" not in step:
                raise ValueError("gap-aware repeats require baseline_gap_before_ns on every step")
            gap = step["baseline_gap_before_ns"]
            if i == 0:
                if gap is not None:
                    raise ValueError("first step gap must be null")
            else:
                number(gap, "baseline_gap_before_ns")
            if "profiled_gap_before_ns" in step:
                profile_gap = step["profiled_gap_before_ns"]
                if i == 0 and profile_gap is not None:
                    raise ValueError("first profiled step gap must be null")
                if i:
                    number(profile_gap, "profiled_gap_before_ns")
            for prefix in ("baseline", "profiled"):
                start, end = (step.get(f"{prefix}_wall_{bound}_ns") for bound in ("start", "end"))
                if start is None and end is None:
                    continue
                start = number(start, f"{prefix}_wall_start_ns")
                end = number(end, f"{prefix}_wall_end_ns")
                if end < start:
                    raise ValueError(f"{prefix} timestamps must be ordered")
                if i and prefix == "baseline":
                    previous_end = steps[i - 1].get("baseline_wall_end_ns")
                    if previous_end is not None and abs(start - previous_end - gap) > 1:
                        raise ValueError("baseline timestamps must match gap_before_ns")
    for request in data["requests"]:
        if request["warmup"]:
            continue
        for field in ("arrival_ns", "pre_first_step_ns", "post_last_step_ns"):
            if field in request:
                number(request[field], field)


def fit_gaps(data, model):
    fits = {}
    for kind in ("prefill", "decode"):
        samples = [
            s
            for steps in grouped_steps(data).values()
            for s in steps[1:]
            if s["kind"] == kind and "baseline_gap_before_ns" in s
        ]
        if not samples:
            continue
        xs = [number(s["num_reqs"], "num_reqs") for s in samples]
        ys = [number(s["baseline_gap_before_ns"], "baseline_gap_before_ns") for s in samples]
        xbar, ybar = mean(xs), mean(ys)
        variance = sum((x - xbar) ** 2 for x in xs)
        b = (
            sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys, strict=True)) / variance
            if model == "linear" and variance
            else 0
        )
        fits[kind] = {
            "a_ns": ybar - b * xbar,
            "b_ns_per_req": b,
            "samples": len(samples),
            "slope_identifiable": model == "linear" and bool(variance),
        }
    return fits


def apply_gaps(data, model):
    if model not in ("measured", "mean", "linear"):
        raise ValueError("unknown gap model")
    fits = fit_gaps(data, model) if model != "measured" else {}
    active = False
    requests = {r["repeat"]: r for r in data["requests"] if not r["warmup"]}
    for repeat, steps in grouped_steps(data).items():
        if "baseline_gap_before_ns" not in steps[0]:
            continue
        active = True
        request = requests.get(repeat)
        pre = pre_step(request) if request else 0
        arrival = number(request.get("arrival_ns", 0), "arrival_ns") if request else 0
        cursor = arrival + pre if request else (steps[0].get("baseline_wall_start_ns") or 0)
        for i, step in enumerate(steps):
            gap = step["baseline_gap_before_ns"]
            if i:
                if fits:
                    fit = fits[step["kind"]]
                    gap = max(0, fit["a_ns"] + fit["b_ns_per_req"] * step["num_reqs"])
                cursor += gap
            step["baseline_gap_before_ns"] = gap
            step["baseline_wall_start_ns"] = cursor
            cursor += step["baseline_wall_ns"]
            step["baseline_wall_end_ns"] = cursor
            for field, value in (
                ("profiled_gap_before_ns", gap),
                ("profiled_wall_start_ns", step["baseline_wall_start_ns"]),
                ("profiled_wall_end_ns", cursor),
            ):
                if field in step:
                    step[field] = value
    return {"gap_model": model, "gap_fit": fits} if active else {}


def derive_gap_request(request, steps):
    arrival = number(request.get("arrival_ns", 0), "arrival_ns")
    pre = pre_step(request)
    post = number(request.get("post_last_step_ns", 0), "post_last_step_ns")
    prefills = [s for s in steps if s["kind"] == "prefill"]
    first_token = prefills[-1]["baseline_wall_end_ns"]
    if any(s["kind"] == "decode" for s in steps[: steps.index(prefills[-1])]):
        raise ValueError("prefill steps must precede decode steps")
    decode = [
        (s["baseline_wall_ns"] + (s["baseline_gap_before_ns"] or 0)) / 1e6
        for s in steps
        if s["kind"] == "decode"
    ]
    last_end = steps[-1]["baseline_wall_end_ns"]
    request.update(
        arrival_ns=arrival,
        pre_first_step_ns=pre,
        post_last_step_ns=post,
        first_step_start_ns=steps[0]["baseline_wall_start_ns"],
        last_step_end_ns=last_end,
        first_token_ns=first_token,
        last_token_ns=last_end + post,
        ttft_ms=(first_token - arrival) / 1e6,
        itl_ms_mean=mean(decode) if decode else None,
        itl_ms_p50=median(decode) if decode else None,
        e2e_ms=(last_end + post - arrival) / 1e6,
    )


def gap_stats(data):
    stats = {}
    for kind in ("prefill", "decode"):
        values = [
            s["baseline_gap_before_ns"]
            for steps in grouped_steps(data).values()
            for s in steps[1:]
            if s["kind"] == kind and "baseline_gap_before_ns" in s
        ]
        if values:
            stats[kind] = {
                "n": len(values),
                "mean": mean(values),
                "p50": median(values),
                "p95": quantiles(values, n=100, method="inclusive")[94]
                if len(values) > 1
                else values[0],
            }
    return stats
