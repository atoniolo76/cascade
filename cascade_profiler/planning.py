"""Baseline-derived context lengths and repeat-local tracing plans; no tokenizer dependency."""

from cascade_profiler.align import indexed_steps


def step_contexts(steps: list[dict], iterations: list[dict] | None = None) -> dict:
    logs = {s["step"]: s for s in (iterations or [])}
    prompt, decode, result = {}, {}, {}
    for (repeat, _), step in indexed_steps(steps).items():
        log = logs.get(step["step"], step)
        kind = log.get("kind", step.get("kind"))
        if kind == "prefill":
            count = log.get("ctx_reqs", step.get("num_prefill_reqs", step.get("num_reqs")))
            tokens = log.get("ctx_tokens", step.get("num_scheduled_tokens"))
            per_request = list(step.get("tokens_per_req", {}).values())
            homogeneous = count == 1 or (per_request and len(set(per_request)) == 1)
            if count and tokens and tokens % count == 0 and homogeneous:
                prompt[repeat], decode[repeat] = tokens // count, 0
        elif kind == "decode":
            decode[repeat] = decode.get(repeat, 0) + 1
        context = step.get("context_length")
        if context is None and repeat in prompt:
            context = prompt[repeat] + decode[repeat]
        if context is not None:
            result[repeat, step["step"]] = context
    return result


def trace_plan(workload, context_lengths=(1, 8, 16, 32), *, baseline_steps=None, iterations=None,
               repeat=1) -> list[tuple[int, int]]:
    """Select prefill + existing decode contexts using measured token counts, not prompt text."""
    if baseline_steps is None:
        raise ValueError("trace_plan requires baseline_steps (and iteration logs when available)")
    if not 0 <= repeat < workload.repeats:
        raise ValueError("trace plan repeat must be in [0, repeats)")
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in context_lengths):
        raise ValueError("context lengths must be positive integers")
    indexed = indexed_steps(baseline_steps)
    selected = {key: s for key, s in indexed.items() if key[0] == repeat}
    logs = {s["step"]: s for s in (iterations or [])}
    prefills = [key for key, step in selected.items() if logs.get(step["step"], step).get("kind") == "prefill"]
    if len(prefills) != 1 or any(logs.get(s["step"], s).get("kind") == "mixed" for s in selected.values()):
        raise ValueError("trace planning requires one non-chunked, homogeneous prefill per repeat")
    contexts = step_contexts(baseline_steps, iterations)
    if (repeat, selected[prefills[0]]["step"]) not in contexts:
        raise ValueError("baseline has no usable prompt token count")
    return [key for key, step in selected.items() if key == prefills[0] or
            (logs.get(step["step"], step).get("kind") == "decode" and
             contexts.get((repeat, step["step"])) in context_lengths)]
