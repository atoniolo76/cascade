"""Single-request sizing from simulated kernel floors and an approximate dense roofline.

All times are nanoseconds internally. This is not SASS replay or a batching scheduler.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import isfinite, prod
from statistics import mean, median

GROUPS = ("linear", "attention", "kv_write", "lm_head", "sampler", "elementwise")
COMPUTE_FRACTION = 0.7
MEMORY_FRACTION = 0.8
LINEAR_ACTIVATION_PASSES = 8
ELEMENTWISE_PASSES = 16


@dataclass(frozen=True)
class GPUSpec:
    tflops: float
    bandwidth_tb_s: float
    memory_gb: int


# NVIDIA SXM datasheets: dense BF16/FP16 tensor throughput, HBM bandwidth and capacity.
# https://www.nvidia.com/en-us/data-center/{h100,h200,a100}/
GPU_SPECS = {
    "SM90_H100": GPUSpec(989, 3.35, 80),
    "SM90_H200": GPUSpec(989, 4.8, 141),
    "SM80_A100": GPUSpec(312, 2.039, 80),
}


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int
    ffn_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    dtype_bytes: int

    @property
    def non_embedding_params(self) -> int:
        h = self.hidden_size
        return self.num_layers * (
            2 * h * h + 2 * h * self.num_kv_heads * self.head_dim + 2 * h * self.ffn_size
        )

    @property
    def kv_bytes_per_token(self) -> int:
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * self.dtype_bytes


@dataclass(frozen=True)
class PerfProfile:
    gpu_config: str
    model_config: ModelConfig
    prompt_tokens: int
    output_tokens: int
    max_model_len: int
    prefill_gpu_ns: dict[str, float]
    decode_gpu_ns: dict[int, dict[str, float]]
    host_ns: dict[str, float]
    residual_gpu_ns: dict[str, float]
    gap_ns: dict[str, float]
    pre_first_step_ns: float
    post_last_step_ns: float
    calibration_range: dict
    assumptions: tuple[str, ...]
    gpu_memory_utilization: float = 0.8


@dataclass(frozen=True)
class StepGPUTime:
    groups_ns: dict[str, float]
    residual_ns: float

    @property
    def total_ns(self) -> float:
        return sum(self.groups_ns.values()) + self.residual_ns


@dataclass(frozen=True)
class RequestStep:
    kind: str
    tokens: int
    context: int
    groups_ns: dict[str, float]
    residual_gpu_ns: float
    gpu_ns: float
    host_ns: float
    gap_before_ns: float
    wall_ns: float
    start_ns: float
    end_ns: float
    provenance: str
    host_provenance: str = "assumed-host"


@dataclass(frozen=True)
class RequestResult:
    prompt_tokens: int
    output_tokens: int
    ttft_ms: float | None
    itl_ms_mean: float | None
    itl_ms_p50: float | None
    e2e_ms: float
    steps: list[RequestStep]
    calibration_range: dict
    assumptions: tuple[str, ...]
    provenance_counts: dict[str, int]


def _positive_int(value, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _time(value, name: str, *, signed=False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite time")
    if not isfinite(value) or (value < 0 and not signed):
        raise ValueError(f"invalid {name}")
    return float(value)


def classify_kernel(name: str) -> str:
    name = name.lower()
    if "reshape_and_cache" in name or "kv_write" in name or "cache_kv" in name:
        return "kv_write"
    if any(s in name for s in ("flash", "attn", "attention", "paged")):
        return "attention"
    if "lm_head" in name or "lmhead" in name:
        return "lm_head"
    if any(s in name for s in ("sampl", "topk", "top_k", "multinomial", "argmax", "softmax")):
        return "sampler"
    if "reduce" in name:
        return "elementwise"
    if any(s in name for s in ("gemm", "gemv", "cutlass", "xmma", "cublas", "nvjet", "matmul")):
        return "linear"
    if any(s in name for s in ("sm80", "sm90")) and any(s in name for s in ("mma", "tensorop")):
        return "linear"
    return "elementwise"


def kernel_groups(kernels: list[dict]) -> list[str]:
    def grid_size(kernel):
        grid = kernel.get("grid", (1,))
        if isinstance(grid, str):
            grid = tuple(map(int, grid.split("x")))
        return prod(grid)

    groups = [classify_kernel(k["name"]) for k in kernels]
    linear = [i for i, group in enumerate(groups) if group == "linear"]
    if linear and "lm_head" not in groups:
        last = linear[-1]
        # The vocab projection is the final linear launch, with a substantially larger grid.
        grids = [grid_size(kernels[i]) for i in linear]
        if grids[-1] >= 2 * median(grids):
            groups[last] = "lm_head"
    return groups


def _model(meta: dict, assumptions: list[str]) -> ModelConfig:
    config = meta.get("model_config", {})
    if config.get("tensor_parallel_size", 1) != 1 or config.get("quantization"):
        raise ValueError("request sizing requires an unquantized, single-GPU model")
    h = _positive_int(config.get("hidden_size"), "hidden_size")
    layers = _positive_int(config.get("num_layers", config.get("num_hidden_layers")), "num_layers")
    heads = _positive_int(config.get("num_heads", config.get("num_attention_heads")), "num_heads")
    ffn = config.get("ffn_size", config.get("intermediate_size", config.get("ffn_dim")))
    if ffn is None:
        ffn = 4 * h
        assumptions.append("Missing FFN size: assumed 4 * hidden_size.")
    vocab = config.get("vocab_size")
    if vocab is None and (
        config.get("architecture") == "OPTForCausalLM"
        or meta.get("model", "").startswith("facebook/opt-")
    ):
        vocab = 50272
        assumptions.append("Missing OPT vocabulary size: assumed 50272 (OPT model family).")
    dtype = config.get("dtype", "float16").removeprefix("torch.")
    widths = {"float16": 2, "bfloat16": 2, "float32": 4, "float64": 8}
    if dtype not in widths:
        raise ValueError(f"unsupported model dtype: {dtype}")
    assumptions.append(f"Weight/activation byte width: {widths[dtype]} ({dtype}).")
    return ModelConfig(
        h,
        _positive_int(ffn, "ffn_size"),
        layers,
        heads,
        _positive_int(config.get("num_kv_heads", heads), "num_kv_heads"),
        _positive_int(config.get("head_dim", h // heads), "head_dim"),
        _positive_int(vocab, "vocab_size"),
        widths[dtype],
    )


def model_limit(real: dict) -> int:
    meta = real.get("meta", {})
    for source in (meta, meta.get("run", {}), real.get("run", {}), meta.get("workload", {})):
        for config in (
            source,
            source.get("effective_llm_kwargs", {}),
            source.get("llm_kwargs", {}),
        ):
            if "max_model_len" in config:
                return _positive_int(config["max_model_len"], "max_model_len")
    config = meta.get("model_config", {})
    return _positive_int(
        config.get("max_model_len", config.get("max_position_embeddings", 2048)), "max_model_len"
    )


def _memory_utilization(real: dict) -> float:
    meta = real.get("meta", {})
    for source in (meta, meta.get("run", {}), real.get("run", {}), meta.get("workload", {})):
        for config in (
            source,
            source.get("effective_llm_kwargs", {}),
            source.get("llm_kwargs", {}),
        ):
            if "gpu_memory_utilization" in config:
                value = config["gpu_memory_utilization"]
                if isinstance(value, bool) or not isinstance(value, (float, int)):
                    raise ValueError("invalid gpu_memory_utilization")
                if not 0 < value <= 1:
                    raise ValueError("gpu_memory_utilization must be in (0, 1]")
                return float(value)
    return 0.8


def build_profile(real_timeline: dict, simulated_timeline: dict, gpu_config: str) -> PerfProfile:
    if gpu_config not in GPU_SPECS:
        raise ValueError(f"unknown GPU config: {gpu_config}")
    assumptions = [
        f"Achievable compute fraction={COMPUTE_FRACTION}; memory fraction={MEMORY_FRACTION}.",
        "Dense, ungated two-projection FFN; biases/norm parameters omitted; "
        "roofline omits KV capacity.",
        f"Linear activation passes={LINEAR_ACTIVATION_PASSES}; "
        f"elementwise memory passes={ELEMENTWISE_PASSES}.",
        "Host/gaps/envelopes and unattributed GPU busy are kind/repeat medians, held constant.",
        "Signed post-step token envelope is retained, including for prefill-only requests.",
        "Roofline is anchored down if necessary to reproduce simulated group floors.",
        "Measured means a matching simulated shape, not independent physical calibration; "
        "the input simulation may itself use deduplication or context interpolation.",
        "Outside calibration, baseline floors are endpoint-clamped; only roofline work grows.",
        "max_model_len comes from supplied run/meta; absent metadata defaults to 2048.",
    ]
    model = _model(real_timeline.get("meta", {}), assumptions)
    steps = [s for s in simulated_timeline.get("steps", []) if not s.get("warmup", False)]
    requests = [r for r in simulated_timeline.get("requests", []) if not r.get("warmup", False)]
    if not steps or not requests:
        raise ValueError("a finished timeline with steady steps and requests is required")
    prefills = [s for s in steps if s["kind"] == "prefill"]
    repeats = Counter(s["repeat"] for s in prefills)
    if not prefills or any(n != 1 for n in repeats.values()):
        raise ValueError("one unchunked prefill per repeat is required")
    if Counter(r["repeat"] for r in requests) != repeats:
        raise ValueError("one request per repeat is required")
    prompts = {_positive_int(s["num_scheduled_tokens"], "prefill tokens") for s in prefills}
    outputs = {r["num_output_tokens"] for r in requests}
    if len(prompts) != 1 or len(outputs) != 1:
        raise ValueError("a single profiled request shape is required")
    p0, n0 = prompts.pop(), outputs.pop()
    if type(n0) is not int or n0 < 0:
        raise ValueError("invalid profiled output length")
    groups = defaultdict(list)
    hosts, gaps, residuals = (defaultdict(list) for _ in range(3))
    for step in steps:
        kind = step["kind"]
        if kind not in ("prefill", "decode") or step.get("num_reqs", 1) != 1:
            raise ValueError("only single-request prefill/decode timelines are supported")
        context = _positive_int(step.get("context_length"), "context_length")
        if kind == "prefill" and context != p0:
            raise ValueError("prefill context must equal prompt tokens")
        totals = dict.fromkeys(GROUPS, 0.0)
        kernels = step["kernels"]
        for kernel, group in zip(kernels, kernel_groups(kernels), strict=True):
            totals[group] += _time(kernel["duration_ns"], "kernel duration")
        busy = _time(step["gpu_busy_ns"], "gpu_busy_ns")
        wall = _time(step["baseline_wall_ns"], "baseline_wall_ns")
        hosts[kind].append(_time(wall - busy, "host time"))
        residuals[kind].append(max(0, busy - sum(totals.values())))
        groups[(kind, context)].append(totals)
        if step.get("baseline_gap_before_ns") is not None:
            gaps[kind].append(_time(step["baseline_gap_before_ns"], "gap"))
    baselines = {
        key: {g: median(row[g] for row in rows) for g in GROUPS} for key, rows in groups.items()
    }
    decode = {ctx: values for (kind, ctx), values in baselines.items() if kind == "decode"}
    return PerfProfile(
        gpu_config,
        model,
        p0,
        n0,
        model_limit(real_timeline),
        baselines[("prefill", p0)],
        dict(sorted(decode.items())),
        {kind: median(values) for kind, values in hosts.items()},
        {kind: median(values) for kind, values in residuals.items()},
        {kind: median(gaps[kind]) if gaps[kind] else 0.0 for kind in hosts},
        median(_time(r.get("pre_first_step_ns", 0), "pre-step envelope") for r in requests),
        median(
            _time(r.get("post_last_step_ns", 0), "post-step envelope", signed=True)
            for r in requests
        ),
        {
            "prefill_tokens": (p0, p0),
            "decode_context": (min(decode), max(decode)) if decode else None,
        },
        tuple(assumptions),
        _memory_utilization(real_timeline),
    )


def _roofline(
    profile: PerfProfile, queries: list[int], contexts: list[int], sampling_seqs: int | None = None
) -> dict[str, float]:
    m, spec = profile.model_config, GPU_SPECS[profile.gpu_config]
    t, seqs = sum(queries), len(contexts)
    seqs = seqs if sampling_seqs is None else sampling_seqs
    h, layers, width = m.hidden_size, m.num_layers, m.dtype_bytes
    work = {
        "linear": (
            2 * m.non_embedding_params * t,
            m.non_embedding_params * width + LINEAR_ACTIVATION_PASSES * t * h * layers * width,
        ),
        "attention": (
            4 * layers * h * sum(q * c for q, c in zip(queries, contexts, strict=True)),
            sum(contexts) * m.kv_bytes_per_token,
        ),
        "kv_write": (0, t * m.kv_bytes_per_token),
        "elementwise": (0, ELEMENTWISE_PASSES * t * h * layers * width),
        "lm_head": (2 * h * m.vocab_size * seqs, m.vocab_size * h * width if seqs else 0),
        "sampler": (m.vocab_size * seqs, 2 * m.vocab_size * width * seqs),
    }
    return {
        group: max(
            flops / (spec.tflops * 1e12 * COMPUTE_FRACTION),
            size / (spec.bandwidth_tb_s * 1e12 * MEMORY_FRACTION),
        )
        * 1e9
        for group, (flops, size) in work.items()
    }


def _decode_baseline(profile: PerfProfile, context: int) -> tuple[dict[str, float], int]:
    points = list(profile.decode_gpu_ns)
    if not points:
        raise ValueError("profile has no decode calibration")
    anchor = min(max(context, points[0]), points[-1])
    upper = bisect_left(points, anchor)
    if points[upper] == anchor:
        return profile.decode_gpu_ns[anchor], anchor
    lo, hi = points[upper - 1], points[upper]
    weight = (anchor - lo) / (hi - lo)
    return {
        g: profile.decode_gpu_ns[lo][g] * (1 - weight) + profile.decode_gpu_ns[hi][g] * weight
        for g in GROUPS
    }, anchor


def step_gpu_time(
    profile: PerfProfile,
    tokens: int | list[int],
    contexts: int | list[int],
    *,
    kind: str | None = None,
    sampling_seqs: int | None = None,
) -> StepGPUTime:
    """Total query tokens (uniform per sequence), or explicit per-sequence query counts.

    Pass kind='prefill' explicitly for a one-token prompt, indistinguishable from decode by shape.
    """
    contexts = [contexts] if isinstance(contexts, int) else list(contexts)
    if not contexts:
        raise ValueError("contexts must not be empty")
    if sampling_seqs is not None and (
        type(sampling_seqs) is not int or not 0 <= sampling_seqs <= len(contexts)
    ):
        raise ValueError("sampling_seqs must be between zero and the batch size")
    for context in contexts:
        _positive_int(context, "context")
    if isinstance(tokens, int):
        _positive_int(tokens, "tokens")
        if tokens % len(contexts):
            raise ValueError("provide per-sequence query counts for nonuniform batches")
        queries = [tokens // len(contexts)] * len(contexts)
    else:
        queries = list(tokens)
    if len(queries) != len(contexts):
        raise ValueError("query/context counts differ")
    for q, c in zip(queries, contexts, strict=True):
        if _positive_int(q, "query tokens") > c or c > profile.max_model_len:
            raise ValueError("queries/contexts exceed supported model shape")
    kind = kind or ("prefill" if any(q > 1 for q in queries) else "decode")
    if kind == "prefill":
        baseline, anchor_q, anchor_c = (
            profile.prefill_gpu_ns,
            profile.prompt_tokens,
            profile.prompt_tokens,
        )
    elif kind == "decode":
        baseline, anchor_c = _decode_baseline(profile, max(contexts))
        anchor_q = 1
    else:
        raise ValueError("kind must be prefill or decode")
    roof = _roofline(profile, queries, contexts, sampling_seqs)
    anchor = _roofline(profile, [anchor_q], [anchor_c])
    groups = {g: max(baseline[g], roof[g] * min(1.0, baseline[g] / anchor[g])) for g in GROUPS}
    if sampling_seqs == 0:
        groups.update(lm_head=0.0, sampler=0.0)
    return StepGPUTime(groups, profile.residual_gpu_ns[kind])


def step_provenance(profile: PerfProfile, kind: str, tokens: int, context: int) -> str:
    if kind == "prefill":
        return "measured" if tokens == context == profile.prompt_tokens else "extrapolated:roofline"
    if context in profile.decode_gpu_ns:
        return "measured"
    bounds = profile.calibration_range["decode_context"]
    if bounds and bounds[0] <= context <= bounds[1]:
        return "interpolated"
    return "extrapolated:roofline"


def simulate_request(profile: PerfProfile, prompt_tokens: int, output_tokens: int) -> RequestResult:
    _positive_int(prompt_tokens, "prompt_tokens")
    if type(output_tokens) is not int or output_tokens < 0:
        raise ValueError("output_tokens must be a nonnegative integer")
    if prompt_tokens + output_tokens > profile.max_model_len:
        raise ValueError(f"request exceeds max_model_len={profile.max_model_len}")
    cursor = profile.pre_first_step_ns
    steps = []
    counts = Counter(
        dict.fromkeys(("measured", "interpolated", "extrapolated:roofline", "assumed-host"), 0)
    )
    shapes = [("prefill", prompt_tokens, prompt_tokens)] + [
        ("decode", 1, prompt_tokens + i) for i in range(1, output_tokens)
    ]
    for kind, tokens, context in shapes:
        gpu = step_gpu_time(profile, tokens, context, kind=kind)
        provenance = step_provenance(profile, kind, tokens, context)
        gap = profile.gap_ns[kind] if steps else 0.0
        start = cursor + gap
        wall = profile.host_ns[kind] + gpu.total_ns
        cursor = start + wall
        steps.append(
            RequestStep(
                kind,
                tokens,
                context,
                gpu.groups_ns,
                gpu.residual_ns,
                gpu.total_ns,
                profile.host_ns[kind],
                gap,
                wall,
                start,
                cursor,
                provenance,
            )
        )
        counts.update((provenance, "assumed-host"))
    itls = [(s.wall_ns + s.gap_before_ns) / 1e6 for s in steps[1:]]
    return RequestResult(
        prompt_tokens,
        output_tokens,
        steps[0].end_ns / 1e6 if output_tokens else None,
        mean(itls) if itls else None,
        median(itls) if itls else None,
        (cursor + profile.post_last_step_ns) / 1e6,
        steps,
        dict(profile.calibration_range),
        profile.assumptions,
        dict(counts),
    )
