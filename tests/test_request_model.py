from copy import deepcopy
from dataclasses import replace
from statistics import mean, median

import pytest

from cascade.samples import SAMPLES_ROOT, Bundle
from cascade.sim.request_model import (
    COMPUTE_FRACTION,
    GPU_SPECS,
    GROUPS,
    MEMORY_FRACTION,
    build_profile,
    classify_kernel,
    kernel_groups,
    simulate_request,
    step_gpu_time,
)


@pytest.fixture(scope="module", params=[("h100", "SM90_H100"), ("a100", "SM80_A100")])
def sample(request):
    name, config = request.param
    bundle = Bundle(SAMPLES_ROOT / f"opt125m-{name}-e2e")
    real = bundle.read_json("corrected_timeline.json")
    real["meta"]["run"] = bundle.read_json("run.json")
    simulated = bundle.read_json("sim/simulated_timeline.json")
    return real, simulated, build_profile(real, simulated, config)


def test_profiled_request_matches_finished_simulation(sample):
    _, simulated, profile = sample
    result = simulate_request(profile, profile.prompt_tokens, profile.output_tokens)
    requests = [r for r in simulated["requests"] if not r["warmup"]]
    assert result.e2e_ms == pytest.approx(mean(r["e2e_ms"] for r in requests), rel=0.01)
    assert result.ttft_ms == pytest.approx(mean(r["ttft_ms"] for r in requests), rel=0.02)
    assert len(result.steps) == profile.output_tokens
    assert result.provenance_counts == {
        "measured": profile.output_tokens,
        "interpolated": 0,
        "extrapolated:roofline": 0,
        "assumed-host": profile.output_tokens,
    }
    assert all(s.host_provenance == "assumed-host" for s in result.steps)
    for step in result.steps:
        baseline = (
            profile.prefill_gpu_ns
            if step.kind == "prefill"
            else profile.decode_gpu_ns[step.context]
        )
        assert step.groups_ns == pytest.approx(baseline)
        assert step.gpu_ns == pytest.approx(sum(baseline.values()) + step.residual_gpu_ns)
        assert step.wall_ns == step.host_ns + step.gpu_ns


def test_classifier_identifies_native_vocab_projection_and_all_groups(sample):
    _, simulated, profile = sample
    for step in simulated["steps"]:
        if step["warmup"]:
            continue
        groups = kernel_groups(step["kernels"])
        head = groups.index("lm_head")
        assert groups.count("lm_head") == 1
        assert groups.count("linear") == 48
        assert groups.count("attention") == (13 if profile.gpu_config == "SM90_H100" else 12)
        assert groups.count("kv_write") == 12
        assert groups.count("sampler") == 1
        assert head > max(i for i, g in enumerate(groups) if g == "linear")
        assert step["kernels"][head]["seq"] == (
            173 if profile.gpu_config == "SM90_H100" else 160 if step["kind"] == "prefill" else 148
        )
        totals = {
            g: sum(
                k["duration_ns"]
                for k, group in zip(step["kernels"], groups, strict=True)
                if group == g
            )
            for g in GROUPS
        }
        baseline = (
            profile.prefill_gpu_ns
            if step["kind"] == "prefill"
            else profile.decode_gpu_ns[step["context_length"]]
        )
        assert baseline == pytest.approx(totals)
    for kind in ("prefill", "decode"):
        steps = [s for s in simulated["steps"] if not s["warmup"] and s["kind"] == kind]
        assert profile.host_ns[kind] == median(
            s["baseline_wall_ns"] - s["gpu_busy_ns"] for s in steps
        )


@pytest.mark.parametrize(
    ("name", "group"),
    [
        ("cublas_gemv", "linear"),
        ("cutlass_xmma", "linear"),
        ("sm90_wgmma", "linear"),
        ("sm80_tensorop", "linear"),
        ("flash_fwd_sm90", "attention"),
        ("paged_attention", "attention"),
        ("reshape_and_cache_flash", "kv_write"),
        ("lm_head_matmul", "lm_head"),
        ("top_k_sampling", "sampler"),
        ("vectorized_layer_norm", "elementwise"),
        ("cublasLt_splitKreduce_kernel", "elementwise"),
    ],
)
def test_kernel_names(name, group):
    assert classify_kernel(name) == group


def test_empty_and_non_vocab_kernel_lists():
    assert kernel_groups([]) == []
    assert kernel_groups([{"name": "gemm", "grid": [1, 1, 1]}]) == ["linear"]


def test_legacy_string_grid_vocab_projection():
    kernels = [{"name": "gemm", "grid": grid} for grid in ("1x1x1", "2x1x1", "132x1x1")]
    assert kernel_groups(kernels) == ["linear", "linear", "lm_head"]


def test_prompt_and_output_monotonicity(sample):
    _, _, profile = sample
    prompts = list(range(1, 42)) + [128, 1024, 1800]
    values = [simulate_request(profile, p, 32).e2e_ms for p in prompts]
    assert values == sorted(values)
    for p in (1, 6, 128, 1024, 1800):
        values = [simulate_request(profile, p, n).e2e_ms for n in (0, 1, 2, 32, 128, 200)]
        assert values == sorted(values)


def test_large_prompt_is_roofline_extrapolation(sample):
    _, _, profile = sample
    result = simulate_request(profile, 1800, 200)
    prefill = result.steps[0]
    assert prefill.provenance == "extrapolated:roofline"
    assert prefill.groups_ns["linear"] > profile.prefill_gpu_ns["linear"]
    assert result.provenance_counts["extrapolated:roofline"] == 200
    assert result.provenance_counts["assumed-host"] == 200
    assert result.calibration_range == profile.calibration_range
    assert "0.7" in result.assumptions[0] and "0.8" in result.assumptions[0]
    # The vocabulary projection and sampling are per sequence, not per prompt token.
    assert prefill.groups_ns["lm_head"] == profile.prefill_gpu_ns["lm_head"]
    assert prefill.groups_ns["sampler"] == profile.prefill_gpu_ns["sampler"]


def test_sparse_calibration_interpolation_and_provenance(sample):
    _, _, profile = sample
    profile = replace(
        profile,
        decode_gpu_ns={c: profile.decode_gpu_ns[c] for c in (8, 16, 32)},
        calibration_range={"prefill_tokens": (6, 6), "decode_context": (8, 32)},
    )
    result = simulate_request(profile, 6, 32)
    assert result.provenance_counts == {
        "measured": 4,
        "interpolated": 22,
        "extrapolated:roofline": 6,
        "assumed-host": 32,
    }
    middle = step_gpu_time(profile, 1, 12)
    for group in GROUPS:
        assert middle.groups_ns[group] == pytest.approx(
            (profile.decode_gpu_ns[8][group] + profile.decode_gpu_ns[16][group]) / 2
        )


def test_zero_and_one_output_tokens_are_prefill_only(sample):
    _, _, profile = sample
    for output in (0, 1):
        result = simulate_request(profile, 1, output)
        assert len(result.steps) == 1
        assert result.steps[0].kind == "prefill"
        assert result.ttft_ms == (result.steps[0].end_ns / 1e6 if output else None)
        assert result.itl_ms_mean is result.itl_ms_p50 is None
        assert (
            result.e2e_ms
            == (profile.pre_first_step_ns + result.steps[0].wall_ns + profile.post_last_step_ns)
            / 1e6
        )


@pytest.mark.parametrize(
    "prompt,output",
    [(0, 1), (-1, 2), (1, -1), (2048, 1), (1800, 249), (1.5, 1), (1, 2.0), (True, 1), (1, False)],
)
def test_invalid_requests(sample, prompt, output):
    with pytest.raises(ValueError):
        simulate_request(sample[2], prompt, output)


def test_model_limit_metadata_and_default(sample):
    real, simulated, profile = sample
    real = deepcopy(real)
    real["meta"].pop("run", None)
    assert build_profile(real, simulated, profile.gpu_config).max_model_len == 2048
    real["meta"]["run"] = {"effective_llm_kwargs": {"max_model_len": 100}}
    limited = build_profile(real, simulated, profile.gpu_config)
    assert limited.max_model_len == 100
    simulate_request(limited, 68, 32)
    with pytest.raises(ValueError, match="max_model_len=100"):
        simulate_request(limited, 69, 32)


def test_invalid_and_empty_profiles(sample):
    real, simulated, profile = sample
    with pytest.raises(ValueError, match="unknown GPU"):
        build_profile(real, simulated, "B200")
    for field in ("steps", "requests"):
        empty = {**simulated, field: []}
        with pytest.raises(ValueError, match="finished timeline"):
            build_profile(real, empty, profile.gpu_config)
    invalid = deepcopy(simulated)
    next(s for s in invalid["steps"] if not s["warmup"])["num_reqs"] = 2
    with pytest.raises(ValueError, match="single-request"):
        build_profile(real, invalid, profile.gpu_config)
    invalid = deepcopy(simulated)
    next(s for s in invalid["steps"] if not s["warmup"])["kernels"][0]["duration_ns"] = -1
    with pytest.raises(ValueError, match="duration"):
        build_profile(real, invalid, profile.gpu_config)


def test_missing_decode_is_explicit(sample):
    _, _, profile = sample
    profile = replace(profile, decode_gpu_ns={})
    simulate_request(profile, 6, 0)
    with pytest.raises(ValueError, match="no decode calibration"):
        simulate_request(profile, 6, 2)


@pytest.mark.parametrize(
    "tokens,contexts", [(0, 8), (1, []), ([1], [8, 9]), (1, [8, 9]), (9, 8), (1, 3000)]
)
def test_invalid_step_shapes(sample, tokens, contexts):
    with pytest.raises(ValueError):
        step_gpu_time(sample[2], tokens, contexts)


def test_roofline_anchor_and_model_dimensions(sample):
    _, _, profile = sample
    config = replace(profile.model_config, hidden_size=100_000)
    anchored = replace(profile, model_config=config)
    gpu = step_gpu_time(anchored, profile.prompt_tokens, profile.prompt_tokens)
    assert gpu.groups_ns == pytest.approx(profile.prefill_gpu_ns)
    assert step_gpu_time(anchored, 1800, 1800).groups_ns["linear"] > gpu.groups_ns["linear"]


def test_datasheet_specs():
    assert COMPUTE_FRACTION == 0.7
    assert MEMORY_FRACTION == 0.8
    assert [(s.tflops, s.bandwidth_tb_s, s.memory_gb) for s in GPU_SPECS.values()] == [
        (989, 3.35, 80),
        (989, 4.8, 141),
        (312, 2.039, 80),
    ]
