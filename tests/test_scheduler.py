from dataclasses import replace

import pytest

from cascade.samples import SAMPLES_ROOT, Bundle
from cascade.sim.request_model import GPU_SPECS, build_profile, simulate_request, step_gpu_time
from cascade.sim.scheduler import (
    FIXED_ACTIVATION_RESERVE_BYTES,
    SchedulerConfig,
    SchedulerRequest,
    simulate_scheduler,
)


@pytest.fixture(scope="module", params=[("h100", "SM90_H100"), ("a100", "SM80_A100")])
def profile(request):
    name, config = request.param
    bundle = Bundle(SAMPLES_ROOT / f"opt125m-{name}-e2e")
    real = bundle.read_json("corrected_timeline.json")
    real["meta"]["run"] = bundle.read_json("run.json")
    return build_profile(real, bundle.read_json("sim/simulated_timeline.json"), config)


def requests(n, prompt=512, output=128):
    return [SchedulerRequest(str(i), 0, prompt, output) for i in range(n)]


@pytest.mark.parametrize("prompt,output", [(6, 32), (512, 128), (1, 0), (1, 1), (128, 2)])
def test_single_request_equals_request_model(profile, prompt, output):
    expected = simulate_request(profile, prompt, output)
    actual = simulate_scheduler(profile, requests(1, prompt, output))
    req = actual.requests[0]
    for metric in ("ttft_ms", "itl_ms_mean", "itl_ms_p50", "e2e_ms"):
        value = getattr(expected, metric)
        assert getattr(req, metric) == (
            pytest.approx(value, rel=1e-6) if value is not None else None
        )
    assert len(actual.steps) == len(expected.steps)
    assert len(req.token_timestamps_s) == output
    for a, e in zip(actual.steps, expected.steps, strict=True):
        assert a.wall_s == pytest.approx(e.wall_ns / 1e9, rel=1e-6)
        assert a.t_start_s == pytest.approx(e.start_ns / 1e9, rel=1e-6)
        assert a.groups_ns == e.groups_ns
    assert req.preemptions == 0


def test_two_requests_batch_instead_of_serial_execution(profile):
    batch = simulate_scheduler(profile, requests(2, 6, 8))
    serial = simulate_scheduler(profile, requests(2, 6, 8), SchedulerConfig(max_num_seqs=1))
    assert batch.steps[0].num_seqs == 2
    assert batch.steps[0].sampling_seqs == 2
    assert len(batch.steps) == 8 < len(serial.steps) == 16
    assert batch.throughput_output_tokens_s > serial.throughput_output_tokens_s
    assert batch.steps[0].provenance_counts == {
        "measured": 0,
        "interpolated": 0,
        "extrapolated:roofline": 1,
        "assumed-host": 1,
    }


def test_throughput_increases_with_concurrency(profile):
    values = [
        simulate_scheduler(profile, requests(n)).throughput_output_tokens_s for n in (1, 8, 64)
    ]
    assert values[0] < values[1] < values[2]


def test_long_prompt_chunking_and_sampling_only_on_completion(profile):
    profile = replace(profile, max_model_len=32768)
    result = simulate_scheduler(
        profile, requests(1, 10000, 3), SchedulerConfig(max_num_batched_tokens=2048)
    )
    prefills = [s for s in result.steps if s.prefill_tokens]
    assert len(prefills) == 5
    assert sum(s.prefill_tokens for s in prefills) == 10000
    assert all(s.num_tokens <= 2048 for s in result.steps)
    assert all(s.sampling_seqs == 0 for s in prefills[:-1])
    assert all(s.groups_ns["lm_head"] == s.groups_ns["sampler"] == 0 for s in prefills[:-1])
    assert prefills[-1].sampling_seqs == 1
    assert len(result.requests[0].token_timestamps_s) == 3
    assert all(s.provenance_counts["extrapolated:roofline"] == 1 for s in prefills)
    with pytest.raises(ValueError, match="chunked prefill disabled"):
        simulate_scheduler(
            profile,
            requests(1, 10000, 3),
            SchedulerConfig(
                max_num_batched_tokens=2048,
                enable_chunked_prefill=False,
            ),
        )


def test_non_chunked_admission_waits_for_full_token_budget(profile):
    result = simulate_scheduler(
        profile,
        requests(2, 64, 3),
        SchedulerConfig(
            max_num_batched_tokens=64,
            enable_chunked_prefill=False,
        ),
    )
    assert all(s.prefill_tokens in (0, 64) for s in result.steps)
    assert all(s.num_tokens <= 64 for s in result.steps)
    assert result.requests[1].queue_time_ms > result.requests[0].queue_time_ms


def test_recompute_preemption_finishes_without_duplicate_tokens(profile):
    result = simulate_scheduler(
        profile, requests(3, 8, 8), SchedulerConfig(block_size=4, num_kv_blocks=4)
    )
    assert sum(r.preemptions for r in result.requests) > 0
    assert result.requests[0].preemptions == 0
    assert all(len(r.token_timestamps_s) == 8 for r in result.requests)
    assert all(
        all(a < b for a, b in zip(r.token_timestamps_s, r.token_timestamps_s[1:], strict=False))
        for r in result.requests
    )
    assert max(s.kv_blocks_used for s in result.steps) <= 4
    assert sum(s.num_tokens for s in result.steps) > 3 * (8 + 7)
    assert all(s.num_tokens <= result.config.max_num_batched_tokens for s in result.steps)
    with pytest.raises(ValueError, match="in isolation"):
        simulate_scheduler(
            profile, requests(1, 8, 8), SchedulerConfig(block_size=4, num_kv_blocks=3)
        )


def test_decode_has_priority_over_partial_prefill(profile):
    result = simulate_scheduler(
        profile,
        [SchedulerRequest("a", 0, 6, 3), SchedulerRequest("b", 0, 20, 2)],
        SchedulerConfig(max_num_batched_tokens=8),
    )
    assert result.steps[0].prefill_tokens == 8
    assert result.steps[1].decode_tokens == 1
    assert result.steps[1].prefill_tokens == 7
    assert result.steps[1].sampling_seqs == 1
    assert all(s.num_seqs <= 2 and s.num_tokens <= 8 for s in result.steps)


def test_unchunked_recompute_must_fit_budget(profile):
    with pytest.raises(ValueError, match="recompute prompt exceeds token budget"):
        simulate_scheduler(
            profile,
            requests(2, 4, 8),
            SchedulerConfig(
                block_size=4,
                num_kv_blocks=4,
                max_num_batched_tokens=8,
                enable_chunked_prefill=False,
            ),
        )


def test_arrival_gaps_produce_idle_time(profile):
    result = simulate_scheduler(
        profile, [SchedulerRequest("a", 0, 6, 2), SchedulerRequest("b", 1, 6, 2)]
    )
    assert result.idle_time_s > 0.9
    assert result.requests[0].ttft_ms == pytest.approx(result.requests[1].ttft_ms, rel=1e-6)
    assert result.requests[0].e2e_ms == pytest.approx(result.requests[1].e2e_ms, rel=1e-6)
    assert result.steps[2].t_start_s >= 1
    assert all(s.num_seqs == 1 for s in result.steps)


def test_arrivals_during_step_wait_until_next_step(profile):
    result = simulate_scheduler(
        profile, [SchedulerRequest("a", 0, 6, 1), SchedulerRequest("b", 0.001, 6, 1)]
    )
    assert result.steps[0].request_ids == ("a",)
    assert result.steps[1].request_ids == ("b",)
    assert result.steps[1].t_start_s >= result.steps[0].t_end_s
    assert result.idle_time_s == 0
    assert result.requests[1].queue_time_ms > 1


def test_stable_fcfs_and_determinism(profile):
    inputs = [
        SchedulerRequest("z", 1, 6, 2),
        SchedulerRequest("b", 0, 6, 2),
        SchedulerRequest("a", 0, 6, 2),
    ]
    config = SchedulerConfig(max_num_seqs=1)
    first = simulate_scheduler(profile, inputs, config)
    assert first == simulate_scheduler(profile, inputs, config)
    assert first.steps[0].request_ids == ("b",)
    assert first.steps[2].request_ids == ("a",)
    assert [r.id for r in first.requests] == ["z", "b", "a"]


def test_capacity_estimate_and_utilization_override(profile):
    result = simulate_scheduler(profile, [])
    m = profile.model_config
    weights = (m.non_embedding_params + m.hidden_size * m.vocab_size) * m.dtype_bytes
    expected = int(
        (
            GPU_SPECS[profile.gpu_config].memory_gb * 1e9 * 0.8
            - weights
            - FIXED_ACTIVATION_RESERVE_BYTES
        )
        // (16 * m.kv_bytes_per_token)
    )
    assert result.config.num_kv_blocks == expected
    assert result.config.gpu_memory_utilization == 0.8
    assert (
        simulate_scheduler(
            replace(profile, gpu_memory_utilization=0.6), []
        ).config.gpu_memory_utilization
        == 0.6
    )
    config = SchedulerConfig(gpu_memory_utilization=0.5, num_kv_blocks=5)
    explicit = simulate_scheduler(profile, [], config)
    assert explicit.config.num_kv_blocks == 5
    assert explicit.config.gpu_memory_utilization == 0.5
    assert any("not captured vLLM" in a for a in result.assumptions)
    assert any("fixed activation reserve" in a for a in result.assumptions)


def test_profile_retains_run_memory_utilization():
    bundle = Bundle(SAMPLES_ROOT / "opt125m-h100-e2e")
    real = bundle.read_json("corrected_timeline.json")
    real["meta"]["run"] = {"effective_llm_kwargs": {"gpu_memory_utilization": 0.65}}
    sim = bundle.read_json("sim/simulated_timeline.json")
    assert build_profile(real, sim, "SM90_H100").gpu_memory_utilization == 0.65


def test_empty_input_and_dictionary_requests(profile):
    empty = simulate_scheduler(profile, [])
    assert empty.requests == empty.steps == []
    assert empty.throughput_output_tokens_s == 0
    assert empty.p50_e2e_ms is empty.p99_ttft_ms is empty.p50_itl_ms is None
    result = simulate_scheduler(
        profile, [{"id": "r", "arrival_s": 0, "prompt_tokens": 6, "output_tokens": 2}]
    )
    assert len(result.requests[0].token_timestamps_s) == 2


@pytest.mark.parametrize(
    "config",
    [
        SchedulerConfig(max_num_seqs=0),
        SchedulerConfig(block_size=-1),
        SchedulerConfig(max_num_batched_tokens=True),
        SchedulerConfig(num_kv_blocks=0),
        SchedulerConfig(gpu_memory_utilization=float("nan")),
        SchedulerConfig(gpu_memory_utilization=1.1),
        SchedulerConfig(enable_chunked_prefill=1),
    ],
)
def test_invalid_configs(profile, config):
    with pytest.raises(ValueError):
        simulate_scheduler(profile, requests(1), config)


@pytest.mark.parametrize(
    "change",
    [
        {"arrival_s": -1},
        {"arrival_s": float("inf")},
        {"id": ""},
        {"prompt_tokens": 0},
        {"output_tokens": -1},
        {"prompt_tokens": 2048, "output_tokens": 1},
    ],
)
def test_invalid_requests(profile, change):
    request = {"id": "r", "arrival_s": 0, "prompt_tokens": 6, "output_tokens": 2, **change}
    with pytest.raises(ValueError):
        simulate_scheduler(profile, [request])
    with pytest.raises(ValueError, match="unique"):
        simulate_scheduler(profile, requests(1) * 2)


def test_sampling_sequence_validation_and_zero_sample_cost(profile):
    gpu = step_gpu_time(profile, [4, 4], [4, 4], kind="prefill", sampling_seqs=0)
    assert gpu.groups_ns["lm_head"] == gpu.groups_ns["sampler"] == 0
    for count in (-1, 3, True):
        with pytest.raises(ValueError, match="sampling_seqs"):
            step_gpu_time(profile, [4, 4], [4, 4], kind="prefill", sampling_seqs=count)
