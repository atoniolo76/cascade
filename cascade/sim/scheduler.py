"""Deterministic, single-GPU discrete-event approximation of vLLM v1 scheduling."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from math import ceil, isfinite
from statistics import mean, median

from cascade.sim.request_model import GPU_SPECS, PerfProfile, step_gpu_time, step_provenance

FIXED_ACTIVATION_RESERVE_BYTES = 1 << 30


@dataclass(frozen=True)
class SchedulerConfig:
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    block_size: int = 16
    enable_chunked_prefill: bool = True
    gpu_memory_utilization: float | None = None
    num_kv_blocks: int | None = None


@dataclass(frozen=True)
class SchedulerRequest:
    id: str
    arrival_s: float
    prompt_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class RequestMetrics:
    id: str
    arrival_s: float
    ttft_ms: float | None
    itl_ms_mean: float | None
    itl_ms_p50: float | None
    e2e_ms: float
    queue_time_ms: float
    preemptions: int
    token_timestamps_s: tuple[float, ...]


@dataclass(frozen=True)
class SchedulerStep:
    t_start_s: float
    wall_s: float
    t_end_s: float
    num_seqs: int
    num_tokens: int
    prefill_tokens: int
    decode_tokens: int
    sampling_seqs: int
    kv_blocks_used: int
    request_ids: tuple[str, ...]
    query_tokens: tuple[int, ...]
    contexts: tuple[int, ...]
    groups_ns: dict[str, float]
    provenance_counts: dict[str, int]


@dataclass(frozen=True)
class SchedulerResult:
    requests: list[RequestMetrics]
    steps: list[SchedulerStep]
    throughput_output_tokens_s: float
    p50_ttft_ms: float | None
    p99_ttft_ms: float | None
    p50_itl_ms: float | None
    p50_e2e_ms: float | None
    p99_e2e_ms: float | None
    idle_time_s: float
    config: SchedulerConfig
    assumptions: tuple[str, ...]


@dataclass
class _State:
    request: SchedulerRequest
    target: int | None
    processed: int = 0
    blocks: int = 0
    emissions: list[float] = field(default_factory=list)
    first_start_ns: float | None = None
    completion_ns: float | None = None
    preemptions: int = 0


@dataclass
class _Work:
    state: _State
    query: int
    kind: str

    @property
    def context(self) -> int:
        return self.state.processed + self.query


def _integer(value, name: str, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _resolved_config(profile: PerfProfile, config: SchedulerConfig) -> SchedulerConfig:
    for name in ("max_num_seqs", "max_num_batched_tokens", "block_size"):
        _integer(getattr(config, name), name)
    if type(config.enable_chunked_prefill) is not bool:
        raise ValueError("enable_chunked_prefill must be boolean")
    util = config.gpu_memory_utilization
    util = profile.gpu_memory_utilization if util is None else util
    if isinstance(util, bool) or not isinstance(util, (int, float)) or not 0 < util <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    blocks = config.num_kv_blocks
    if blocks is None:
        model = profile.model_config
        weights = (
            model.non_embedding_params + model.hidden_size * model.vocab_size
        ) * model.dtype_bytes
        available = GPU_SPECS[profile.gpu_config].memory_gb * 1e9 * util
        available -= weights + FIXED_ACTIVATION_RESERVE_BYTES
        blocks = int(available // (config.block_size * model.kv_bytes_per_token))
    _integer(blocks, "num_kv_blocks")
    return replace(config, gpu_memory_utilization=float(util), num_kv_blocks=blocks)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


class _Scheduler:
    def __init__(self, profile, requests, config, max_steps):
        self.profile = profile
        self.config = _resolved_config(profile, config)
        self.max_steps = max_steps
        if max_steps is not None:
            _integer(max_steps, "max_steps")
        self.states = []
        seen = set()
        for request in requests:
            if isinstance(request, dict):
                try:
                    request = SchedulerRequest(**request)
                except TypeError as error:
                    raise ValueError("invalid request fields") from error
            if not isinstance(request, SchedulerRequest):
                raise ValueError("requests must be SchedulerRequest objects or dictionaries")
            if not isinstance(request.id, str) or not request.id or request.id in seen:
                raise ValueError("request IDs must be nonempty and unique")
            seen.add(request.id)
            arrival = request.arrival_s
            if isinstance(arrival, bool) or not isinstance(arrival, (float, int)):
                raise ValueError("arrival_s must be finite and nonnegative")
            if not isfinite(arrival) or arrival < 0 or not isfinite(arrival * 1e9):
                raise ValueError("arrival_s must be finite and nonnegative")
            _integer(request.prompt_tokens, "prompt_tokens")
            _integer(request.output_tokens, "output_tokens", 0)
            if request.prompt_tokens + request.output_tokens > profile.max_model_len:
                raise ValueError("request exceeds max_model_len")
            if not self.config.enable_chunked_prefill:
                if request.prompt_tokens > self.config.max_num_batched_tokens:
                    raise ValueError("prompt exceeds token budget with chunked prefill disabled")
            context = request.prompt_tokens + max(0, request.output_tokens - 1)
            if ceil(context / self.config.block_size) > self.config.num_kv_blocks:
                raise ValueError(f"request {request.id} cannot fit KV capacity even in isolation")
            if request.output_tokens > 1 and not profile.decode_gpu_ns:
                raise ValueError("profile has no decode calibration")
            self.states.append(_State(request, request.prompt_tokens))
        self.future = deque(sorted(self.states, key=lambda s: s.request.arrival_s))
        self.waiting = deque()
        self.running = []
        self.steps = []
        self.now = 0.0
        self.idle_ns = 0.0

    @property
    def free_blocks(self):
        return self.config.num_kv_blocks - sum(s.blocks for s in self.running)

    def _blocks(self, state, query):
        return ceil((state.processed + query) / self.config.block_size)

    def _query(self, state, budget):
        if state.target is None:
            return min(1, budget)
        remaining = state.target - state.processed
        if self.config.enable_chunked_prefill:
            return min(remaining, budget)
        if remaining > self.config.max_num_batched_tokens:
            raise ValueError("recompute prompt exceeds token budget with chunked prefill disabled")
        return remaining if remaining <= budget else 0

    def _preempt(self, planned, preempted):
        victim = self.running.pop()
        planned[:] = [w for w in planned if w.state is not victim]
        victim.blocks = victim.processed = 0
        victim.target = victim.request.prompt_tokens + len(victim.emissions)
        victim.preemptions += 1
        self.waiting.appendleft(victim)
        preempted.add(victim.request.id)
        return victim

    def _plan(self):
        planned, preempted = [], set()

        def budget():
            return self.config.max_num_batched_tokens - sum(w.query for w in planned)

        # Decode first; admission order is retained separately for LIFO preemption.
        for state in sorted(self.running, key=lambda s: s.target is not None):
            if state not in self.running:
                continue
            query = self._query(state, budget())
            if not query:
                continue
            while self._blocks(state, query) - state.blocks > self.free_blocks:
                if self._preempt(planned, preempted) is state:
                    break
            if state not in self.running:
                continue
            state.blocks = self._blocks(state, query)
            planned.append(_Work(state, query, "decode" if state.target is None else "prefill"))
        while self.waiting and len(self.running) < self.config.max_num_seqs and budget():
            state = self.waiting[0]
            if state.request.id in preempted:
                break
            query = self._query(state, budget())
            if self.config.enable_chunked_prefill:
                query = min(query, self.free_blocks * self.config.block_size)
            if not query or self._blocks(state, query) > self.free_blocks:
                break
            self.waiting.popleft()
            self.running.append(state)
            state.blocks = self._blocks(state, query)
            planned.append(_Work(state, query, "prefill"))
        return planned, preempted

    def _execute(self, planned, first):
        profile = self.profile
        prefills = sum(w.query for w in planned if w.kind == "prefill")
        decodes = sum(w.query for w in planned if w.kind == "decode")
        kind = "prefill" if prefills else "decode"
        queries, contexts = [w.query for w in planned], [w.context for w in planned]
        sampling = sum(w.state.target is None or w.context == w.state.target for w in planned)
        gpu = step_gpu_time(profile, queries, contexts, kind=kind, sampling_seqs=sampling)
        wall = profile.host_ns[kind] + gpu.total_ns
        start = self.now + (0 if first else profile.gap_ns[kind])
        end = start + wall
        provenance = "extrapolated:roofline"
        if len(planned) == 1 and sampling == 1:
            provenance = step_provenance(profile, kind, queries[0], contexts[0])
        counts = dict.fromkeys(
            ("measured", "interpolated", "extrapolated:roofline", "assumed-host"), 0
        )
        counts[provenance] = counts["assumed-host"] = 1
        self.steps.append(
            SchedulerStep(
                start / 1e9,
                wall / 1e9,
                end / 1e9,
                len(planned),
                prefills + decodes,
                prefills,
                decodes,
                sampling,
                self.config.num_kv_blocks - self.free_blocks,
                tuple(w.state.request.id for w in planned),
                tuple(queries),
                tuple(contexts),
                gpu.groups_ns,
                counts,
            )
        )
        for work in planned:
            state = work.state
            if state.first_start_ns is None:
                state.first_start_ns = start
            state.processed += work.query
            if state.target is not None and state.processed != state.target:
                continue
            state.target = None
            if len(state.emissions) < state.request.output_tokens:
                state.emissions.append(end)
            if len(state.emissions) == state.request.output_tokens:
                state.completion_ns = end + profile.post_last_step_ns
                self.running.remove(state)
                state.blocks = 0
        self.now = end

    def run(self):
        first = True
        while self.future or self.waiting or self.running:
            if self.max_steps is not None and len(self.steps) >= self.max_steps:
                raise ValueError(f"scenario exceeds step limit={self.max_steps}")
            if not self.waiting and not self.running:
                arrival = self.future[0].request.arrival_s * 1e9
                self.idle_ns += max(0, arrival - self.now)
                self.now = max(self.now, arrival) + self.profile.pre_first_step_ns
                first = True
            while self.future and self.future[0].request.arrival_s * 1e9 <= self.now:
                self.waiting.append(self.future.popleft())
            planned, preempted = self._plan()
            if not planned:
                if preempted:
                    continue
                raise RuntimeError("scheduler cannot make progress")
            self._execute(planned, first)
            while self.future and self.future[0].request.arrival_s * 1e9 <= self.now:
                self.waiting.append(self.future.popleft())
            first = False
        return self._result()

    def _result(self):
        results, intervals = [], []
        for state in self.states:
            arrival = state.request.arrival_s * 1e9
            itls = [
                (b - a) / 1e6 for a, b in zip(state.emissions, state.emissions[1:], strict=False)
            ]
            intervals.extend(itls)
            results.append(
                RequestMetrics(
                    state.request.id,
                    state.request.arrival_s,
                    (state.emissions[0] - arrival) / 1e6 if state.emissions else None,
                    mean(itls) if itls else None,
                    median(itls) if itls else None,
                    (state.completion_ns - arrival) / 1e6,
                    (state.first_start_ns - arrival) / 1e6,
                    state.preemptions,
                    tuple(t / 1e9 for t in state.emissions),
                )
            )
        elapsed = (
            max(s.completion_ns for s in self.states)
            - min(s.request.arrival_s * 1e9 for s in self.states)
            if self.states
            else 0
        ) / 1e9
        ttfts = [r.ttft_ms for r in results if r.ttft_ms is not None]
        e2es = [r.e2e_ms for r in results]
        assumptions = self.profile.assumptions + (
            "Scheduler settings are assumptions, not captured vLLM resolved values: "
            f"{self.config}.",
            "KV estimate uses decimal GPU GB * utilization minus dense body + tied embedding "
            f"weights and fixed activation reserve={FIXED_ACTIVATION_RESERVE_BYTES} bytes; "
            "no fragmentation.",
            "Explicit num_kv_blocks overrides the estimated memory budget.",
            "Decode-first running work, FCFS waiting (stable arrival ties); most recently admitted "
            "running request preempted by recompute, preserving previously emitted tokens.",
            "Preempted requests cannot be readmitted in the same step; queue time ends at first "
            "GPU step start. Recompute consumes tokens but does not re-emit prior output tokens.",
            "Without chunking, recompute prompts must also fit the token budget "
            "or raise ValueError.",
            "Pre-step envelope once per busy period; kind-specific gaps between busy steps; "
            "completion uses signed post-step envelope, emissions use actual step ends.",
            "All multi-sequence/chunked shapes are roofline extrapolations; host is assumed-host.",
            "Percentiles use linear interpolation; throughput spans first arrival to "
            "last completion.",
        )
        return SchedulerResult(
            results,
            self.steps,
            sum(len(s.emissions) for s in self.states) / elapsed if elapsed > 0 else 0,
            _percentile(ttfts, 0.5),
            _percentile(ttfts, 0.99),
            _percentile(intervals, 0.5),
            _percentile(e2es, 0.5),
            _percentile(e2es, 0.99),
            self.idle_ns / 1e9,
            self.config,
            assumptions,
        )


def simulate_scheduler(
    profile: PerfProfile,
    requests: list[SchedulerRequest | dict],
    config: SchedulerConfig | None = None,
    *,
    max_steps: int | None = None,
) -> SchedulerResult:
    """Run without I/O; output order follows input order, scheduling ties are stable."""
    return _Scheduler(profile, requests, config or SchedulerConfig(), max_steps).run()
