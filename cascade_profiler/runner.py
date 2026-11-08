"""vLLM workload traced with Accel-Sim 2.0's NVBit torch hook.

Mirrors util/tracer_nvbit/others/torch_hook/vllm_example.py from accel-sim-framework.
NVBit runs launch under accel-sim's torch_hook/run.sh, which sets CUDA_INJECTION64_PATH,
NVBIT_INSTRUMENTATION_ENABLED=0, VLLM_ENABLE_V1_MULTIPROCESSING=0, etc. The same script runs
uninstrumented under nsys; step-scope timing does not require torch_hook.

Trace scopes:
  layers  NVBit is on only inside the --layers modules (original behaviour).
  steps   NVBit is on for every kernel of the selected engine steps (--trace-steps; default
          all), i.e. complete forward passes incl. input prep, sampler and logits kernels.
          Startup / profiling / warmup / CUDA-graph-capture kernels all run inside LLM() and
          are never traced.

Every engine step (one LLMEngine.step() == one scheduler iteration == one forward pass, since
async scheduling is disabled with --no-async-scheduling) is wrapped in an NVTX range
"vllm_step_<i>" (with a nested "prefill_<i>"/"decode_<i>"/"mixed_<i>" range from schedule()
onwards; LLM() construction -- weight load, profiling, warmup, graph capture -- is "vllm_init")
and bracketed by torch.cuda.synchronize(), so its kernels can't leak into the
neighbouring step. Per-step metadata is written to --step-log (JSON):
  step, kind (prefill/decode/mixed), num_scheduled_tokens, num_reqs, wall_start_ns/wall_end_ns
  (time.time_ns), and under the tracer the tracer kernel-id range [first_kernel_id,
  end_kernel_id) launched inside the step (read from the tracer's stats_ctx_* file).
"""

import argparse
import atexit
import gc
import json
import logging
import os
import time
from ctypes import CDLL
from dataclasses import asdict
from pathlib import Path

import torch
from vllm import LLM, SamplingParams

from cascade_profiler.workload import Workload, execution_kwargs as llm_kwargs

try:
    import nvtx
except ImportError:  # NVTX ranges are only needed for the nsys run
    nvtx = None

EXEC_MODES = ("eager", "compile", "graphs")


def print_layer_names(model: torch.nn.Module):
    print("Available layers:")
    for name, module in model.named_modules():
        print(f"  {name}: {module.__class__.__name__}")


def apply_nvbit_hook(model: torch.nn.Module, layers_to_trace: list[str]):
    from torch_hook import TorchModelHookWrapper, hook_nvbit_to_layer

    hook_wrapper = TorchModelHookWrapper(model)
    for layer in layers_to_trace:
        hook_nvbit_to_layer(hook_wrapper, layer)  # NVBit on only during this layer


def load_nvbit_tool():
    """The injected NVBit tool (tracer or spinlock tool), or None when not instrumented."""
    path = os.environ.get("CUDA_INJECTION64_PATH")
    if not path:
        return None
    lib = CDLL(path)  # already loaded by the CUDA driver; dlopen returns the same handle
    try:
        lib.enable_nvbit_instrumentation.restype = None
        lib.disable_nvbit_instrumentation.restype = None
    except AttributeError:
        return None
    return lib


def tracer_next_kernel_id() -> int | None:
    """Id the Accel-Sim tracer will give the next kernel launch (ids start at 1).

    The tracer appends one stats line per launch (first half at launch, rest + newline after
    the kernel completes), so after a device sync the number of complete lines is exact.
    """
    if not os.environ.get("CUDA_INJECTION64_PATH", "").endswith("tracer_tool.so"):
        return None
    folder = Path(os.environ.get("TRACES_FOLDER") or os.getcwd()) / "traces"
    files = sorted(folder.glob("stats_ctx_*"))
    if len(files) != 1:
        return None
    return 1 + max(files[0].read_bytes().count(b"\n") - 1, 0)


class GCPauseRecorder:
    def __init__(self):
        self.total_ns = 0
        self.events = []
        self.starts = {}
        gc.callbacks.append(self.callback)

    def callback(self, phase, info):
        now = time.monotonic_ns()
        generation = info["generation"]
        if phase == "start":
            self.starts[generation] = now
        elif generation in self.starts:
            start = self.starts.pop(generation)
            self.total_ns += now - start
            self.events.append({"generation": generation, "start_ns": start,
                                "end_ns": now, "duration_ns": now - start})

    def close(self):
        if self.callback in gc.callbacks:
            gc.callbacks.remove(self.callback)


class StepRecorder:
    """Wraps LLMEngine.step with NVTX ranges, syncs, optional NVBit enable, and a JSON log."""

    def __init__(self, llm: LLM, nvbit_lib, trace_steps: set[int] | None, log_path: str):
        self.engine = llm.llm_engine
        self.llm = llm
        self.repeat = 0
        self.trace_repeat = None
        self.trace_steps_in_repeat = None
        self._repeat_steps = {}
        self.profiler_from_repeat = None
        self.profiling = False
        self.sync_steps = True
        self.gc = GCPauseRecorder()
        self._previous_gc = None
        self._closed = False
        atexit.register(self.close)
        self.nvbit_lib, self.trace_steps, self.log_path = nvbit_lib, trace_steps, log_path
        self.steps: list[dict] = []
        self._sched = None
        self._kind_pushed = False
        self._orig_step = self.engine.step
        self.engine.step = self.step
        try:  # InprocClient -> EngineCore -> Scheduler (VLLM_ENABLE_V1_MULTIPROCESSING=0)
            scheduler = self.engine.engine_core.engine_core.scheduler
            orig_schedule = scheduler.schedule

            def schedule(*a, **kw):
                out = orig_schedule(*a, **kw)
                self._sched = out
                # Nested range named by step kind, e.g. "prefill_0" / "decode_3".
                if nvtx and not self._kind_pushed:
                    kind = self._describe(out)["kind"]
                    nvtx.push_range(f"{kind}_{len(self.steps)}", domain="vllm_steps")
                    self._kind_pushed = True
                return out

            scheduler.schedule = schedule
        except AttributeError as e:
            print(f"[steps] scheduler not reachable ({e!r}); step kinds will be unknown")

    def step(self):
        i = len(self.steps)
        step_in_repeat = self._repeat_steps.get(self.repeat, 0)
        if self.repeat == self.profiler_from_repeat and not self.profiling:
            self.llm.start_profile()
            self.profiling = True
        trace_this = self.nvbit_lib is not None and self.trace_steps is not None and (
            not self.trace_steps or i in self.trace_steps)
        if self.trace_steps_in_repeat is not None:
            trace_this = (self.nvbit_lib is not None and self.repeat == self.trace_repeat
                          and (not self.trace_steps_in_repeat or step_in_repeat in self.trace_steps_in_repeat))
        self._sched = None
        self._kind_pushed = False
        sync_start = time.monotonic_ns()
        if self.sync_steps:
            torch.cuda.synchronize()
        pre_sync_ns = time.monotonic_ns() - sync_start if self.sync_steps else 0
        first_id = tracer_next_kernel_id()
        wall_start = time.time_ns()
        monotonic_start = time.monotonic_ns()
        gc_pause_start = self.gc.total_ns
        gc_before = {"count": list(gc.get_count()),
                     "collections": [s["collections"] for s in gc.get_stats()]}
        if nvtx:
            nvtx.push_range(f"vllm_step_{i}", domain="vllm_steps")
        if trace_this:
            self.nvbit_lib.enable_nvbit_instrumentation()
        try:
            out = self._orig_step()
            sync_start = time.monotonic_ns()
            if self.sync_steps:
                torch.cuda.synchronize()
            post_sync_ns = time.monotonic_ns() - sync_start if self.sync_steps else 0
        finally:
            if trace_this:
                self.nvbit_lib.disable_nvbit_instrumentation()
            if nvtx:
                if self._kind_pushed:
                    nvtx.pop_range(domain="vllm_steps")
                nvtx.pop_range(domain="vllm_steps")
        wall_end = time.time_ns()
        monotonic_end = time.monotonic_ns()
        gc_pause_end = self.gc.total_ns
        gc_after = {"count": list(gc.get_count()),
                    "collections": [s["collections"] for s in gc.get_stats()]}
        rec = {"step": i, "repeat": self.repeat, "step_in_repeat": step_in_repeat,
               "warmup": self.repeat == 0,
               "nvtx_range": f"vllm_step_{i}", "traced": trace_this,
               "wall_start_ns": wall_start, "wall_end_ns": wall_end,
               "monotonic_start_ns": monotonic_start, "monotonic_end_ns": monotonic_end,
               "pre_step_sync_ns": pre_sync_ns, "post_step_sync_ns": post_sync_ns,
               "gc_before": gc_before, "gc_after": gc_after,
               "gc_pause_ns_in_step": gc_pause_end - gc_pause_start,
               "gc_pause_ns_in_gap_before": gc_pause_start - self._previous_gc[1]
               if self._previous_gc is not None and self._previous_gc[0] == self.repeat else None,
               "first_kernel_id": first_id, "end_kernel_id": tracer_next_kernel_id()}
        rec.update(self._describe(self._sched))
        self.steps.append(rec)
        self._repeat_steps[self.repeat] = step_in_repeat + 1
        self._previous_gc = (self.repeat, gc_pause_end)
        return out

    @staticmethod
    def _describe(so) -> dict:
        if so is None:
            return {"kind": "unknown"}
        n_tok = dict(getattr(so, "num_scheduled_tokens", {}) or {})
        new_reqs = getattr(so, "scheduled_new_reqs", []) or []
        new_ids = {getattr(r, "req_id", None) for r in new_reqs}
        computed = {getattr(r, "req_id", None): getattr(r, "num_computed_tokens", 0) for r in new_reqs}
        prefill = [r for r, n in n_tok.items() if r in new_ids or n > 1]
        decode = [r for r in n_tok if r not in prefill]
        kind = "prefill" if prefill and not decode else "decode" if decode and not prefill else (
            "mixed" if n_tok else "empty")
        return {
            "kind": kind,
            "num_scheduled_tokens": getattr(so, "total_num_scheduled_tokens", sum(n_tok.values())),
            "num_reqs": len(n_tok),
            "num_prefill_reqs": len(prefill),
            "num_decode_reqs": len(decode),
            "tokens_per_req": n_tok,
            "new_req_num_computed_tokens": computed,
        }

    def write(self):
        if self.log_path:
            Path(self.log_path).write_text(json.dumps(self.steps, indent=2))

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.gc.close()
        atexit.unregister(self.close)
        self.write()


def main():
    # Surface torch_hook's "Found nvbit tracer, attaching hooks to layer ..." messages.
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--model")
    p.add_argument("--layers", default="", help="comma-separated module names (--trace-scope layers)")
    p.add_argument("--prompts", help="JSON list of prompts")
    p.add_argument("--workload-json", help="serialized Workload JSON file")
    p.add_argument("--max-tokens", type=int, default=4)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--list-layers", action="store_true")
    p.add_argument("--trace-scope", choices=("layers", "steps"))
    p.add_argument("--trace-steps", default="",
                   help="comma-separated engine step indices to trace (0 = first step / prefill); "
                        "empty = all steps. Only for --trace-scope steps")
    p.add_argument("--trace-repeat", type=int, default=1)
    p.add_argument("--trace-steps-in-repeat", help="comma-separated repeat-local step indices")
    p.add_argument("--exec-mode", choices=EXEC_MODES, default="eager")
    p.add_argument("--no-async-scheduling", action="store_true",
                   help="disable vLLM async scheduling so each engine step is one forward pass")
    p.add_argument("--step-log", default=os.environ.get("VLLM_STEP_LOG", ""),
                   help="write per-step JSON here (default: $VLLM_STEP_LOG)")
    p.add_argument("--iteration-details", action="store_true")
    p.add_argument("--layerwise-nvtx", action="store_true")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--cuda-profiler-from-repeat", type=int)
    p.add_argument("--metrics-out", default="")
    p.add_argument("--no-step-sync", action="store_true",
                   help="skip StepRecorder CUDA synchronizations (profiling experiment)")
    p.add_argument("--gc-between-repeats", action=argparse.BooleanOptionalAction, default=True,
                   help="collect garbage between generate repeats (default: on)")
    args = p.parse_args()
    if args.trace_scope is None:
        args.trace_scope = "steps" if args.workload_json else "layers"
    if args.workload_json:
        workload = Workload.from_dict(json.loads(Path(args.workload_json).read_text()))
        for field in ("model", "max_tokens", "max_model_len", "exec_mode", "repeats"):
            setattr(args, field, getattr(workload, field))
        args.no_async_scheduling = True
    else:
        if args.model is None or args.prompts is None:
            p.error("--model and --prompts are required without --workload-json")
        workload = Workload(args.model, json.loads(args.prompts), args.max_tokens,
                            args.repeats, args.exec_mode, args.max_model_len)
    driver = workload.resolve_driver()
    if args.trace_steps_in_repeat is not None:
        if args.trace_scope != "steps" or not 0 <= args.trace_repeat < args.repeats:
            p.error("repeat-local tracing requires steps scope and a trace-repeat in [0, repeats)")
        local_steps = {int(s) for s in args.trace_steps_in_repeat.split(",") if s.strip()}
        if any(s < 0 for s in local_steps):
            p.error("repeat-local step indices must be nonnegative")
    if args.layerwise_nvtx and args.exec_mode != "eager":
        p.error("--layerwise-nvtx requires --exec-mode eager")
    if args.repeats < 1:
        p.error("--repeats must be positive")
    if args.cuda_profiler_from_repeat is not None and not (
            0 <= args.cuda_profiler_from_repeat < args.repeats):
        p.error("--cuda-profiler-from-repeat must be in [0, repeats)")
    extra = {}
    if args.iteration_details:
        extra.update(enable_logging_iteration_details=True, disable_log_stats=False)
    if args.metrics_out:
        extra["disable_log_stats"] = False
    if args.layerwise_nvtx:
        extra["enable_layerwise_nvtx_tracing"] = True
    if args.cuda_profiler_from_repeat is not None:
        extra["profiler_config"] = {"profiler": "cuda"}
    mono_before = time.monotonic_ns()
    wall = time.time_ns()
    mono_after = time.monotonic_ns()
    clock = {"step_clock": "monotonic",
             "wall_minus_monotonic_ns": wall - (mono_before + mono_after) // 2,
             "how": "time.time_ns sampled between two time.monotonic_ns calls; midpoint offset",
             "sample_bracket_ns": mono_after - mono_before}

    # Startup: weight load, memory profiling, warmup, torch.compile, CUDA-graph capture.
    if nvtx:
        nvtx.push_range("vllm_init", domain="vllm_steps")
    llm = LLM(**workload.merged_llm_kwargs({**extra, **llm_kwargs(args.exec_mode, args.no_async_scheduling)}))
    torch.cuda.synchronize()
    if nvtx:
        nvtx.pop_range(domain="vllm_steps")

    llm.collective_rpc(lambda self: print_layer_names(self.model_runner.model))
    if args.list_layers:
        return

    trace_steps = None
    if args.trace_scope == "layers":
        layers = [l.strip() for l in args.layers.split(",") if l.strip()]
        if not layers:
            raise SystemExit("--layers is required for --trace-scope layers")
        llm.collective_rpc(lambda self: apply_nvbit_hook(self.model_runner.model, layers))
    else:
        trace_steps = {int(s) for s in args.trace_steps.split(",") if s.strip()}
    recorder = StepRecorder(llm, load_nvbit_tool() if trace_steps is not None else None,
                            trace_steps, args.step_log)
    recorder.profiler_from_repeat = args.cuda_profiler_from_repeat
    recorder.sync_steps = not args.no_step_sync
    if args.trace_steps_in_repeat is not None:
        recorder.trace_repeat = args.trace_repeat
        recorder.trace_steps_in_repeat = local_steps

    # Greedy + ignore_eos: the spinlock-detection passes, the trace pass and the nsys pass must
    # launch the exact same kernel sequence (same number of decode steps), so no randomness.
    sampling_params = SamplingParams(**workload.sampling_params_kwargs())
    requests = []
    repeat_gc = []
    try:
        for repeat in range(args.repeats):
            recorder.repeat = repeat
            gc_pause_before = recorder.gc.total_ns
            collect_ns = 0
            if repeat > 0 and args.gc_between_repeats:
                collect_start = time.monotonic_ns()
                gc.collect()
                collect_ns = time.monotonic_ns() - collect_start
            outputs = driver(llm, repeat, workload) if driver else llm.generate(workload.prompts, sampling_params)
            generate_return_ns = time.monotonic_ns()
            if args.metrics_out:
                for output in outputs:
                    m = asdict(output.metrics) if output.metrics is not None else None
                    row = {"repeat": repeat, "request_id": output.request_id, "metrics": m,
                           "generate_return_monotonic_ns": generate_return_ns}
                    if m:
                        decode = m["last_token_ts"] - m["first_token_ts"]
                        row.update(queue_s=m["scheduled_ts"] - m["queued_ts"],
                                   ttft_s=m["first_token_ts"] - m["scheduled_ts"],
                                   queued_ttft_s=m["first_token_ts"] - m["queued_ts"],
                                   decode_s=decode,
                                   mean_itl_s=decode / (m["num_generation_tokens"] - 1)
                                   if m["num_generation_tokens"] > 1 else None)
                    requests.append(row)
            repeat_gc.append({"repeat": repeat, "gc_collect_before_ns": collect_ns,
                              "total_gc_pause_ns": recorder.gc.total_ns - gc_pause_before})
    finally:
        try:
            if recorder.profiling:
                llm.stop_profile()
        finally:
            recorder.close()
    if args.metrics_out:
        Path(args.metrics_out).write_text(json.dumps({
            "clock": clock, "requests": requests,
            "gc": {"between_repeats": args.gc_between_repeats, "repeats": repeat_gc,
                   "events": recorder.gc.events},
            "metrics": [asdict(m) for m in llm.get_metrics()]}, indent=2))
    for output in outputs:
        print(f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}")

    del llm


if __name__ == "__main__":
    main()
