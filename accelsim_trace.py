"""Generate Accel-Sim 2.0 SASS traces of a vLLM model on a Modal H100, streamed to S3.

Follows the "Tracing LLMs with vLLM" section of the accel-sim-framework README:
  1. build NVBit + the tracer (tracer_tool.so) and install vLLM
  2. hook NVBit onto selected layers via llm.collective_rpc (see vllm_trace.py)
  3. run the script through util/tracer_nvbit/others/torch_hook/run.sh, preceded by the
     two-phase spinlock detection that run_hw_trace.py does (required for mark_region)
  4. post-process the raw traces into .tracez + kernelslist.g -- done *off* Modal (see below)

Traces go to an S3-compatible bucket while tracing is still running. The tracer writes one
`kernel-N-ctx_*.trace.xz` per kernel and fully closes it before the next kernel launches, so
a background thread uploads each finished kernel file (streaming multipart, constant RAM) and
deletes it locally. Local disk only ever holds the in-progress kernel plus any upload backlog.

Post-processing is skipped on Modal: it loads a whole kernel trace into RAM and needs every
file locally, so run it where the traces land (it's a CPU-only tool):
  ./util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing <dir>/traces -j 8

S3 config is passed on the command line (--s3-endpoint, --s3-bucket, --s3-prefix). Credentials
are not wired up yet: if AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY happen to be set in the
container they're used, otherwise requests are sent unsigned.

Usage:
  modal run accelsim_trace.py::main --list-layers
  modal run accelsim_trace.py::main --layers model.decoder.layers.10.self_attn --max-tokens 4
      # -> accelsim-traces volume (post-processed); add --s3-endpoint/--s3-bucket to stream to S3
  modal run accelsim_trace.py::main ... --spinlock-handling none   # skip spinlock detection
  modal run accelsim_trace.py::main ... --trace-scope steps [--trace-steps 0,1] [--exec-mode compile]
  modal run accelsim_trace.py::nsys_main --trace-run <run> [--exec-mode graphs]
  modal run accelsim_trace.py::microbench_main
  (see PROFILER_RUNS.md)
Objects land at s3://<bucket>/<prefix>/<run_name>/{traces/,spinlock_detection/,*.log,metadata.json}
Without --s3-bucket the run dir goes to the accelsim-traces Modal volume at /<run_name>/ instead,
post-processed on Modal (kernelslist.g + .tracez) unless --no-post-process.
"""

import json
from pathlib import Path

import modal

from cascade_profiler import Workload, legacy, profile
from cascade_profiler.modal_support import image as profiler_image
from accelsim_local import accelsim_image, simulate_accelsim

ACCELSIM_REPO = "https://github.com/accel-sim/accel-sim-framework.git"
ACCELSIM_COMMIT = "7ba54ded60c4ba0083410c37d22d719166acd6dc"  # dev branch, 2026-10-27
ACCELSIM_DIR = "/opt/accel-sim-framework"
POST_PROCESS_BIN = (
    f"{ACCELSIM_DIR}/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing"
)
TRACER_SO = f"{ACCELSIM_DIR}/util/tracer_nvbit/tracer_tool/tracer_tool.so"
SPINLOCK_SO = f"{ACCELSIM_DIR}/util/tracer_nvbit/others/spinlock_tool/spinlock_tool.so"
TRACE_SCRIPT = "/root/vllm_trace.py"
MICROBENCH_SCRIPT = "/root/kernel_microbench.py"
EXEC_MODES = ("eager", "compile", "graphs")
FUNCTION_TIMEOUT_S = 4 * 60 * 60  # NVBit instrumentation is slow
# Local scratch for the in-progress kernel + upload backlog. Modal's max is 3 TiB; disk is billed
# by raising the memory request 20:1 (512 GiB -> 25.6 GiB, below the 64 GiB requested anyway).
EPHEMERAL_DISK_MIB = 512 * 1024


image = (
    profiler_image(accelsim_commit=ACCELSIM_COMMIT, arch="all")
    .add_local_file(Path(__file__).parent / "vllm_trace.py", TRACE_SCRIPT)
    .add_local_file(Path(__file__).parent / "kernel_microbench.py", MICROBENCH_SCRIPT)
)

app = modal.App("accelsim-vllm-trace", image=image)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
traces_vol = modal.Volume.from_name("accelsim-traces", create_if_missing=True)
TRACES_MOUNT = "/traces"


@app.function(image=accelsim_image(), cpu=32, memory=64 * 1024, timeout=24 * 60 * 60,
              volumes={TRACES_MOUNT: traces_vol})
def simulate(run_name: str, gpu_config: str | None = None, propagate: str = "none",
             dedup: bool = True, verify_dedup: int = 0):
    if not run_name or Path(run_name).name != run_name or run_name in {".", ".."}:
        raise ValueError("run_name must be a single directory name")
    traces_vol.reload()
    try:
        return simulate_accelsim(Path(TRACES_MOUNT) / "phases" / run_name, gpu_config=gpu_config,
                                 jobs=32, propagate=propagate, dedup=dedup, verify_dedup=verify_dedup)
    finally:
        traces_vol.commit()


@app.function(gpu="H100!", timeout=FUNCTION_TIMEOUT_S, cpu=8, memory=64 * 1024,
              ephemeral_disk=EPHEMERAL_DISK_MIB,
              volumes={TRACES_MOUNT: traces_vol, "/root/.cache/huggingface": hf_cache_vol})
def profile_e2e(workload_dict: dict, run_name: str | None, trace_steps: list[int] | str | None, trace_repeat: int = 1):
    try:
        result = profile(Workload.from_dict(workload_dict), Path(TRACES_MOUNT) / "phases", run_name=run_name,
                         trace_steps=trace_steps, trace_repeat=trace_repeat)
        return {"run_name": result.run_dir.name, "stage_wall_s": result.stage_wall_s}
    finally:
        traces_vol.commit()


@app.local_entrypoint()
def e2e_main(model: str = "facebook/opt-125m", prompt: str = "Hello, my name is", max_tokens: int = 32,
             repeats: int = 4, trace_steps: str = "0,5", trace_repeat: int = 1,
             run_name: str = "", timing_only: bool = False, gpu: str = "H100!",
             gpu_config: str = "", propagate: str = "none", dedup: bool = True, verify_dedup: int = 0,
             profile_only: bool = False):
    selection = None if timing_only else "plan" if trace_steps == "plan" else [int(i.strip()) for i in trace_steps.split(",") if i.strip()]
    result = profile_e2e.with_options(gpu=gpu).remote(Workload(model, [prompt], max_tokens=max_tokens, repeats=repeats).to_dict(),
                                run_name or None, selection, trace_repeat)
    print("Profile stage wall seconds:", result["stage_wall_s"])
    if timing_only:
        print("Timing-only run:", result["run_name"], "(NVBit and simulation skipped)")
        return
    print("Profile run:", result["run_name"])
    if profile_only:
        return
    summary = simulate.remote(result["run_name"], gpu_config=gpu_config or None, propagate=propagate,
                              dedup=dedup, verify_dedup=verify_dedup)
    print("Simulation wall seconds:", summary["wall_s"], "coverage:", summary["coverage"])
    print("Simulation counts:", summary["counts"], "provenance labels:", summary["provenance_labels"])


# --------------------------------------------------------------------------------------
# Modal function
# --------------------------------------------------------------------------------------
@app.function(
    gpu="H100!",  # "!" pins H100; plain "H100" may be upgraded to H200, which would mis-tag traces
    timeout=FUNCTION_TIMEOUT_S,
    cpu=8,
    memory=64 * 1024,
    ephemeral_disk=EPHEMERAL_DISK_MIB,
    volumes={"/root/.cache/huggingface": hf_cache_vol, TRACES_MOUNT: traces_vol},
)
def trace_vllm(
    model: str = "facebook/opt-125m",
    layers: list[str] | None = None,
    prompts: list[str] | None = None,
    max_tokens: int = 4,
    max_model_len: int = 2048,
    run_name: str | None = None,
    list_layers: bool = False,
    spinlock_handling: str = "mark_region",
    s3_endpoint: str = "",
    s3_bucket: str = "",
    s3_prefix: str = "accelsim-traces",
    s3_region: str = "auto",
    s3_addressing_style: str = "auto",
    trace_scope: str = "layers",
    trace_steps: list[int] | None = None,
    exec_mode: str = "eager",
    post_process: bool = True,
) -> str | None:
    try:
        return legacy.trace_vllm(
            model=model, layers=layers or ["model.decoder.layers.10.self_attn"], prompts=prompts, max_tokens=max_tokens, max_model_len=max_model_len,
            run_name=run_name, list_layers=list_layers, spinlock_handling=spinlock_handling,
            s3_endpoint=s3_endpoint, s3_bucket=s3_bucket, s3_prefix=s3_prefix, s3_region=s3_region,
            s3_addressing_style=s3_addressing_style, trace_scope=trace_scope, trace_steps=trace_steps,
            exec_mode=exec_mode, post_process=post_process, out_dir=Path(TRACES_MOUNT), work_root=Path("/tmp/accelsim"),
            accelsim_root=ACCELSIM_DIR, trace_script=TRACE_SCRIPT, output_uri="modal-volume://accelsim-traces")
    finally:
        traces_vol.commit()


# --------------------------------------------------------------------------------------
# Real-hardware kernel timing (Nsight Systems), matched to the tracer's kernel ids
# --------------------------------------------------------------------------------------
NSYS_TIMEOUT_S = 60 * 60


@app.function(
    gpu="H100!",
    timeout=NSYS_TIMEOUT_S,
    cpu=8,
    memory=64 * 1024,
    volumes={"/root/.cache/huggingface": hf_cache_vol, TRACES_MOUNT: traces_vol},
)
def profile_nsys(trace_run: str, exec_mode: str = "", lock_clocks_mhz: int = 0,
                 gpu_metrics: bool = False, out_name: str = "",
                 keep_nsys_rep: bool = False) -> dict:
    try:
        return legacy.profile_nsys(
            trace_run=trace_run, exec_mode=exec_mode, lock_clocks_mhz=lock_clocks_mhz,
            gpu_metrics=gpu_metrics, out_name=out_name, keep_nsys_rep=keep_nsys_rep,
            out_dir=Path(TRACES_MOUNT), work_root=Path("/tmp/nsys"), accelsim_root=ACCELSIM_DIR, trace_script=TRACE_SCRIPT)
    finally:
        traces_vol.commit()


# --------------------------------------------------------------------------------------
# Launch-overhead calibration microbenchmark (no tracer)
# --------------------------------------------------------------------------------------
@app.function(
    gpu="H100!",
    timeout=NSYS_TIMEOUT_S,
    cpu=8,
    memory=32 * 1024,
    volumes={TRACES_MOUNT: traces_vol},
)
def microbench_nsys(n: int = 200, tokens: str = "1,6", ops: str = "", run_name: str = "",
                    keep_nsys_rep: bool = False) -> dict:
    try:
        return legacy.microbench_nsys(
            n=n, tokens=tokens, ops=ops, run_name=run_name, keep_nsys_rep=keep_nsys_rep,
            out_dir=Path(TRACES_MOUNT), work_root=Path("/tmp/microbench"), microbench_script=MICROBENCH_SCRIPT)
    finally:
        traces_vol.commit()


@app.function(
    gpu="H100!", timeout=NSYS_TIMEOUT_S, cpu=8, memory=64 * 1024,
    volumes={"/root/.cache/huggingface": hf_cache_vol, TRACES_MOUNT: traces_vol},
)
def profile_phases(model: str = "facebook/opt-125m", prompt: str = "Hello, my name is",
                   max_tokens: int = 32, repeats: int = 4, max_model_len: int = 2048,
                   exec_mode: str = "eager", phase: str = "both", layerwise_nvtx: bool = False,
                   run_name: str = "", keep_nsys_rep: bool = False, no_step_sync: bool = False,
                   gc_between_repeats: bool = True) -> dict:
    if phase not in {"baseline", "profiled", "both"}:
        raise ValueError("phase must be baseline, profiled, or both")
    workload = Workload(model, [prompt], max_tokens, repeats, exec_mode, max_model_len)
    try:
        result = profile(workload, Path(TRACES_MOUNT) / "phases", run_name=run_name or None,
                         phases=("baseline", "profiled") if phase == "both" else (phase,),
                         layerwise_nvtx=layerwise_nvtx, no_step_sync=no_step_sync,
                         gc_between_repeats=gc_between_repeats, keep_nsys_rep=keep_nsys_rep,
                         accelsim_root=ACCELSIM_DIR)
        return json.loads(result.paths["reconcile.json"].read_text())
    finally:
        traces_vol.commit()


@app.local_entrypoint()
def phases_main(model: str = "facebook/opt-125m", prompt: str = "Hello, my name is",
                max_tokens: int = 32, repeats: int = 4, max_model_len: int = 2048,
                exec_mode: str = "eager", phase: str = "both", layerwise_nvtx: bool = False,
                run_name: str = "", keep_nsys_rep: bool = False, no_step_sync: bool = False,
                gc_between_repeats: bool = True):
    s = profile_phases.remote(model=model, prompt=prompt, max_tokens=max_tokens, repeats=repeats,
                              max_model_len=max_model_len, exec_mode=exec_mode, phase=phase,
                              layerwise_nvtx=layerwise_nvtx, run_name=run_name,
                              keep_nsys_rep=keep_nsys_rep, no_step_sync=no_step_sync,
                              gc_between_repeats=gc_between_repeats)
    print("kind       A wall mean(ns) B wall mean(ns) delta(%) A elapsed(ms) B elapsed(ms)")
    for r in s["table"]:
        print(r["kind"], *(r[k] for k in ("baseline_wall_ns_mean", "profiled_wall_ns_mean",
              "delta_wall_ns_mean_pct", "baseline_elapsed_ms_mean", "profiled_elapsed_ms_mean")))
    print(f"Saved to accelsim-traces:/phases/{s['run_name']}/\n"
          f"modal volume get accelsim-traces phases/{s['run_name']} ./phases")


@app.local_entrypoint()
def microbench_main(n: int = 200, tokens: str = "1,6", ops: str = "", run_name: str = ""):
    """modal run accelsim_trace.py::microbench_main [--n 200 --tokens 1,6 --ops fill,copy]"""
    s = microbench_nsys.remote(n=n, tokens=tokens, ops=ops, run_name=run_name)
    print(f"\nSaved to volume accelsim-traces:/microbench/{s['run_name']}/  "
          f"(modal volume get accelsim-traces microbench/{s['run_name']} ./microbench)")
    print(f"{'range':45s} {'kern':>5s} {'dur_med':>9s} {'s2s_med':>9s} {'api_med':>9s}")
    for r in s["ranges"]:
        print(f"{r['range']:45s} {r['kernels']:5d} {r['dur_median_ns']!s:>9} "
              f"{r['start_to_start_median_ns']!s:>9} {r['api_median_ns']!s:>9}")


@app.local_entrypoint()
def nsys_main(trace_run: str = "20261108-004021__facebook_opt-125m__model.decoder.layers.10.self_attn",
              exec_mode: str = "", lock_clocks_mhz: int = 0, gpu_metrics: bool = False,
              out_name: str = ""):
    """modal run accelsim_trace.py::nsys_main --trace-run <run dir on the accelsim-traces volume>
    [--exec-mode graphs] [--lock-clocks-mhz 1980] [--gpu-metrics]"""
    s = profile_nsys.remote(trace_run, exec_mode=exec_mode, lock_clocks_mhz=lock_clocks_mhz,
                            gpu_metrics=gpu_metrics, out_name=out_name)
    print(f"\nSaved to volume accelsim-traces:/{trace_run}/{s['out_name']}/  "
          f"(modal volume get accelsim-traces {trace_run}/{s['out_name']} ./{s['out_name']})")
    for st in s["per_step"]:
        print(f"  step {st['step']} {st['kind']:8s} span={st['span_ns']}ns busy={st['gpu_busy_ns']}ns "
              f"nsys={st['nsys_kernels']} tracer={st['tracer_kernels']} matched={st['matched_in_step']}")
    durs = s["traced_kernel_durations_ns"]
    if durs and len(durs) <= 50:
        print("Real durations of the traced kernels (ns):")
        for kid, d in sorted(durs.items(), key=lambda kv: int(kv[0])):
            print(f"  kernel-{kid}: {d}")


@app.local_entrypoint()
def main(
    model: str = "facebook/opt-125m",
    layers: str = "model.decoder.layers.10.self_attn",
    prompt: str = "Hello, my name is",
    max_tokens: int = 4,
    max_model_len: int = 2048,
    run_name: str = "",
    list_layers: bool = False,
    spinlock_handling: str = "mark_region",
    s3_endpoint: str = "",
    s3_bucket: str = "",
    s3_prefix: str = "accelsim-traces",
    s3_region: str = "auto",
    s3_addressing_style: str = "auto",
    trace_scope: str = "layers",
    trace_steps: str = "",
    exec_mode: str = "eager",
    post_process: bool = True,
):
    result = trace_vllm.remote(
        model=model,
        layers=[l.strip() for l in layers.split(",") if l.strip()],
        prompts=[prompt],
        max_tokens=max_tokens,
        max_model_len=max_model_len,
        run_name=run_name or None,
        list_layers=list_layers,
        spinlock_handling=spinlock_handling,
        s3_endpoint=s3_endpoint,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        s3_region=s3_region,
        s3_addressing_style=s3_addressing_style,
        trace_scope=trace_scope,
        trace_steps=[int(s) for s in trace_steps.split(",") if s.strip()],
        exec_mode=exec_mode,
        post_process=post_process,
    )
    if result and not s3_bucket:
        run = result.rsplit("/", 1)[1]
        print(f"\nDone. Run saved to volume accelsim-traces:/{run}/\n"
              f"  modal volume get accelsim-traces {run} ./{run}\n"
              f"  modal run accelsim_trace.py::nsys_main --trace-run {run}")
    elif result:
        print(f"\nDone. Traces at {result}")
        print("Next, wherever you download them (CPU-only):\n"
              "  post-traces-processing <run>/traces -j 8\n"
              "  accel-sim.out -config .../SM90_H100/gpgpusim.config "
              "-config .../SM90_H100/trace.config -trace <run>/traces/kernelslist.g")
