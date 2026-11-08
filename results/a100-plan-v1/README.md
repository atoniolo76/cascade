# Native A100: complete Modal reference and secondary partial Sail experiment

This is a new **native A100** run, not Hopper SASS replayed under an older
configuration. See [the detailed analysis](../../docs/sample-analysis-opt125m.md)
and the generated [complete Modal tables](modal-complete/tables.md).

**Authoritative result:** real mean E2E **222.083118 ms**, complete native
SM80_A100 mean **229.580065 ms**, **+3.38%**. The complete run has zero
failures/timeouts and 100% kernel coverage (98.26% measured GPU-time coverage).
Its 33 representative executions took **565.086033 s** orchestration wall.

The **secondary Sail run is partial**, not the headline result: three actual
lm_head simulations hit the 900-s cap. Its replay retains measured lm_head
durations, so its smaller E2E error is not comparable as a complete prediction.
See [secondary tables](sail-partial/tables.md) and its saved failed app state.

## Inputs and execution

* Profiler checkout: `f190c99ac31f9241324a817424e55cc7762020fb`, including the
  standalone local-disk optimization `0b54e42`.
* Modal GPU: `A100-80GB`; measured device `NVIDIA A100-SXM4-80GB`; vLLM 0.21.0.
* Model `facebook/opt-125m`, prompt `Hello, my name is`, max_tokens 32,
  repeats 4, eager, trace_steps=plan, trace_repeat=1. Selected local steps
  0/2/10/26 correspond to contexts 6/8/16/32.
* Modal run: `e2e-a100-plan-v1`, app `ap-wUfWWPC7g965egC46n3mul`.
* Full 1.4-GiB run: `/home/ubuntu/runs/phases/e2e-a100-plan-v1` and Modal
  volume `accelsim-traces`, prefix `phases/e2e-a100-plan-v1`.
* Native config `SM80_A100`, core clock 1410 MHz from the saved
  [shipped config snapshot](sail-partial/gpgpusim.config), not the profiled clock
  median (1155 MHz).
  This is the shipped config, not frequency-calibrated to the measured GPU.
* Accel-Sim `7ba54ded60c4ba0083410c37d22d719166acd6dc`, GPGPU-Sim
  `4195dd4f229cbf03f4f38c49139ae94d01d23f03`.
* The complete reference was collected from Modal volume `accelsim-traces`,
  prefix **`hw-sweep/e2e-a100-plan-v1-sm80-v1/`**, as requested by the lead.
  It ran 33 representatives with verification disabled: 33 exact, 591 dedup,
  14,100 identical propagated records. No new Modal simulation was launched
  for this update. Its source baseline matched the real A100 timeline byte-for-byte.
* [drive_sail.py](drive_sail.py) uses the repository Dockerfile and
  `scripts/deploy_sail.py`: the main app runs **inside Sail**, with
  `CASCADE_LAUNCHER=sail`. `/api/runs` checkpoints/forks the simulator sailbox.
* [sail_worker.py](sail_worker.py) executes real CPU Accel-Sim with eight
  processes, dedup enabled, **verify_dedup=2**, per-kernel timeout 900 s,
  propagation `interp`. Artifacts are uploaded before the terminal heartbeat.
  Outputs are isolated under `sail-partial/`, never merged into the complete run.

Sail made 99 actual executions (96 completed, three lm_head timeouts), with
620/624 aligned occurrences successful and four aligned timeouts after dedup
expansion. Wall was **1124.049524 s**. Secondary coverage is 14,628/14,724
kernels and 92.86% measured GPU time; all 96 steady lm_head durations remain
measured. Verification found **39.378776% / 4171 cycles** disagreement in a
GEMVX group shared by inferred out-projection and MLP-down roles. The complete
Modal run did not perform these verification copies.

No secrets are stored here. The Sail key was bound by credential-store
reference and the app bearer token was generated in-shell without printing it.
The first main-image build encountered a GitHub Python-download HTTP 504;
[the exact error log](sail-partial/sail-deploy-first-attempt.log) is retained. Retrying the
same deployment succeeded without modifying the Dockerfile.

## Profiling evidence

`profile/*.json.gz` and `profile/traced/trace_steps.json.gz` are compressed
copies of the profiler's real JSON artifacts, including the complete corrected
timeline and alignment. They are **not a complete runnable trace directory**:
raw instruction traces remain in the run store. Decompress a saved JSON with
`gzip -dc profile/run.json.gz`; no extraction protocol was added to the profiler.

There are 624 aligned occurrences, zero unmatched records, 624 `.trace.xz`
and 624 `.tracez` files. NVBit and post-processing used `/tmp`; collection to
the volume took 3.912637 s. Instrumented envelopes were **625.565 s prefill**
and **869.279/813.347/786.332 s decode**, not the earlier 52-s anecdote.
Different GPU/native kernels prevent a controlled local-disk speedup claim.

## Reproduction

In the profiler checkout (Modal credentials already bound):

```sh
modal run accelsim_trace.py::e2e_main --run-name e2e-a100-plan-v1 \
  --gpu A100-80GB --model facebook/opt-125m --prompt 'Hello, my name is' \
  --max-tokens 32 --repeats 4 --trace-steps plan --trace-repeat 1
# LOCAL_DESTINATION must already be a directory with this Modal CLI version.
modal volume get accelsim-traces /phases/e2e-a100-plan-v1 /home/ubuntu/runs/phases
```

In this checkout, with `SAIL_API_KEY` already bound by reference:

```sh
export CASCADE_TOKEN=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')
uv run --extra sail python results/a100-plan-v1/drive_sail.py
# Replay the authoritative complete reference, not the partial Sail durations.
RUN=/home/ubuntu/runs/phases/e2e-a100-plan-v1
SIM=results/a100-plan-v1/modal-complete
gzip -dk "$SIM/sim_durations.json.gz"
uv run cascade simulate "$RUN/corrected_timeline.json" \
  --durations "$SIM/sim_durations.json" --output "$SIM/simulated_timeline.json"
uv run cascade compare "$RUN/corrected_timeline.json" \
  "$SIM/simulated_timeline.json" --output "$SIM/compare.json"
uv run python results/a100-plan-v1/report.py
uv run python results/a100-plan-v1/report.py --sim-dir results/a100-plan-v1/sail-partial
uv run pytest -q results/a100-plan-v1/test_report.py
```

Large simulation JSON files may be gzip-compressed in the committed results;
decompress them before using the CLI, which consumes plain JSON. `report.py`
reads either form. Matrix role labels are inferred from the 12 OPT layers'
ordered QKV/out/up/down operations and the final vocabulary projection; the
report rejects other matrix layouts rather than applying Hopper grid rules.

See [sail-resources.json](sail-partial/sail-resources.json) for created/terminated sailbox
and deleted checkpoint IDs. The empty app grouping remains because this SDK
does not expose app deletion; it is not running compute.
