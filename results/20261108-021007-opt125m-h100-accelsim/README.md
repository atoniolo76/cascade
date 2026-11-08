# OPT-125m on Accel-Sim SM90_H100 (2026-11-08 02:10)

A real Accel-Sim run by Alessio: 20 vLLM kernels of `facebook/opt-125m`, traced with NVBit on an H100 and simulated on the `SM90_H100` config. This is the replay behind `results/20261108-043849/` ("earlier H100 OPT-125m replay `sim_H100/kernel-stats.csv`, UIDs 1–20").

## Inputs

- `kernelslist.g` lists the 20 traced kernels: four groups of five (trace kernel ids 656–660, 833–837, 1010–1014, 1187–1191). Per `results/20261108-043849/README.md`, UIDs 1–5 are prefill-like and UIDs 6–20 decode-like.
- `configs/` holds `gpgpusim.config`, `trace.config` and `config_hopper_islip.icnt`. The core clock is 1980 MHz across 132 clusters (`-gpgpu_clock_domains 1980:3960:3960:3106`).
- `run.sh` is the exact command: Accel-Sim v2.0.1 (CMake build), run from `configs/` because the interconnect file is read from the working directory.
- The 20 `.tracez` files (16 MB) are not in git. They are uploaded to Cascade as trace `20261108-021007__facebook_opt-125m__h100-accelsim`.

## Outputs

| File | Contents |
|---|---|
| `outputs/run.log` | Full Accel-Sim stdout, 79,110 lines, GPGPU-Sim 4.2.0 (commit `4195dd4f`). Also at `tests/fixtures/logs/opt125m_sm90_h100.log`. |
| `outputs/kernel-stats.csv` | `uid,name,gpu_sim_cycle,gpu_sim_insn,ipc` for all 20 kernels; 169,686 cycles in total. |
| `outputs/opt125m_H100_perfetto.json.gz` | The kernels as a Chrome/Perfetto trace in simulated time. Durations include the configured launch latency. |
| `outputs/perf_counter.csv.gz` | Per-interval L2 bank counters. |
| `outputs/gpgpu_inst_stats.txt` | Per-kernel instruction-latency stats header. |
| `outputs/exit-status.txt` | `0`. |
| `outputs/setup.log` | `setup_environment.sh` output. |

## How Cascade uses it

- `cascade worker --replay tests/fixtures/logs/opt125m_sm90_h100.log` is the stand-in simulator on Sail. sm_90 traces get real H100 numbers for real vLLM kernels instead of the BabelStream log.
- Its UIDs and names match `tests/fixtures/timeline/mapping.json` exactly, so the worker's step timing covers all 20 mapped kernels. `tests/test_local_loop.py` checks this end to end.
