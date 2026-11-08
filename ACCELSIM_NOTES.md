# Accel-Sim 2.0 ready-to-run VM

## Installed version and architecture

- Checkout: `/home/ubuntu/repos/accel-sim-framework`, tag **v2.0.1**, commit `1c9cc831c5d948f89fa9dc8f21f6c87952f5f2ea`. This is the newest 2.0 patch-release tag, rather than the moving `dev` branch.
- Setup automatically cloned GPGPU-Sim `dev` at `4195dd4f229cbf03f4f38c49139ae94d01d23f03` and pybind11. The framework setup defaults to this moving GPGPU-Sim branch; preserve this existing checkout to reproduce the build.
- Execution is trace-driven: NVBit observes actual SASS execution on NVIDIA hardware; the CPU simulator replays instruction/memory traces through the GPGPU-Sim timing model to produce simulated cycles, IPC, cache/memory statistics, etc. This VM needs no GPU to replay traces. It cannot collect fresh traces without real NVIDIA hardware.

## Installation actually performed

Installed NVIDIA's Ubuntu 22.04 CUDA apt keyring and `cuda-toolkit-12-8` (nvcc 12.8.93), toolkit only, no GPU driver. Apt dependencies: `build-essential xutils-dev bison zlib1g-dev flex libglu1-mesa-dev libssl-dev libxml2-dev libboost-all-dev libzstd-dev python3-setuptools python3-pip cmake`. Installed Python dependencies with `python3 -m pip install -r requirements.txt`.

```bash
cd /home/ubuntu/repos/accel-sim-framework
export CUDA_INSTALL_PATH=/usr/local/cuda-12.8
export PATH="$CUDA_INSTALL_PATH/bin:$PATH"
source gpu-simulator/setup_environment.sh
cmake -S gpu-simulator -B gpu-simulator/build/release
cmake --build gpu-simulator/build/release -j8
cmake --install gpu-simulator/build/release
```

The initially requested legacy `make -j8 -C gpu-simulator` failed with `fatal error: tracez_format.h: No such file or directory`. The current README/CI CMake build above succeeded without source modifications. Optional Python stub generation printed `stubgen: not found` (mypy is not in requirements); the build explicitly tolerates that, and the simulator executable and Python extension built successfully. Logs are `/home/ubuntu/repos/accelsim-*.log`.

## Verified sample run

Downloaded the CI's V100 sample archive from https://engineering.purdue.edu/tgrogers/accel-sim/traces/tesla-v100/latest/rodinia_2.0-ft.tgz (about 21 MiB), extracted into `hw_run/`. With the environment above, ran:

```bash
gpu-simulator/bin/release/accel-sim.out \
  -trace /home/ubuntu/repos/accel-sim-framework/hw_run/rodinia_2.0-ft/9.1/nn-rodinia-2.0-ft/__data_filelist_4_3_30_90___data_filelist_4_3_30_90_result_txt/traces/kernelslist.g \
  -config gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM7_QV100/gpgpusim.config \
  -config gpu-simulator/configs/tested-cfgs/SM7_QV100/trace.config \
  > /home/ubuntu/repos/accelsim-verify-direct.log 2>&1
```

Completed all four nearest-neighbor kernels, exit status 0: `gpu_tot_sim_cycle = 29962`, `gpu_tot_sim_insn = 6714538`, simulation time 16 seconds, `GPGPU-Sim: *** exit detected ***`.

Also submitted the same app via the local launcher (no Slurm required):

```bash
python3 util/job_launching/run_simulations.py \
  -B rodinia_2.0-ft:nn-rodinia-2.0-ft -C QV100-SASS \
  -T /home/ubuntu/repos/accel-sim-framework/hw_run/rodinia_2.0-ft/9.1 \
  -N setup-smoke -l local -c 1
```

Launcher output: `/home/ubuntu/repos/accelsim-verify-launcher.log`. Run files/results: `sim_run_12.8/nn-rodinia-2.0-ft/`. The local launch also completed all four kernels with `gpu_tot_sim_cycle = 29962` and `GPGPU-Sim: *** exit detected ***` in its `.o1` output. `-T` points at the app-layout root, not the individual trace directory; `-B` must match app YAML definitions under `util/job_launching/apps/`.

## Simulating your trace (next input needed)

Provide a trace archive or downloadable location **and the GPU/configuration to model** (and the GPU which collected the trace, CUDA/tracer version if known). Ideally provide a small app or selected kernel first. A binary or PTX file alone is not a SASS trace.

The processed directory must contain `kernelslist.g` and every `kernel-*.traceg` (legacy text) or `kernel-*.tracez` (2.0 compressed) file referenced by it. Keep their names and directory structure. Include any additional referenced trace/context/cluster sidecar files. Do not mix raw `.trace` files with processed lists.

```bash
cd /home/ubuntu/repos/accel-sim-framework
export CUDA_INSTALL_PATH=/usr/local/cuda-12.8
export PATH="$CUDA_INSTALL_PATH/bin:$PATH"
source gpu-simulator/setup_environment.sh
# Replace /absolute/path/to/traces and SM7_QV100 with the chosen trace/config.
gpu-simulator/bin/release/accel-sim.out \
  -trace /absolute/path/to/traces/kernelslist.g \
  -config gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM7_QV100/gpgpusim.config \
  -config gpu-simulator/configs/tested-cfgs/SM7_QV100/trace.config \
  > /home/ubuntu/repos/user-trace-simulation.log 2>&1
```

Hardware configs: `gpu-simulator/gpgpu-sim/configs/tested-cfgs/`; matching trace frontend configs: `gpu-simulator/configs/tested-cfgs/`. Examples: `SM7_QV100`, `SM80_A100`, `SM90_H100`, `SM90_H200`. Launcher aliases such as `QV100-SASS`/`H200-SASS` live in `util/job_launching/configs/define-standard-cfgs.yml`. Architecture compatibility matters; do not blindly replay Hopper instructions on a Volta model. Configs with relative auxiliary paths/power XML may require copying their companion files into the run directory (the launcher handles this).

## Producing or processing traces

On a separate NVIDIA GPU host with compatible CUDA/driver, follow `util/tracer_nvbit/README.md` and the main README:

```bash
./util/tracer_nvbit/install_nvbit.sh
make -C util/tracer_nvbit
NVBIT_INSTRUMENTATION_ENABLED=1 TRACES_FOLDER=./my_traces \
  ./util/tracer_nvbit/others/torch_hook/run.sh ./your_cuda_app [args...]
./util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing ./my_traces -j 8
```

For benchmark YAML apps, `util/tracer_nvbit/run_hw_trace.py -B rodinia_2.0-ft -D 0` automates collection and post-processing. The postprocessor accepts a raw `kernelslist` file or its directory, emits `kernelslist.g` plus compressed `.tracez` by default; `--text` requests `.traceg`. Raw-trace processing and NVBit tracing have not been run on this GPU-less VM; the tracer itself has not been built here.

The release also provides large H200 ML traces: `python3 get-accel-sim-traces.py -l` successfully lists them. None downloaded (smallest ~802 MB; all together ~208 GB).
