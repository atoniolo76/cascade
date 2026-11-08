Real Accel-Sim config files, vendored so the GPU config registry can be tested and used in local dev
(`CASCADE_ACCELSIM_DIR=tests/fixtures/accelsim`). Only the two config files per GPU are kept, not
their companion files, so these cannot drive a real simulation.

- `gpu-simulator/gpgpu-sim/configs/tested-cfgs/<cfg>/gpgpusim.config`: from
  https://github.com/accel-sim/gpgpu-sim_distribution at `4195dd4f229cbf03f4f38c49139ae94d01d23f03`,
  the GPGPU-Sim `dev` commit that Accel-Sim v2.0.1's `setup_environment.sh` cloned (see
  ACCELSIM_NOTES.md on `team-charter`). Accel-Sim keeps these in the GPGPU-Sim repo, not in
  accel-sim-framework itself.
- `gpu-simulator/configs/tested-cfgs/<cfg>/trace.config`: from
  https://github.com/accel-sim/accel-sim-framework at tag `v2.0.1` (`1c9cc831`).

Configs: SM90_H100, SM90_H200, SM80_A100, SM7_QV100, SM86_RTX3070 (the RTX 3070 config matches the
vendored log in `tests/fixtures/logs/`).
