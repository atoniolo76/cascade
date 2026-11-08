#!/bin/bash
# Run from this inputs/ directory. ACCELSIM = path to accel-sim-framework/gpu-simulator (v2.0.1, CMake build).
set -e
cd "$(dirname "$0")/configs"   # config_hopper_islip.icnt must be in the working directory
source "$ACCELSIM/setup_environment.sh"
"$ACCELSIM/bin/release/accel-sim.out" \
  -trace ../traces/kernelslist.g \
  -config gpgpusim.config \
  -config trace.config > run.log 2>&1
