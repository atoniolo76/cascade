# Cascade Simulator

CPU simulation and timeline replay for GPU profiling runs. Accel-Sim execution
lives in `cascade.sim`; GPU measurement and tracing live on the `profiler` branch.

- [Profiler → simulator run-directory contract](docs/run-dir-contract.md): file
  ownership, required inputs, schemas, real examples and end-to-end commands.
- [Accel-Sim execution](docs/accelsim.md): CPU API and Modal setup.
- [Corrected timeline simulation](docs/corrected-timeline-simulation.md): replay
  and comparison behavior.
