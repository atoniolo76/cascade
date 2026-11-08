# Cascade team charter

Teammates: Amani Bobo, Alessio Toniolo.

Project: offline simulation of multi-GPU systems.

## Goal

1. Take an application trace from an LLM or other ML workload.
2. Simulate it on other GPU hardware.
3. Scale this to many GPUs beyond the original system it ran on.

## Components

1. **Profiling.** A Python package that profiles the workload on its original hardware, for example a live vLLM inference server handling a request trace from many users. The user passes it the public (or local, in dev) Cascade app endpoint. Under the hood it calls Accel-Sim's tracing hooks. When tracing finishes, it sends the saved trace file to the app's storage.
2. **Cascade app.** A container that runs as a sailbox. It receives the trace over HTTP, saves it to local disk, and serves a UI for managing simulation runs. When it receives a simulation config, it uses the Sail Research API to spin up other sailboxes that fork the current checkpoint, so each fork already has the trace on disk.
   The image has two modes: **main app** mode (launches sandboxes, serves UI) and **simulator** mode (runs the simulation in a forked sandbox). Both modes must be the same image so the checkpoint can be reused.
3. **Simulator.** In simulator mode, the app starts Accel-Sim 2.0 with the trace as input. Accel-Sim writes observability to a running log, which is forwarded back to the main app.

## Ownership

- **Alessio:** profiling and simulator. Sending trace information from the simulator to the app. Sending profile traces to the running app.
- **Amani:** Cascade app. Launching other sailboxes, the REST/CRUD API between main and simulator containers (observability, health), and the UI that shows the running simulation and the original hardware trace.

## Ambitious goal

Once many copies of a simulation run across different hardware, add a coordination layer so the GPUs act as one system. For the inference example: Cascade intercepts the client workload and partitions it evenly across the GPU simulators so the application runs faster.
