Real Accel-Sim stdout logs, copied byte for byte from public repositories. Neither was written or
edited by hand. Neither upstream repository states a license; they are used here only as parser
test input.

- `babelstream_sm86_rtx3070.log`: trace-driven Accel-Sim (`accelsim-commit-e02c99db`, GPGPU-Sim
  4.2.0) on the SM86_RTX3070 config, 6 kernels. About 17 lines of `setup_environment` output precede
  the banner, and this older build prints `launching kernel name: X uid: N` without
  `cuda_stream_id`.
  Source: https://github.com/romnn/simulators/blob/ccc6f7c522bdde9fd166d275387ce86cde364ad3/run/accelsim-sass/sm86_rtx3070/babelstream/input-arraysize-1024-numtimes-1/results/log.txt
- `synth_lop3_sm90_h100.log`: a recent Accel-Sim build (`accelsim-commit-7ba54de`, the dev commit
  Alessio's tracer pins) on the SM90_H100 config, 1 kernel. The upstream author generated the trace
  with a script rather than NVBit and runs a locally patched simulator, so treat it as real
  simulator output for a synthetic workload.
  Source: https://github.com/emberian/zkml-research/blob/04599ce8b5c4505b5896b07e61b740b46dde62f4/notes/hw-optimality/sim/logs/trh-lop3.log
