"""Launch-overhead vs execution-time calibration for small kernels (run under nsys).

For each op (shapes taken from facebook/opt-125m: hidden 768, 12 heads x 64, fp16) and each
token count, launches the op N times back-to-back in two modes, each inside its own NVTX
range "<op>|tokens=<t>|<mode>":
  eager  N plain launches, no sync in between (CPU launch-bound if the kernel is tiny)
  graph  the same N launches captured once into a CUDA graph, then replayed (GPU-bound:
         kernel-to-kernel gaps are pure device-side scheduling)
nsys (--cuda-graph-trace=node) then gives per-kernel GPU durations and gaps for both.
"""

import argparse

import nvtx
import torch

HIDDEN, HEADS, HEAD_DIM, BLOCK = 768, 12, 64, 16
DT = torch.float16


def make_ops(tokens: int) -> dict:
    dev = "cuda"
    x = torch.randn(tokens, HIDDEN, device=dev, dtype=DT)
    w = torch.randn(HIDDEN, HIDDEN, device=dev, dtype=DT)
    w_qkv = torch.randn(3 * HIDDEN, HIDDEN, device=dev, dtype=DT)
    ln_w = torch.ones(HIDDEN, device=dev, dtype=DT)
    ln_b = torch.zeros(HIDDEN, device=dev, dtype=DT)
    buf = torch.empty(tokens, HIDDEN, device=dev, dtype=DT)
    ops = {
        "fill": lambda: buf.fill_(1.0),
        "copy": lambda: buf.copy_(x),
        "add": lambda: torch.add(x, x, out=buf),
        "layer_norm": lambda: torch.nn.functional.layer_norm(x, (HIDDEN,), ln_w, ln_b),
        "relu": lambda: torch.relu(x),
        "gemm_768x768": lambda: torch.mm(x, w.t()),
        "gemm_qkv_768x2304": lambda: torch.mm(x, w_qkv.t()),
    }
    try:
        from vllm import _custom_ops as vops

        k = torch.randn(tokens, HEADS, HEAD_DIM, device=dev, dtype=DT)
        v = torch.randn_like(k)
        kc = torch.zeros(64, BLOCK, HEADS, HEAD_DIM, device=dev, dtype=DT)
        vc = torch.zeros_like(kc)
        slots = torch.arange(tokens, device=dev, dtype=torch.int64)
        one = torch.ones((), device=dev, dtype=torch.float32)
        ops["reshape_and_cache_flash"] = lambda: vops.reshape_and_cache_flash(
            k, v, kc, vc, slots, "auto", one, one)
    except Exception as e:  # vLLM's custom ops unavailable -> skip only that op
        print(f"skipping reshape_and_cache_flash: {e!r}")
    return ops


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200, help="launches per (op, tokens, mode)")
    p.add_argument("--tokens", default="1,6", help="comma-separated token counts")
    p.add_argument("--ops", default="", help="comma-separated subset (default: all)")
    args = p.parse_args()
    want = {o for o in args.ops.split(",") if o}

    for tokens in [int(t) for t in args.tokens.split(",")]:
        ops = make_ops(tokens)
        for name, fn in ops.items():
            if want and name not in want:
                continue
            for _ in range(10):  # warm up (cuBLAS handles, lazy module loading)
                fn()
            torch.cuda.synchronize()

            with nvtx.annotate(f"{name}|tokens={tokens}|eager", domain="microbench"):
                for _ in range(args.n):
                    fn()
                torch.cuda.synchronize()

            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                fn()  # warm up on the capture stream
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            with torch.cuda.graph(g):
                for _ in range(args.n):
                    fn()
            torch.cuda.synchronize()
            g.replay()  # first replay uploads the graph; not measured
            torch.cuda.synchronize()
            with nvtx.annotate(f"{name}|tokens={tokens}|graph", domain="microbench"):
                g.replay()
                torch.cuda.synchronize()
            print(f"done {name} tokens={tokens}", flush=True)


if __name__ == "__main__":
    main()
