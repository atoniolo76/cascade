"""vLLM workload traced with Accel-Sim 2.0's NVBit torch hook.

Mirrors util/tracer_nvbit/others/torch_hook/vllm_example.py from accel-sim-framework.
Must be launched under accel-sim's torch_hook/run.sh, which sets CUDA_INJECTION64_PATH,
NVBIT_INSTRUMENTATION_ENABLED=0, VLLM_ENABLE_V1_MULTIPROCESSING=0, etc.
"""

import argparse
import json
import logging

import torch
from torch_hook import TorchModelHookWrapper, hook_nvbit_to_layer
from vllm import LLM, SamplingParams


def print_layer_names(model: torch.nn.Module):
    print("Available layers:")
    for name, module in model.named_modules():
        print(f"  {name}: {module.__class__.__name__}")


def apply_nvbit_hook(model: torch.nn.Module, layers_to_trace: list[str]):
    hook_wrapper = TorchModelHookWrapper(model)
    for layer in layers_to_trace:
        hook_nvbit_to_layer(hook_wrapper, layer)  # NVBit on only during this layer


def main():
    # Surface torch_hook's "Found nvbit tracer, attaching hooks to layer ..." messages.
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--layers", default="", help="comma-separated module names")
    p.add_argument("--prompts", required=True, help="JSON list of prompts")
    p.add_argument("--max-tokens", type=int, default=4)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--list-layers", action="store_true")
    args = p.parse_args()

    # enforce_eager=True is recommended by Accel-Sim (disables CUDA Graphs).
    llm = LLM(
        model=args.model,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.8,
    )

    llm.collective_rpc(lambda self: print_layer_names(self.model_runner.model))
    if args.list_layers:
        return

    layers = [l.strip() for l in args.layers.split(",") if l.strip()]
    if not layers:
        raise SystemExit("--layers is required unless --list-layers is set")

    llm.collective_rpc(lambda self: apply_nvbit_hook(self.model_runner.model, layers))

    # Greedy + ignore_eos: the spinlock-detection passes and the trace pass must launch the
    # exact same kernel sequence (same number of decode steps), so no sampling randomness.
    sampling_params = SamplingParams(
        temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True, seed=0
    )
    outputs = llm.generate(json.loads(args.prompts), sampling_params)
    for output in outputs:
        print(f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}")

    del llm


if __name__ == "__main__":
    main()
