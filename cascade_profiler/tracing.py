"""NVBit launch environment shared by the compatibility app and package pipeline."""

from pathlib import Path


def trace_command(command: list[str], accelsim_root: str, extra: dict | None = None) -> list[str]:
    tools = Path(accelsim_root) / "util/tracer_nvbit"
    overrides = {"CUDA_INJECTION64_PATH": str(tools / "tracer_tool/tracer_tool.so"),
                 "SPINLOCK_HANDLING_MODE": "2", **(extra or {})}
    return [str(tools / "others/torch_hook/run.sh"), "env",
            *(f"{k}={v}" for k, v in overrides.items()), *command]


def post_process_command(accelsim_root: str, traces_dir: Path) -> list[str]:
    return [str(Path(accelsim_root) / "util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing"),
            str(traces_dir), "-j", "8"]
