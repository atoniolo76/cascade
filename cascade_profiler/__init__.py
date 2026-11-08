"""Profiler orchestration; GPU and Modal dependencies are optional."""

from cascade_profiler.pipeline import RunResult, profile
from cascade_profiler.workload import Workload
from cascade_profiler.planning import trace_plan

__all__ = ["Workload", "RunResult", "profile", "trace_plan", "simulate_accelsim", "image", "accelsim_image", "register"]


def __getattr__(name):
    if name == "simulate_accelsim":
        from cascade_profiler.accelsim import simulate_accelsim
        return simulate_accelsim
    if name in {"image", "accelsim_image", "register"}:
        from cascade_profiler import modal_support
        return getattr(modal_support, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
