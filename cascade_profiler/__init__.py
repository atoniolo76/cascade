"""Profiler orchestration; GPU and Modal dependencies are optional."""

from cascade_profiler.pipeline import RunResult, profile
from cascade_profiler.planning import trace_plan
from cascade_profiler.workload import Workload

__all__ = ["RunResult", "Workload", "image", "profile", "register", "trace_plan"]


def __getattr__(name):
    if name in {"image", "register"}:
        from cascade_profiler import modal_support
        return getattr(modal_support, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
