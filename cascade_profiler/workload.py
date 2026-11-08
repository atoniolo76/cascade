"""Serializable workload configuration shared by the parent and GPU subprocesses."""

import importlib
import json
from dataclasses import asdict, dataclass, field
from typing import Callable

EXEC_MODES = ("eager", "compile", "graphs")


def execution_kwargs(exec_mode: str, no_async_scheduling: bool) -> dict:
    if exec_mode == "eager":
        kw = {"enforce_eager": True}
    elif exec_mode == "compile":
        kw = {"enforce_eager": False, "compilation_config": {"cudagraph_mode": "NONE"}}
    else:
        kw = {"enforce_eager": False}
    if no_async_scheduling:
        kw["async_scheduling"] = False
    return kw


@dataclass
class Workload:
    model: str
    prompts: list[str]
    max_tokens: int = 32
    repeats: int = 4
    exec_mode: str = "eager"
    max_model_len: int = 2048
    llm_kwargs: dict = field(default_factory=dict)
    sampling_kwargs: dict = field(default_factory=dict)
    driver: str | None = None

    def __post_init__(self):
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("model must be a nonempty string")
        if not isinstance(self.prompts, list) or not all(isinstance(p, str) for p in self.prompts):
            raise ValueError("prompts must be a list of strings")
        if not self.prompts and self.driver is None:
            raise ValueError("prompts must not be empty for the default driver")
        for name in ("max_tokens", "repeats", "max_model_len"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.exec_mode not in EXEC_MODES:
            raise ValueError(f"exec_mode must be one of {EXEC_MODES}")
        if not isinstance(self.llm_kwargs, dict) or not isinstance(self.sampling_kwargs, dict):
            raise ValueError("llm_kwargs and sampling_kwargs must be dictionaries")
        if any(not isinstance(k, str) for k in (*self.llm_kwargs, *self.sampling_kwargs)):
            raise ValueError("llm_kwargs and sampling_kwargs keys must be strings")
        if self.driver is not None and (not isinstance(self.driver, str) or self.driver.count(":") != 1):
            raise ValueError("driver must be an importable 'module:function' string")
        self.merged_llm_kwargs(execution_kwargs(self.exec_mode, True))
        json.dumps(asdict(self))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Workload":
        return cls(**data)

    def resolve_driver(self) -> Callable | None:
        if self.driver is None:
            return None
        module, name = self.driver.split(":")
        if module in {"__main__", "__mp_main__"}:
            raise ValueError("driver must live in an importable module, not __main__")
        try:
            driver = getattr(importlib.import_module(module), name)
        except (ImportError, AttributeError, ValueError) as exc:
            raise ValueError(f"cannot import driver {self.driver!r}") from exc
        if not callable(driver):
            raise ValueError(f"driver {self.driver!r} is not callable")
        return driver

    def merged_llm_kwargs(self, required: dict) -> dict:
        required = {"model": self.model, "max_model_len": self.max_model_len, **required}
        for key, value in required.items():
            if key in self.llm_kwargs and self.llm_kwargs[key] != value:
                raise ValueError(f"llm_kwargs[{key!r}] conflicts with required profiler value {value!r}")
        return {"gpu_memory_utilization": 0.8, **self.llm_kwargs, **required}

    def sampling_params_kwargs(self) -> dict:
        return {"temperature": 0.0, "seed": 0, "ignore_eos": True,
                "max_tokens": self.max_tokens, **self.sampling_kwargs}
