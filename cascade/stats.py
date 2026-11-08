"""Streaming parser for Accel-Sim stdout.

Per-kernel stats come from gpgpu_sim::print_stats, which starts with `kernel_name = ...` and is
followed by print_simulation_time (`gpgpu_simulation_time = ...`, then two
`gpgpu_simulation_rate` lines). Formats checked against gpgpu-sim_distribution 4195dd4
(src/gpgpu-sim/gpu-sim.cc, src/gpgpusim_entrypoint.cc) and accel-sim-framework v2.0.1
(gpu-simulator/accel-sim.cc).
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable

from cascade.models import KernelStat, SimMetrics

_KV_RE = re.compile(r"^\s*([A-Za-z_][\w\[\]\.]*)\s*=\s*(.*?)\s*$")
_LAUNCH_RE = re.compile(r"^launching kernel name: (\S+) uid: (\d+)")
_RATE_RE = re.compile(r"^gpgpu_simulation_rate = (\d+) \(inst/sec\)")
_BLOCK_END_PREFIXES = (
    "gpgpu_simulation_time",
    "launching kernel name:",
    "Processing kernel",
    "Header info loaded",
    "GPGPU-Sim: *** exit detected",
)


def _value(raw: str) -> float | str:
    try:
        v = float(raw)
    except ValueError:
        return raw
    return v if math.isfinite(v) else raw


def _int(stats: dict[str, float | str], key: str) -> int | None:
    v = stats.get(key)
    return int(v) if isinstance(v, float) else None


def _first_int(v: float | str | None) -> int | None:
    """`kernel_launch_uid = 3 ` parses as a float; several concurrent kernels leave a string."""
    if isinstance(v, float):
        return int(v)
    head = str(v or "").split()[:1]
    return int(head[0]) if head and head[0].isdigit() else None


class LogParser:
    """Feed lines in order; completed kernel blocks are returned by feed() and finish()."""

    def __init__(self, sim_id: str, core_clock_mhz: float | None):
        self.sim_id = sim_id
        self.core_clock_mhz = core_clock_mhz
        self.kernels_done = 0
        self.current_kernel: str | None = None
        self.latest = SimMetrics()
        self._block: dict[str, float | str] | None = None
        self._seen: dict[str, int] = {}

    def feed(self, lines: Iterable[str]) -> list[KernelStat]:
        done: list[KernelStat] = []
        for raw in lines:
            line = raw.rstrip("\r\n")
            if line.startswith("kernel_name ="):
                if self._block is not None:
                    done.append(self._close())
                self._block, self._seen = {}, {}
            elif self._block is not None and line.startswith(_BLOCK_END_PREFIXES):
                done.append(self._close())

            if self._block is not None:
                if m := _KV_RE.match(line):
                    self._put(m.group(1), m.group(2))
                continue

            if m := _LAUNCH_RE.match(line):
                self.current_kernel = m.group(1)
            elif m := _RATE_RE.match(line):
                self.latest = self.latest.model_copy(update={"sim_rate_ips": float(m.group(1))})
        return done

    def finish(self) -> list[KernelStat]:
        return [self._close()] if self._block is not None else []

    def _put(self, key: str, raw: str) -> None:
        # Per-channel sections (DRAM, L2 banks) repeat keys; keep every occurrence.
        assert self._block is not None
        n = self._seen[key] = self._seen.get(key, 0) + 1
        self._block[key if n == 1 else f"{key}#{n}"] = _value(raw)

    def _close(self) -> KernelStat:
        stats = self._block or {}
        self._block = None
        names = str(stats.get("kernel_name", "")).split()
        uid = stats.get("kernel_launch_uid")
        cycles = _int(stats, "gpu_sim_cycle")
        ipc = stats.get("gpu_ipc")
        stat = KernelStat(
            sim_id=self.sim_id,
            kernel_index=self.kernels_done,
            kernel_name=names[0] if names else "",
            kernel_launch_uid=_first_int(uid),
            gpu_sim_cycle=cycles,
            gpu_sim_insn=_int(stats, "gpu_sim_insn"),
            gpu_ipc=ipc if isinstance(ipc, float) else None,
            sim_time_ns=cycles / self.core_clock_mhz * 1000
            if cycles is not None and self.core_clock_mhz
            else None,
            stats=stats,
        )
        self.kernels_done += 1
        totals = {
            field: v
            for field, key in (
                ("gpu_tot_sim_cycle", "gpu_tot_sim_cycle"),
                ("gpu_tot_sim_insn", "gpu_tot_sim_insn"),
                ("gpu_tot_ipc", "gpu_tot_ipc"),
            )
            if isinstance(v := stats.get(key), float)
        }
        self.latest = self.latest.model_copy(update=totals)
        return stat
