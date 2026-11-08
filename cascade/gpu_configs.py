from __future__ import annotations

import re
from pathlib import Path

from cascade.models import GpuConfig

HW_CONFIGS_REL = Path("gpu-simulator/gpgpu-sim/configs/tested-cfgs")

_DIR_RE = re.compile(r"^SM(\d+)_")


def arch_from_dir(name: str) -> str | None:
    """SM90_H100 -> sm_90, SM7_QV100 -> sm_70 (single-digit prefixes name the major only)."""
    m = _DIR_RE.match(name)
    if not m:
        return None
    digits = m.group(1)
    return f"sm_{digits}0" if len(digits) == 1 else f"sm_{digits}"


def core_clock_mhz(config_text: str) -> float | None:
    for raw in config_text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line.startswith("-gpgpu_clock_domains"):
            parts = line.split()
            if len(parts) >= 2:
                return float(parts[1].split(":")[0])
    return None


class GpuConfigRegistry:
    def __init__(self, accelsim_dir: Path):
        self.accelsim_dir = accelsim_dir
        self._configs: dict[str, GpuConfig] = {}
        root = accelsim_dir / HW_CONFIGS_REL
        for path in sorted(root.glob("*/gpgpusim.config")):
            name = path.parent.name
            arch = arch_from_dir(name)
            clock = core_clock_mhz(path.read_text(errors="replace"))
            if arch and clock:
                self._configs[name] = GpuConfig(name=name, arch=arch, core_clock_mhz=clock)

    def all(self) -> list[GpuConfig]:
        return list(self._configs.values())

    def get(self, name: str) -> GpuConfig | None:
        return self._configs.get(name)
