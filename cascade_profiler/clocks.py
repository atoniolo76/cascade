import subprocess
import threading
import time
from pathlib import Path


def _write_clock_snapshot(path: Path):
    """nvidia-smi clock/throttle snapshot (supported, current, max, application clocks)."""
    out = subprocess.run(["nvidia-smi", "-q", "-d", "CLOCK,PERFORMANCE,POWER"],
                         capture_output=True, text=True)
    path.write_text(out.stdout + out.stderr)


class ClockSampler(threading.Thread):
    """Samples SM/mem clocks + throttle reasons via NVML every `period_s` into a CSV.

    Each row carries wall_ns (time.time_ns, same clock as vllm_trace.py's step log) so samples
    can be mapped onto the nsys timeline through the step NVTX ranges.
    """

    COLS = ["wall_ns", "sm_mhz", "mem_mhz", "gr_mhz", "power_w", "temp_c", "throttle_reasons_hex",
            "throttle_reasons"]

    def __init__(self, path: Path, period_s: float = 0.05):
        super().__init__(daemon=True, name="clock-sampler")
        self.path, self.period_s = path, period_s
        self._stop_event = threading.Event()
        self.rows: list[dict] = []
        self.error: str | None = None

    def run(self):
        try:
            import pynvml as nv  # nvidia-ml-py, a vLLM dependency
            nv.nvmlInit()
            h = nv.nvmlDeviceGetHandleByIndex(0)
            get_reasons = getattr(nv, "nvmlDeviceGetCurrentClocksEventReasons", None) or \
                nv.nvmlDeviceGetCurrentClocksThrottleReasons
            names = {getattr(nv, a): a.split("Reason", 1)[1] for a in dir(nv)
                     if a.startswith(("nvmlClocksThrottleReason", "nvmlClocksEventReason"))
                     and "All" not in a and "None" not in a and isinstance(getattr(nv, a), int)}
            while not self._stop_event.is_set():
                r = get_reasons(h)
                self.rows.append({
                    "wall_ns": time.time_ns(),
                    "sm_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
                    "mem_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM),
                    "gr_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_GRAPHICS),
                    "power_w": nv.nvmlDeviceGetPowerUsage(h) / 1000,
                    "temp_c": nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU),
                    "throttle_reasons_hex": hex(r),
                    "throttle_reasons": "|".join(sorted({n for b, n in names.items() if r & b})),
                })
                self._stop_event.wait(self.period_s)
        except Exception as e:
            self.error = repr(e)
            print(f"[clocks] NVML sampling failed: {e!r}", flush=True)

    def stop(self):
        self._stop_event.set()
        self.join(10)
        import csv
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.COLS)
            w.writeheader()
            w.writerows(self.rows)


def _lock_clocks(mhz: int) -> dict:
    """Try to lock SM clocks (needs root on the host; usually refused in containers)."""
    res = {"requested_sm_mhz": mhz}
    for key, cmd in (("lgc", ["nvidia-smi", "-lgc", f"{mhz},{mhz}"]),):
        r = subprocess.run(cmd, capture_output=True, text=True)
        res[key] = {"returncode": r.returncode, "output": (r.stdout + r.stderr).strip()}
    res["locked"] = res["lgc"]["returncode"] == 0
    return res


def _clock_stats(rows: list[dict], t0_ns: int | None, t1_ns: int | None) -> dict:
    import statistics
    sel = [r for r in rows if t0_ns is None or t0_ns <= r["wall_ns"] <= t1_ns] or rows
    if not sel:
        return {}
    out = {"samples": len(sel)}
    for k in ("sm_mhz", "mem_mhz"):
        v = [r[k] for r in sel]
        out[k] = {"min": min(v), "median": statistics.median(v), "max": max(v)}
    out["throttle_reasons_seen"] = sorted({x for r in sel for x in r["throttle_reasons"].split("|") if x})
    return out


def _gpu_name() -> str:
    return subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()
