from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

POSTPROCESS_REL = "util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing"


class SettingsError(ValueError):
    pass


def _bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    raise SettingsError(f"not a boolean: {value!r}")


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    token: str
    launcher: Literal["local", "sail"]
    public_url: str
    s3_bucket: str
    s3_prefix: str
    accelsim_dir: Path
    sim_cmd: str | None
    postprocess_cmd: str
    heartbeat_timeout_s: float
    run_s3: bool

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Settings:
        env = dict(os.environ if env is None else env)
        token = env.get("CASCADE_TOKEN", "")
        if not token:
            raise SettingsError("CASCADE_TOKEN is required")
        launcher = env.get("CASCADE_LAUNCHER", "local")
        if launcher not in ("local", "sail"):
            raise SettingsError(f"CASCADE_LAUNCHER must be local or sail, got {launcher!r}")
        accelsim_dir = Path(env.get("CASCADE_ACCELSIM_DIR", "/opt/accel-sim-framework"))
        return cls(
            data_dir=Path(env.get("CASCADE_DATA_DIR", "/data")),
            token=token,
            launcher=launcher,
            public_url=env.get("CASCADE_PUBLIC_URL", "http://127.0.0.1:8000").rstrip("/"),
            s3_bucket=env.get("CASCADE_S3_BUCKET", "cascade"),
            s3_prefix=env.get("CASCADE_S3_PREFIX", "accelsim-traces").strip("/"),
            accelsim_dir=accelsim_dir,
            sim_cmd=env.get("CASCADE_SIM_CMD") or None,
            postprocess_cmd=env.get("CASCADE_POSTPROCESS_CMD")
            or str(accelsim_dir / POSTPROCESS_REL),
            heartbeat_timeout_s=float(env.get("CASCADE_HEARTBEAT_TIMEOUT_S", "60")),
            run_s3=_bool(env.get("CASCADE_RUN_S3", "true")),
        )

    @property
    def db_path(self) -> Path:
        return self.data_dir / "cascade.db"

    @property
    def s3_root(self) -> Path:
        return self.data_dir / "s3"

    @property
    def traces_root(self) -> Path:
        root = self.s3_root / self.s3_bucket
        return root / self.s3_prefix if self.s3_prefix else root

    @property
    def work_dir(self) -> Path:
        """Main-only scratch (post-processing logs and markers, local sim stdout)."""
        return self.data_dir / "work"

    def trace_dir(self, trace_id: str) -> Path:
        return self.traces_root / trace_id
