"""SQLite state for main. Every sim state change goes through Store.transition_sim."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from cascade.models import (
    SIM_TRANSITIONS,
    TERMINAL_SIM_STATES,
    Heartbeat,
    KernelLaunch,
    KernelStat,
    RunDetail,
    RunEntry,
    RunSummary,
    SimMetrics,
    SimState,
    SimStepTimeline,
    SimSummary,
    StepTimelineUpload,
    TraceDetail,
    TraceState,
    TraceSummary,
    run_state,
)

TRACE_TRANSITIONS: dict[TraceState, frozenset[TraceState]] = {
    TraceState.uploading: frozenset({TraceState.processing, TraceState.failed}),
    TraceState.processing: frozenset({TraceState.ready, TraceState.failed}),
    TraceState.ready: frozenset(),
    TraceState.failed: frozenset(),
}

LOG_TAIL_LINES = 20

SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    id TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata TEXT,
    kernel_count INTEGER NOT NULL DEFAULT 0,
    bytes INTEGER NOT NULL DEFAULT 0,
    kernels TEXT NOT NULL DEFAULT '[]',
    real_timing TEXT,
    has_step_timeline INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL REFERENCES traces(id),
    created_at TEXT NOT NULL,
    entries TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sims (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id),
    gpu_config TEXT NOT NULL,
    replica INTEGER NOT NULL,
    state TEXT NOT NULL,
    error TEXT,
    launcher_ref TEXT,
    created_at TEXT NOT NULL,
    state_changed_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    last_heartbeat_at TEXT,
    exit_code INTEGER,
    kernels_done INTEGER NOT NULL DEFAULT 0,
    kernels_total INTEGER NOT NULL DEFAULT 0,
    current_kernel TEXT,
    latest TEXT NOT NULL DEFAULT '{}',
    log_tail TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS sims_run ON sims(run_id);
CREATE TABLE IF NOT EXISTS log_chunks (
    sim_id TEXT NOT NULL REFERENCES sims(id),
    seq INTEGER NOT NULL,
    lines TEXT NOT NULL,
    PRIMARY KEY (sim_id, seq)
);
CREATE TABLE IF NOT EXISTS kernel_stats (
    sim_id TEXT NOT NULL REFERENCES sims(id),
    kernel_index INTEGER NOT NULL,
    kernel_name TEXT NOT NULL,
    kernel_launch_uid INTEGER,
    gpu_sim_cycle INTEGER,
    gpu_sim_insn INTEGER,
    gpu_ipc REAL,
    sim_time_ns REAL,
    stats TEXT NOT NULL,
    PRIMARY KEY (sim_id, kernel_index)
);
CREATE TABLE IF NOT EXISTS step_timelines (
    sim_id TEXT PRIMARY KEY REFERENCES sims(id),
    simulated TEXT NOT NULL,
    compare TEXT
);
"""

# Columns added after a table first shipped; CREATE TABLE IF NOT EXISTS leaves old databases alone.
ADDED_COLUMNS = (("traces", "has_step_timeline", "INTEGER NOT NULL DEFAULT 0"),)


class IllegalTransition(Exception):
    pass


class NotFound(Exception):
    pass


def now() -> datetime:
    return datetime.now(UTC)


def _ts(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript(SCHEMA)
        for table, column, decl in ADDED_COLUMNS:
            have = {r["name"] for r in self._db.execute(f"PRAGMA table_info({table})")}
            if column not in have:
                self._db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    def close(self) -> None:
        with self._lock:
            self._db.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield self._db
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            self._db.execute("COMMIT")

    def _one(self, sql: str, *args) -> sqlite3.Row | None:
        with self._lock:
            return self._db.execute(sql, args).fetchone()

    def _all(self, sql: str, *args) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    def register_trace(self, trace_id: str, kernel_count: int, nbytes: int) -> bool:
        t = _ts(now())
        with self._tx() as db:
            cur = db.execute(
                "INSERT OR IGNORE INTO traces (id, state, created_at, updated_at, kernel_count,"
                " bytes) VALUES (?, ?, ?, ?, ?, ?)",
                (trace_id, TraceState.uploading, t, t, kernel_count, nbytes),
            )
            return cur.rowcount == 1

    def update_trace(
        self,
        trace_id: str,
        *,
        state: TraceState | None = None,
        error: str | None = None,
        metadata: dict | None = None,
        kernel_count: int | None = None,
        nbytes: int | None = None,
        kernels: list[KernelLaunch] | None = None,
        real_timing: list[dict] | None = None,
        has_step_timeline: bool | None = None,
    ) -> None:
        with self._tx() as db:
            row = db.execute("SELECT state FROM traces WHERE id = ?", (trace_id,)).fetchone()
            if row is None:
                raise NotFound(trace_id)
            current = TraceState(row["state"])
            if state is not None and state != current and state not in TRACE_TRANSITIONS[current]:
                raise IllegalTransition(f"trace {trace_id}: {current} -> {state}")
            sets: dict[str, object] = {"updated_at": _ts(now())}
            if state is not None:
                sets["state"] = state
            if error is not None:
                sets["error"] = error
            if metadata is not None:
                sets["metadata"] = json.dumps(metadata)
            if kernel_count is not None:
                sets["kernel_count"] = kernel_count
            if nbytes is not None:
                sets["bytes"] = nbytes
            if kernels is not None:
                sets["kernels"] = json.dumps([k.model_dump() for k in kernels])
            if real_timing is not None:
                sets["real_timing"] = json.dumps(real_timing)
            if has_step_timeline is not None:
                sets["has_step_timeline"] = int(has_step_timeline)
            cols = ", ".join(f"{k} = ?" for k in sets)
            db.execute(f"UPDATE traces SET {cols} WHERE id = ?", (*sets.values(), trace_id))

    def trace_states(self) -> dict[str, TraceState]:
        return {r["id"]: TraceState(r["state"]) for r in self._all("SELECT id, state FROM traces")}

    def step_timeline_flags(self) -> dict[str, bool]:
        rows = self._all("SELECT id, has_step_timeline FROM traces")
        return {r["id"]: bool(r["has_step_timeline"]) for r in rows}

    def get_trace(self, trace_id: str) -> TraceDetail | None:
        row = self._one("SELECT * FROM traces WHERE id = ?", trace_id)
        if row is None:
            return None
        return TraceDetail(
            **_trace_summary(row).model_dump(),
            metadata=json.loads(row["metadata"]) if row["metadata"] else None,
            kernels=[KernelLaunch(**k) for k in json.loads(row["kernels"])],
        )

    def get_trace_summary(self, trace_id: str) -> TraceSummary | None:
        row = self._one("SELECT * FROM traces WHERE id = ?", trace_id)
        return _trace_summary(row) if row else None

    def list_traces(self) -> list[TraceSummary]:
        rows = self._all("SELECT * FROM traces ORDER BY created_at DESC, id DESC")
        return [_trace_summary(r) for r in rows]

    def real_timing(self, trace_id: str) -> list[dict] | None:
        row = self._one("SELECT real_timing FROM traces WHERE id = ?", trace_id)
        return json.loads(row["real_timing"]) if row and row["real_timing"] else None

    def create_run(self, trace_id: str, entries: list[RunEntry], kernels_total: int) -> str:
        run_id = new_id("run")
        t = _ts(now())
        with self._tx() as db:
            db.execute(
                "INSERT INTO runs (id, trace_id, created_at, entries) VALUES (?, ?, ?, ?)",
                (run_id, trace_id, t, json.dumps([e.model_dump() for e in entries])),
            )
            for entry in entries:
                for replica in range(entry.count):
                    db.execute(
                        "INSERT INTO sims (id, run_id, gpu_config, replica, state, created_at,"
                        " state_changed_at, kernels_total) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            new_id("sim"),
                            run_id,
                            entry.gpu_config,
                            replica,
                            SimState.pending,
                            t,
                            t,
                            kernels_total,
                        ),
                    )
        return run_id

    def get_run_summary(self, run_id: str) -> RunSummary | None:
        row = self._one("SELECT * FROM runs WHERE id = ?", run_id)
        return self._run_summary(row, self.sims_for_run(run_id)) if row else None

    def get_run(self, run_id: str) -> RunDetail | None:
        row = self._one("SELECT * FROM runs WHERE id = ?", run_id)
        if row is None:
            return None
        sims = self.sims_for_run(run_id)
        trace = self.get_trace_summary(row["trace_id"])
        assert trace is not None
        return RunDetail(**self._run_summary(row, sims).model_dump(), trace=trace, sims=sims)

    def list_runs(self) -> list[RunSummary]:
        rows = self._all("SELECT * FROM runs ORDER BY created_at DESC, id DESC")
        return [self._run_summary(r, self.sims_for_run(r["id"])) for r in rows]

    def run_trace_id(self, run_id: str) -> str | None:
        row = self._one("SELECT trace_id FROM runs WHERE id = ?", run_id)
        return row["trace_id"] if row else None

    @staticmethod
    def _run_summary(row: sqlite3.Row, sims: list[SimSummary]) -> RunSummary:
        states = [s.state for s in sims]
        return RunSummary(
            id=row["id"],
            trace_id=row["trace_id"],
            created_at=_dt(row["created_at"]),
            state=run_state(states),
            entries=[RunEntry(**e) for e in json.loads(row["entries"])],
            sim_counts=dict(Counter(states)),
        )

    def get_sim(self, sim_id: str) -> SimSummary | None:
        row = self._one("SELECT * FROM sims WHERE id = ?", sim_id)
        return _sim_summary(row) if row else None

    def sims_for_run(self, run_id: str) -> list[SimSummary]:
        rows = self._all(
            "SELECT * FROM sims WHERE run_id = ? ORDER BY gpu_config, replica, id", run_id
        )
        return [_sim_summary(r) for r in rows]

    def run_ids_with_pending_sims(self) -> list[str]:
        rows = self._all("SELECT DISTINCT run_id FROM sims WHERE state = ?", SimState.pending)
        return [r["run_id"] for r in rows]

    def transition_sim(
        self,
        sim_id: str,
        to: SimState,
        *,
        error: str | None = None,
        exit_code: int | None = None,
        heartbeat: bool = False,
    ) -> SimSummary:
        """The only writer of sims.state. Raises IllegalTransition for moves outside
        SIM_TRANSITIONS."""
        with self._tx() as db:
            return self._transition(db, sim_id, to, error=error, exit_code=exit_code, hb=heartbeat)

    def _transition(
        self,
        db: sqlite3.Connection,
        sim_id: str,
        to: SimState,
        *,
        error: str | None,
        exit_code: int | None,
        hb: bool,
    ) -> SimSummary:
        row = db.execute("SELECT * FROM sims WHERE id = ?", (sim_id,)).fetchone()
        if row is None:
            raise NotFound(sim_id)
        current = SimState(row["state"])
        if to not in SIM_TRANSITIONS[current]:
            raise IllegalTransition(f"sim {sim_id}: {current} -> {to}")
        t = _ts(now())
        sets: dict[str, object] = {"state": to, "state_changed_at": t}
        if to == SimState.running and row["started_at"] is None:
            sets["started_at"] = t
        if to in TERMINAL_SIM_STATES:
            sets["finished_at"] = t
        if error is not None or to == SimState.running:
            sets["error"] = error
        if exit_code is not None:
            sets["exit_code"] = exit_code
        if hb:
            sets["last_heartbeat_at"] = t
        cols = ", ".join(f"{k} = ?" for k in sets)
        db.execute(f"UPDATE sims SET {cols} WHERE id = ?", (*sets.values(), sim_id))
        return _sim_summary(db.execute("SELECT * FROM sims WHERE id = ?", (sim_id,)).fetchone())

    def apply_heartbeat(self, sim_id: str, hb: Heartbeat) -> tuple[SimSummary, bool]:
        """Returns the sim after the heartbeat and whether anything changed."""
        with self._tx() as db:
            row = db.execute("SELECT state FROM sims WHERE id = ?", (sim_id,)).fetchone()
            if row is None:
                raise NotFound(sim_id)
            current = SimState(row["state"])
            if current in TERMINAL_SIM_STATES:
                return self.get_sim(sim_id), False
            to = SimState(hb.state)
            if to != current and to in SIM_TRANSITIONS[current]:
                sim = self._transition(
                    db, sim_id, to, error=hb.error, exit_code=hb.exit_code, hb=True
                )
                return sim, True
            db.execute("UPDATE sims SET last_heartbeat_at = ? WHERE id = ?", (_ts(now()), sim_id))
            return _sim_summary(
                db.execute("SELECT * FROM sims WHERE id = ?", (sim_id,)).fetchone()
            ), True

    def set_launcher_ref(self, sim_id: str, ref: str) -> SimSummary:
        with self._tx() as db:
            db.execute("UPDATE sims SET launcher_ref = ? WHERE id = ?", (ref, sim_id))
            return _sim_summary(db.execute("SELECT * FROM sims WHERE id = ?", (sim_id,)).fetchone())

    def mark_stale_lost(self, cutoff: datetime) -> list[SimSummary]:
        """Moves launching/running sims whose last heartbeat (or state change, if none)
        predates cutoff to lost."""
        return self._expire(
            (SimState.launching, SimState.running), cutoff, SimState.lost, "no heartbeat"
        )

    def fail_lost(self, cutoff: datetime, error: str) -> list[SimSummary]:
        return self._expire((SimState.lost,), cutoff, SimState.failed, error)

    def _expire(
        self, states: tuple[SimState, ...], cutoff: datetime, to: SimState, error: str
    ) -> list[SimSummary]:
        marks = ", ".join("?" for _ in states)
        with self._tx() as db:
            rows = db.execute(
                f"SELECT id FROM sims WHERE state IN ({marks})"
                " AND COALESCE(last_heartbeat_at, state_changed_at) < ?",
                (*states, _ts(cutoff)),
            ).fetchall()
            return [
                self._transition(db, r["id"], to, error=error, exit_code=None, hb=False)
                for r in rows
            ]

    def kernel_stats_for_run(self, run_id: str) -> list[KernelStat]:
        rows = self._all(
            "SELECT k.* FROM kernel_stats k JOIN sims s ON s.id = k.sim_id WHERE s.run_id = ?"
            " ORDER BY s.gpu_config, s.replica, s.id, k.kernel_index",
            run_id,
        )
        return [_kernel_stat(r) for r in rows]

    def add_log_chunk(self, sim_id: str, seq: int, lines: list[str]) -> bool:
        """False when this seq was already stored (a retried chunk)."""
        with self._tx() as db:
            cur = db.execute(
                "INSERT OR IGNORE INTO log_chunks (sim_id, seq, lines) VALUES (?, ?, ?)",
                (sim_id, seq, json.dumps(lines)),
            )
            if cur.rowcount != 1:
                return False
            row = db.execute("SELECT log_tail FROM sims WHERE id = ?", (sim_id,)).fetchone()
            tail = (json.loads(row["log_tail"]) + lines)[-LOG_TAIL_LINES:]
            db.execute("UPDATE sims SET log_tail = ? WHERE id = ?", (json.dumps(tail), sim_id))
            return True

    def log_chunks(self, sim_id: str, before_seq: int | None = None) -> Iterator[list[str]]:
        sql = "SELECT lines FROM log_chunks WHERE sim_id = ?"
        args: tuple = (sim_id,)
        if before_seq is not None:
            sql += " AND seq < ?"
            args += (before_seq,)
        for row in self._all(sql + " ORDER BY seq", *args):
            yield json.loads(row["lines"])

    def record_progress(
        self,
        sim_id: str,
        stats: list[KernelStat],
        current_kernel: str | None,
        latest: SimMetrics,
    ) -> tuple[SimSummary, list[KernelStat]]:
        """Stores parsed kernel stats (ignoring ones already stored) and the sim's live
        metrics. Returns the sim and the newly stored stats."""
        added = []
        with self._tx() as db:
            for s in stats:
                cur = db.execute(
                    "INSERT OR IGNORE INTO kernel_stats (sim_id, kernel_index, kernel_name,"
                    " kernel_launch_uid, gpu_sim_cycle, gpu_sim_insn, gpu_ipc, sim_time_ns, stats)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        sim_id,
                        s.kernel_index,
                        s.kernel_name,
                        s.kernel_launch_uid,
                        s.gpu_sim_cycle,
                        s.gpu_sim_insn,
                        s.gpu_ipc,
                        s.sim_time_ns,
                        json.dumps(s.stats),
                    ),
                )
                if cur.rowcount == 1:
                    added.append(s)
            done = db.execute(
                "SELECT COUNT(*) FROM kernel_stats WHERE sim_id = ?", (sim_id,)
            ).fetchone()[0]
            db.execute(
                "UPDATE sims SET kernels_done = ?, current_kernel = ?, latest = ? WHERE id = ?",
                (done, current_kernel, latest.model_dump_json(), sim_id),
            )
            sim = _sim_summary(db.execute("SELECT * FROM sims WHERE id = ?", (sim_id,)).fetchone())
        return sim, added

    def set_step_timeline(self, sim_id: str, upload: StepTimelineUpload) -> None:
        """Replaces any earlier upload for this sim."""
        with self._tx() as db:
            db.execute(
                "INSERT OR REPLACE INTO step_timelines (sim_id, simulated, compare)"
                " VALUES (?, ?, ?)",
                (
                    sim_id,
                    json.dumps(upload.simulated),
                    None if upload.compare is None else json.dumps(upload.compare),
                ),
            )

    def step_timelines_for_run(self, run_id: str) -> list[SimStepTimeline]:
        rows = self._all(
            "SELECT t.*, s.gpu_config, s.replica FROM step_timelines t"
            " JOIN sims s ON s.id = t.sim_id WHERE s.run_id = ?"
            " ORDER BY s.gpu_config, s.replica, s.id",
            run_id,
        )
        return [
            SimStepTimeline(
                sim_id=r["sim_id"],
                gpu_config=r["gpu_config"],
                replica=r["replica"],
                simulated=json.loads(r["simulated"]),
                compare=json.loads(r["compare"]) if r["compare"] is not None else None,
            )
            for r in rows
        ]

    def kernel_stats(self, sim_id: str) -> list[KernelStat]:
        rows = self._all(
            "SELECT * FROM kernel_stats WHERE sim_id = ? ORDER BY kernel_index", sim_id
        )
        return [_kernel_stat(r) for r in rows]


def _trace_summary(row: sqlite3.Row) -> TraceSummary:
    meta = json.loads(row["metadata"]) if row["metadata"] else {}
    layers = meta.get("layers")
    return TraceSummary(
        id=row["id"],
        state=TraceState(row["state"]),
        error=row["error"],
        created_at=_dt(row["created_at"]),
        updated_at=_dt(row["updated_at"]),
        model=meta.get("model"),
        layers=[str(x) for x in layers] if isinstance(layers, list) else [],
        source_gpu=meta.get("gpu"),
        trace_gpu_arch=meta.get("trace_gpu_arch") or "sm_90",
        suggested_sim_config=meta.get("suggested_sim_config"),
        kernel_count=row["kernel_count"],
        bytes=row["bytes"],
        has_real_timing=row["real_timing"] is not None,
        has_step_timeline=bool(row["has_step_timeline"]),
    )


def _sim_summary(row: sqlite3.Row) -> SimSummary:
    return SimSummary(
        id=row["id"],
        run_id=row["run_id"],
        gpu_config=row["gpu_config"],
        replica=row["replica"],
        state=SimState(row["state"]),
        error=row["error"],
        launcher_ref=row["launcher_ref"],
        created_at=_dt(row["created_at"]),
        started_at=_dt(row["started_at"]),
        finished_at=_dt(row["finished_at"]),
        last_heartbeat_at=_dt(row["last_heartbeat_at"]),
        exit_code=row["exit_code"],
        kernels_done=row["kernels_done"],
        kernels_total=row["kernels_total"],
        current_kernel=row["current_kernel"],
        latest=SimMetrics.model_validate_json(row["latest"]),
        log_tail=json.loads(row["log_tail"]),
    )


def _kernel_stat(row: sqlite3.Row) -> KernelStat:
    return KernelStat(
        sim_id=row["sim_id"],
        kernel_index=row["kernel_index"],
        kernel_name=row["kernel_name"],
        kernel_launch_uid=row["kernel_launch_uid"],
        gpu_sim_cycle=row["gpu_sim_cycle"],
        gpu_sim_insn=row["gpu_sim_insn"],
        gpu_ipc=row["gpu_ipc"],
        sim_time_ns=row["sim_time_ns"],
        stats=json.loads(row["stats"]),
    )
