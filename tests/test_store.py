import sqlite3

import pytest

from cascade.models import Heartbeat, RunEntry, SimState, TraceState
from cascade.store import IllegalTransition, Store


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "cascade.db")
    s.register_trace("t1", 3, 100)
    s.update_trace("t1", state=TraceState.processing)
    s.update_trace("t1", state=TraceState.ready)
    yield s
    s.close()


def one_sim(store):
    run_id = store.create_run("t1", [RunEntry(gpu_config="SM90_H100", count=1)], {"SM90_H100": 3})
    return store.sims_for_run(run_id)[0].id


def test_legal_path_to_succeeded(store):
    sim = one_sim(store)
    store.transition_sim(sim, SimState.launching)
    running = store.transition_sim(sim, SimState.running, heartbeat=True)
    assert running.started_at is not None and running.last_heartbeat_at is not None
    done = store.transition_sim(sim, SimState.succeeded, exit_code=0)
    assert (done.state, done.exit_code) == (SimState.succeeded, 0)
    assert done.finished_at is not None


@pytest.mark.parametrize(
    "path",
    [
        [SimState.running],
        [SimState.lost],
        [SimState.succeeded],
        [SimState.launching, SimState.pending],
        [SimState.cancelled, SimState.running],
        [SimState.launching, SimState.succeeded, SimState.failed],
    ],
)
def test_illegal_moves_are_rejected_and_leave_state_alone(store, path):
    sim = one_sim(store)
    *legal, illegal = path
    for to in legal:
        store.transition_sim(sim, to)
    before = store.get_sim(sim)
    with pytest.raises(IllegalTransition):
        store.transition_sim(sim, illegal)
    assert store.get_sim(sim) == before


def test_heartbeat_on_terminal_sim_changes_nothing(store):
    sim = one_sim(store)
    store.transition_sim(sim, SimState.cancelled)
    before = store.get_sim(sim)
    after, changed = store.apply_heartbeat(sim, Heartbeat(state="running"))
    assert not changed
    assert after == before == store.get_sim(sim)


def test_heartbeat_revives_lost_sim(store):
    sim = one_sim(store)
    store.transition_sim(sim, SimState.launching)
    store.transition_sim(sim, SimState.lost)
    after, _ = store.apply_heartbeat(sim, Heartbeat(state="running"))
    assert after.state == SimState.running


def test_duplicate_log_seq_is_ignored(store):
    sim = one_sim(store)
    assert store.add_log_chunk(sim, 0, ["a", "b"])
    assert store.add_log_chunk(sim, 1, ["c"])
    assert not store.add_log_chunk(sim, 0, ["a", "b"])
    assert not store.add_log_chunk(sim, 1, ["different retry body"])
    assert list(store.log_chunks(sim)) == [["a", "b"], ["c"]]
    assert store.get_sim(sim).log_tail == ["a", "b", "c"]


def test_illegal_trace_move_is_rejected(store):
    store.register_trace("t2", 0, 0)
    with pytest.raises(IllegalTransition):
        store.update_trace("t2", state=TraceState.ready)
    assert store.get_trace("t2").state == TraceState.uploading


def test_state_survives_reopen(tmp_path):
    s = Store(tmp_path / "db")
    s.register_trace("t", 1, 2)
    s.close()
    s2 = Store(tmp_path / "db")
    assert not s2.register_trace("t", 9, 9)
    assert s2.get_trace("t").kernel_count == 1


def test_database_from_before_step_timing_gains_the_column(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute(
        "CREATE TABLE traces (id TEXT PRIMARY KEY, state TEXT NOT NULL, error TEXT,"
        " created_at TEXT NOT NULL, updated_at TEXT NOT NULL, metadata TEXT,"
        " kernel_count INTEGER NOT NULL DEFAULT 0, bytes INTEGER NOT NULL DEFAULT 0,"
        " kernels TEXT NOT NULL DEFAULT '[]', real_timing TEXT)"
    )
    old.execute(
        "INSERT INTO traces (id, state, created_at, updated_at) VALUES"
        " ('t', 'ready', '2026-11-07T00:00:00+00:00', '2026-11-07T00:00:00+00:00')"
    )
    old.commit()
    old.close()
    s = Store(path)
    assert s.get_trace("t").has_step_timeline is False
    s.update_trace("t", has_step_timeline=True)
    s.close()
    assert Store(path).get_trace("t").has_step_timeline is True


def test_database_from_before_samples_gains_the_columns(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute(
        "CREATE TABLE traces (id TEXT PRIMARY KEY, state TEXT NOT NULL, error TEXT,"
        " created_at TEXT NOT NULL, updated_at TEXT NOT NULL, metadata TEXT,"
        " kernel_count INTEGER NOT NULL DEFAULT 0, bytes INTEGER NOT NULL DEFAULT 0,"
        " kernels TEXT NOT NULL DEFAULT '[]', real_timing TEXT,"
        " has_step_timeline INTEGER NOT NULL DEFAULT 0)"
    )
    old.execute(
        "CREATE TABLE runs (id TEXT PRIMARY KEY, trace_id TEXT NOT NULL REFERENCES traces(id),"
        " created_at TEXT NOT NULL, entries TEXT NOT NULL)"
    )
    old.execute(
        "INSERT INTO traces (id, state, created_at, updated_at) VALUES"
        " ('t', 'ready', '2026-11-07T00:00:00+00:00', '2026-11-07T00:00:00+00:00')"
    )
    old.execute(
        "INSERT INTO runs (id, trace_id, created_at, entries) VALUES"
        " ('run-old', 't', '2026-11-07T00:00:00+00:00', '[]')"
    )
    old.commit()
    old.close()
    s = Store(path)
    assert s.get_trace("t").sample is False
    assert s.get_run("run-old").sample is False
    assert s.sample_trace_ids() == set()
