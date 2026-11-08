import copy
import json
from types import SimpleNamespace

import pytest

from cascade.sim.importer import aligned_durations
from cascade.worker import step_timing
from tests.conftest import ACCELSIM_DIR


@pytest.fixture
def aligned(tmp_path):
    steps = []
    for step in range(3):
        steps.append(
            {
                "repeat": 1,
                "step": step,
                "kind": "decode",
                "context_length": 10 + step,
                "warmup": False,
                "baseline_wall_ns": 2000,
                "gpu_busy_ns": 1000,
                "host_ns": 1000,
                "profiled_wall_ns": 1000,
                "kernels": [
                    {
                        "seq": 0,
                        "name": "_Zfoo",
                        "grid": [1, 1, 1],
                        "block": [32, 1, 1],
                        "start_offset_ns": 0,
                        "duration_ns": 1000,
                    }
                ],
            }
        )
    timeline = {"meta": {}, "steps": steps, "requests": []}
    alignment = {
        "steps": [{"repeat": 1, "step": i} for i in (0, 2)],
        "kernels": [
            {
                "repeat": 1,
                "step": step,
                "seq": 0,
                "tracer_uid": uid,
                "name": "_Zfoo",
                "dims_match": True,
            }
            for step, uid in [(0, 6207), (2, 6400)]
        ],
    }
    traces = tmp_path / "traces"
    traces.mkdir()
    listing = traces / "kernelslist.g"
    listing.write_text(
        "MemcpyHtoD,0x1,8\nkernel-6400-ctx_0xabc.tracez\nkernel-6207-ctx_0xabc.tracez\n"
    )
    records = [
        {"uid": 1, "name": "_Zfoo", "sim_cycles": 300},
        {"uid": 2, "name": "_Zfoo", "sim_cycles": 100},
    ]
    return timeline, alignment, listing, records


def test_launch_order_and_interpolation(aligned):
    timeline, alignment, listing, records = aligned
    durations = aligned_durations(records, timeline, alignment, listing, 1000)["kernels"]
    assert {d["step"]: d["sim_cycles"] for d in durations} == {0: 100, 1: 200, 2: 300}
    assert next(d for d in durations if d["step"] == 1)["source"] == "propagated"


@pytest.mark.parametrize("mode", ["empty", "missing", "duplicate", "tracer", "name"])
def test_launch_mismatch_fails(aligned, mode):
    timeline, alignment, listing, records = aligned
    if mode == "empty":
        records = []
    elif mode == "missing":
        records.pop()
    elif mode == "duplicate":
        records[1]["uid"] = 1
    elif mode == "tracer":
        records[0]["uid"] = 6400
    else:
        records[0]["name"] = "_Zwrong"
    with pytest.raises(ValueError, match="positions 1..N|UID/name mapping mismatch"):
        aligned_durations(records, timeline, alignment, listing, 1000)


def test_invalid_alignment_fails(aligned):
    timeline, alignment, listing, records = aligned
    for kernels in ([], alignment["kernels"] * 2, [{**alignment["kernels"][0], "tracer_uid": 9}]):
        invalid = {**alignment, "kernels": kernels}
        with pytest.raises(ValueError, match="alignment tracer UIDs"):
            aligned_durations(records, timeline, invalid, listing, 1000)


def test_worker_posts_aligned_timeline(aligned, tmp_path):
    timeline, alignment, listing, records = aligned
    original = copy.deepcopy(timeline)
    (tmp_path / "corrected_timeline.json").write_text(json.dumps(timeline))
    (tmp_path / "trace_alignment.json").write_text(json.dumps(alignment))
    log = tmp_path / "accelsim.log"
    log.write_text(
        "".join(
            f"kernel_name = {r['name']}\nkernel_launch_uid = {r['uid']}\n"
            f"gpu_sim_cycle = {r['sim_cycles']}\n"
            for r in records
        )
        + "*** exit detected ***\n"
    )
    posts, messages = [], []
    main = SimpleNamespace(post=lambda *args: posts.append(args))
    env = SimpleNamespace(
        trace_dir=tmp_path,
        kernelslist=listing,
        accelsim_dir=ACCELSIM_DIR,
        gpu_config="SM90_H100",
    )
    step_timing(main, env, log, messages.append)
    assert messages == ["cascade worker: step timing posted, 3/3 kernels simulated"]
    assert posts[0][0] == "step-timeline"
    assert len(posts[0][1]["compare"]["steps"]) == 3
    assert timeline == original
    log.write_text(log.read_text().replace("_Zfoo", "_Zwrong"))
    step_timing(main, env, log, messages.append)
    assert len(posts) == 1
    assert "kernel UID/name mapping mismatch" in messages[-1]


def test_worker_posts_recorded_timeline_without_reimporting(tmp_path):
    recorded = tmp_path / "step-timeline.json"
    payload = {"simulated": {"steps": [{"kind": "prefill"}], "summary": {}}, "compare": None}
    recorded.write_text(json.dumps(payload))
    posts, messages = [], []
    main = SimpleNamespace(post=lambda *args: posts.append(args))
    step_timing(main, SimpleNamespace(), tmp_path / "missing.log", messages.append, recorded)
    assert posts == [("step-timeline", payload)]
    assert messages == ["cascade worker: recorded step timing posted"]


@pytest.mark.parametrize("contents", ["", "{", '{"simulated": []}'])
def test_worker_reports_invalid_recorded_timing_without_posting(tmp_path, contents):
    recorded = tmp_path / "step-timeline.json"
    recorded.write_text(contents)
    posts, messages = [], []
    main = SimpleNamespace(post=lambda *args: posts.append(args))
    step_timing(main, SimpleNamespace(), tmp_path / "missing.log", messages.append, recorded)
    assert not posts
    assert messages[0].startswith("cascade worker: step timing failed:")
