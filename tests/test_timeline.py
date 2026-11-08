import copy
import json
import sys
from pathlib import Path

import pytest

from cascade.cli import main
from cascade.sim.compare import aggregate, compare, error, perfetto, table
from cascade.sim.importer import import_durations
from cascade.sim.timeline import fit_host, load_durations, load_timeline, simulate

FIXTURE = Path(__file__).parent / "fixtures" / "timeline"


@pytest.fixture
def capture():
    return load_timeline(FIXTURE / "corrected_timeline.json")


def write(tmp_path, value, name="data.json"):
    path = tmp_path / name
    path.write_text(json.dumps(value))
    return path


def test_measured_reproduces_baseline_and_does_not_mutate(capture, tmp_path):
    original = copy.deepcopy(capture)
    result = simulate(capture, {})
    assert capture == original
    assert result["steps"][0] == original["steps"][0]
    assert result["requests"][0] == original["requests"][0]
    for real, predicted in zip(capture["steps"][1:], result["steps"][1:], strict=True):
        assert predicted["baseline_wall_ns"] == pytest.approx(real["baseline_wall_ns"], abs=1)
        assert predicted["gpu_busy_ns"] == real["gpu_busy_ns"]
        last = predicted["kernels"][-1]
        assert last["start_offset_ns"] + last["duration_ns"] <= predicted["baseline_wall_ns"]
    assert result["requests"][1] == capture["requests"][1]
    load_timeline(write(tmp_path, result))
    report = compare(capture, result)
    assert report["summary"]["wall_ns"]["mean_abs_error_pct"] == 0
    assert report["requests"][0]["e2e_ms"]["error"] == 0


def test_partial_coverage_and_launch_subtraction(capture, tmp_path):
    imported = import_durations(FIXTURE / "kernel-stats.csv", FIXTURE / "mapping.json", 1980)
    path = write(tmp_path, imported)
    durations = load_durations(path, 5000)
    result = simulate(capture, durations)
    coverage = result["summary"]["simulation"]["coverage"]
    assert coverage["kernels"] == 683
    assert coverage["simulated_kernels"] == 20
    assert coverage["kernel_fraction"] == pytest.approx(20 / 683)
    assert coverage["covered_measured_gpu_ns"] == 72384
    assert coverage["gpu_time_fraction"] == pytest.approx(72384 / coverage["measured_gpu_ns"])
    for old, new in zip(capture["steps"][1:], result["steps"][1:], strict=True):
        assert new["host_ns"] == old["host_ns"]
        assert new["baseline_wall_ns"] == new["gpu_busy_ns"] + old["host_ns"]
        for a, b in zip(old["kernels"], new["kernels"], strict=True):
            if (old["repeat"], old["step"], a["seq"]) not in durations:
                assert a["duration_ns"] == b["duration_ns"]
    assert result["requests"][1]["e2e_ms"] < capture["requests"][1]["e2e_ms"]


def test_parametric_fit_and_constant_count(capture):
    steps = copy.deepcopy(capture["steps"])
    decode = [s for s in steps if not s["warmup"] and s["kind"] == "decode"]
    for i, s in enumerate(decode):
        s["kernels"] = s["kernels"][: i + 1]
        s["host_ns"] = 100 + 50 * len(s["kernels"])
    fit = fit_host(steps)["decode"]
    assert fit["a_ns"] == 100
    assert fit["b_ns_per_kernel"] == 50
    result = simulate(capture, {}, "parametric")
    fit = result["summary"]["simulation"]["host_fit"]["decode"]
    assert fit["slope_identifiable"] is False
    assert fit["b_ns_per_kernel"] == 0
    assert len({s["host_ns"] for s in result["steps"][2:]}) == 1


def test_no_kernels_empty_steps_and_no_decode(capture, tmp_path):
    capture["steps"] = capture["steps"][:2]
    capture["steps"][1]["kernels"] = []
    data = load_timeline(write(tmp_path, capture))
    result = simulate(data, {})
    assert result["steps"][1]["baseline_wall_ns"] == capture["steps"][1]["baseline_wall_ns"]
    assert result["summary"]["simulation"]["coverage"]["kernel_fraction"] == 0
    assert result["requests"][1]["itl_ms_mean"] is None
    assert result["requests"][1]["itl_ms_p50"] is None
    empty = {"meta": {}, "steps": [], "requests": []}
    assert simulate(empty, {})["steps"] == []
    assert compare(empty, empty)["summary"]["wall_ns"]["count"] == 0
    assert aggregate([])["mean_abs_error"] is None
    assert "n/a" in table(compare(empty, empty))


def test_zero_profiled_gaps(capture):
    step = capture["steps"][1]
    end = 0
    for kernel in step["kernels"]:
        kernel["start_offset_ns"] = end
        end += kernel["duration_ns"]
    step["profiled_wall_ns"] = end
    result = simulate(capture, {})
    assert result["steps"][1]["kernels"][0]["start_offset_ns"] > 0
    assert result["steps"][1]["baseline_wall_ns"] == step["baseline_wall_ns"]


def test_non_kernel_gpu_budget_is_preserved(capture, tmp_path):
    step = capture["steps"][1]
    step["gpu_busy_ns"] += 10000
    step["host_ns"] -= 10000
    result = simulate(capture, {})
    assert result["steps"][1]["baseline_wall_ns"] == step["baseline_wall_ns"]
    assert result["steps"][1]["gpu_busy_ns"] == step["gpu_busy_ns"]
    assert result["summary"]["simulation"]["coverage"]["unmodeled_gpu_ns"] == 10000
    first = step["kernels"][0]
    result = simulate(capture, {(0, 0, first["seq"]): first["duration_ns"] / 2})
    assert result["steps"][1]["gpu_busy_ns"] == step["gpu_busy_ns"] - first["duration_ns"] / 2
    step["kernels"][1]["start_offset_ns"] = first["start_offset_ns"] + first["duration_ns"] - 128
    loaded = load_timeline(write(tmp_path, capture))
    result = simulate(loaded, {})
    assert result["steps"][1]["baseline_wall_ns"] == step["baseline_wall_ns"]
    coverage = result["summary"]["simulation"]["coverage"]["steps"][0]
    assert coverage["measured_kernel_overlap_ns"] == 128


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda d: d.pop("meta"), "requires meta"),
        (lambda d: d["steps"].append(copy.deepcopy(d["steps"][1])), "duplicate step"),
        (lambda d: d["steps"][1].update(host_ns=-1), "nonnegative"),
        (lambda d: d["steps"][1].update(host_ns=0), "must equal"),
        (lambda d: d["steps"][1].update(kind="unknown"), "kind"),
        (lambda d: d["steps"][1].update(warmup="false"), "boolean"),
        (lambda d: d["steps"][1].update(step=True), "integer"),
        (lambda d: d["steps"][1]["kernels"][1].update(start_offset_ns=0), "out-of-order"),
        (
            lambda d: d["steps"][1].update(
                gpu_busy_ns=0, host_ns=d["steps"][1]["baseline_wall_ns"]
            ),
            "sum",
        ),
        (lambda d: d["steps"][1].update(profiled_wall_ns=1), "exceed"),
        (lambda d: d["steps"][1]["kernels"][0].update(duration_ns=float("nan")), "finite"),
    ],
)
def test_invalid_capture(capture, tmp_path, mutation, match):
    mutation(capture)
    with pytest.raises(ValueError, match=match):
        load_timeline(write(tmp_path, capture))


@pytest.mark.parametrize(
    "records,subtract,match",
    [
        ([{"seq": 1, "duration_ns": -1}], 0, "nonnegative"),
        ([{"seq": 1, "sim_cycles": 3, "clock_mhz": 0}], 0, "positive"),
        ([{"seq": 1, "sim_cycles": 3, "clock_mhz": 1000}], 5, "exceeds"),
        ([{"seq": 1, "duration_ns": 2}], 1, "requires sim_cycles"),
        ([{"seq": 1, "duration_ns": 2, "repeat": 0}], 0, "both repeat"),
        ([{"seq": 1, "duration_ns": 2}] * 2, 0, "duplicate"),
        ([{"seq": 1, "duration_ns": 2, "sim_cycles": 2}], 0, "not both"),
        ({}, 0, "kernels list"),
    ],
)
def test_invalid_durations(tmp_path, records, subtract, match):
    with pytest.raises(ValueError, match=match):
        load_durations(write(tmp_path, records), subtract)


def test_unmatched_duration_and_bad_requests(capture):
    with pytest.raises(ValueError, match="unmatched"):
        simulate(capture, {(0, 99, 99): 2})
    capture["steps"][1]["num_reqs"] = 2
    with pytest.raises(ValueError, match="num_reqs=1"):
        simulate(capture, {})


def test_generic_duration(capture, tmp_path):
    seq = capture["steps"][1]["kernels"][0]["seq"]
    durations = load_durations(write(tmp_path, [{"seq": seq, "duration_ns": 1234}]))
    assert simulate(capture, durations)["steps"][1]["kernels"][0]["duration_ns"] == 1234
    assert load_durations(None) == {}
    assert load_durations(write(tmp_path, {"kernels": []})) == {}
    capture["steps"][2]["kernels"][0]["seq"] = seq
    with pytest.raises(ValueError, match="ambiguous"):
        simulate(capture, durations)


def test_log_import_and_mapping_errors(tmp_path):
    log = tmp_path / "run.log"
    log.write_text(
        "kernel_name = tiny\nkernel_launch_uid = 7\ngpu_sim_cycle = 23\n"
        "gpgpu_simulation_time = 1\nGPGPU-Sim: *** exit detected ***\n"
    )
    mapping = write(tmp_path, [{"uid": 7, "name": "tiny", "seq": 1}], "mapping.json")
    result = import_durations(log, mapping, 1000)
    assert result["kernels"] == [{"seq": 1, "sim_cycles": 23, "clock_mhz": 1000}]
    bad = write(tmp_path, [{"uid": 7, "name": "wrong", "seq": 1}])
    with pytest.raises(ValueError, match="mismatch"):
        import_durations(log, bad, 1000)
    log.write_text("incomplete")
    with pytest.raises(ValueError, match="completion"):
        import_durations(log, mapping, 1000)
    with pytest.raises(ValueError, match="nonempty list"):
        import_durations(FIXTURE / "kernel-stats.csv", write(tmp_path, []), 1980)


def test_zero_reference_and_missing_request_values(capture):
    assert error(0, 0)["error_pct"] == 0
    assert error(0, 1)["error_pct"] is None
    simulated = simulate(capture, {})
    capture["requests"][1]["itl_ms_mean"] = None
    assert compare(capture, simulated)["requests"][0]["itl_ms_mean"] is None
    simulated["requests"][1]["repeat"] = 99
    with pytest.raises(ValueError, match="repeats differ"):
        compare(capture, simulated)


def test_compare_mismatches_and_quantized_nonoverlap(capture):
    simulated = simulate(capture, {})
    events = perfetto(simulate(capture, {}), simulated)["traceEvents"]
    tracks = {}
    for event in json.loads(json.dumps(events)):
        if event["ph"] == "X":
            track = (event["pid"], event["tid"])
            start = round(event["ts"] * 1000)
            end = start + round(event["dur"] * 1000)
            assert start >= tracks.get(track, 0)
            tracks[track] = end
    simulated["steps"].pop()
    with pytest.raises(ValueError, match="keys differ"):
        compare(capture, simulated)


def test_cli_fixture(tmp_path, monkeypatch, capsys):
    durations = tmp_path / "durations.json"
    output = tmp_path / "simulated.json"
    comparison = tmp_path / "compare.json"
    trace = tmp_path / "trace.json"
    commands = [
        [
            "import-sim-durations",
            "--stats",
            str(FIXTURE / "kernel-stats.csv"),
            "--mapping",
            str(FIXTURE / "mapping.json"),
            "--clock-mhz",
            "1980",
            "--output",
            str(durations),
        ],
        [
            "simulate",
            str(FIXTURE / "corrected_timeline.json"),
            "--durations",
            str(durations),
            "--subtract-launch-cycles",
            "5000",
            "--output",
            str(output),
        ],
        [
            "compare",
            str(FIXTURE / "corrected_timeline.json"),
            str(output),
            "--output",
            str(comparison),
            "--perfetto",
            str(trace),
        ],
    ]
    for command in commands:
        monkeypatch.setattr(sys, "argv", ["cascade", *command])
        main()
    assert "Simulated kernels: 20/683" in capsys.readouterr().out
    assert json.loads(comparison.read_text())["summary"]["wall_ns"]["count"] == 4
    assert json.loads(trace.read_text())["traceEvents"]
    monkeypatch.setattr(sys, "argv", ["cascade", "simulate", str(tmp_path / "missing")])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
