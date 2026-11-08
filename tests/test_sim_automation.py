import json
import lzma
import sys
from pathlib import Path

import pytest

from cascade.cli import main
from cascade.sim.compare import compare
from cascade.sim.importer import import_durations
from cascade.sim.mapping import build_mapping, trace_headers
from cascade.sim.pipeline import run_pipeline
from cascade.sim.runner import config_clock, prepare_run
from cascade.sim.timeline import load_durations, load_timeline, simulate

FIXTURE = Path(__file__).parent / "fixtures/timeline"
ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path.write_text(json.dumps(value))
    return path


def traces(directory, kernels, suffix=".tracez"):
    names = []
    for i, kernel in enumerate(kernels, 1):
        name = f"kernel-{i}{suffix}"
        text = (
            f"-kernel name = {kernel['name']}\n-kernel id = {kernel.get('trace_id', i)}\n"
            f"-grid dim = ({','.join(map(str, kernel['grid']))})\n"
            f"-block dim = ({','.join(map(str, kernel['block']))})\n"
        )
        if suffix.endswith(".xz"):
            (directory / name).write_bytes(lzma.compress(text.encode()))
        else:
            (directory / name).write_bytes(text.encode() + b"\xff\x00binary payload")
        names.append(name)
    path = directory / "kernelslist.g"
    path.write_text("# launches\nMemcpyHtoD,0,4\n" + "\n".join(names) + "\n")
    return path


@pytest.fixture
def mapping_inputs(tmp_path):
    data = load_timeline(FIXTURE / "corrected_timeline.json")
    data["steps"] = data["steps"][:2]
    kernel = data["steps"][1]["kernels"][0]
    kernel.update(name="tiny", grid=[1, 1, 1], block=[32, 1, 1])
    headers = [
        dict(name="tiny", grid=[1, 1, 1], block=[32, 1, 1], trace_id=17 + i) for i in range(3)
    ]
    kernelslist = traces(tmp_path, headers)
    stats = tmp_path / "kernel-stats.csv"
    stats.write_text("uid,name,gpu_sim_cycle\n1,tiny,10\n2,tiny,31\n3,tiny,20\n")
    phases = save(tmp_path / "phases.json", [dict(uid=i, kind="prefill") for i in (1, 2, 3)])
    timeline = save(tmp_path / "timeline.json", data)
    return data, kernelslist, stats, phases, timeline


@pytest.mark.parametrize("suffix", [".tracez", ".trace", ".trace.xz"])
def test_trace_header_formats(tmp_path, suffix):
    kernel = dict(name="tiny", grid=[2, 1, 1], block=[32, 1, 1], trace_id=9)
    assert trace_headers(traces(tmp_path, [kernel], suffix)) == [kernel]


def test_transfer_median_import_and_cli(mapping_inputs, tmp_path, monkeypatch):
    _, kernelslist, stats, phases, timeline = mapping_inputs
    result = build_mapping(
        stats, kernelslist, timeline, 1980, phases_path=phases, allow_shape_transfer=True
    )
    entry = result["mapping"][0]
    assert entry["method"] == "shape_transfer"
    assert entry["source_uids"] == [1, 2, 3]
    assert result["durations"]["kernels"][0]["sim_cycles"] == 20
    mapping = save(tmp_path / "mapping.json", result["mapping"])
    assert import_durations(stats, mapping, 1980)["kernels"] == result["durations"]["kernels"]
    entry["source_uids"] = [1, 2]
    save(mapping, [entry])
    assert import_durations(stats, mapping, 1980)["kernels"][0]["sim_cycles"] == 20.5
    output = tmp_path / "durations.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cascade",
            "import-sim-durations",
            "--stats",
            str(stats),
            "--kernelslist",
            str(kernelslist),
            "--timeline",
            str(timeline),
            "--source-phases",
            str(phases),
            "--allow-shape-transfer",
            "--clock-mhz",
            "1980",
            "--mapping-output",
            str(mapping),
            "--output",
            str(output),
        ],
    )
    main()
    assert json.loads(output.read_text()) == result["durations"]


def test_capture_identity_and_explicit_fallback(mapping_inputs, tmp_path):
    data, kernelslist, stats, phases, timeline = mapping_inputs
    with pytest.raises(ValueError, match="no kernels matched"):
        build_mapping(stats, kernelslist, timeline, 1980, phases_path=phases)
    step = data["steps"][1]
    kernel = step["kernels"][0]
    kernel["trace_kernel_id"] = 17
    save(timeline, data)
    result = build_mapping(stats, kernelslist, timeline, 1980)
    assert result["mapping"][0]["method"] == "identity"
    assert result["durations"]["kernels"][0]["sim_cycles"] == 10
    del kernel["trace_kernel_id"]
    save(timeline, data)
    sidecar = save(
        tmp_path / "identity.json",
        [dict(uid=2, name="tiny", repeat=step["repeat"], step=step["step"], seq=kernel["seq"])],
    )
    assert (
        build_mapping(stats, kernelslist, timeline, 1980, identity_path=sidecar)["durations"][
            "kernels"
        ][0]["sim_cycles"]
        == 31
    )
    kernel["block"] = [64, 1, 1]
    save(timeline, data)
    with pytest.raises(ValueError, match="geometry/name"):
        build_mapping(stats, kernelslist, timeline, 1980, identity_path=sidecar)


def test_empty_and_bad_sources(mapping_inputs, tmp_path):
    _, kernelslist, stats, phases, timeline = mapping_inputs
    kernelslist.write_text("")
    with pytest.raises(ValueError, match="nonempty"):
        trace_headers(kernelslist)
    broken = tmp_path / "broken.trace"
    broken.write_text("-kernel id = 1\n")
    kernelslist.write_text(broken.name)
    with pytest.raises(ValueError, match="missing trace header"):
        trace_headers(kernelslist)
    kernelslist = traces(tmp_path, [dict(name="wrong", grid=[1, 1, 1], block=[32, 1, 1])])
    with pytest.raises(ValueError, match="one launch UID"):
        build_mapping(stats, kernelslist, timeline, 1980, phases_path=phases)


def test_regenerate_gaps_final_transfer(tmp_path):
    expected = json.loads((ROOT / "results/gaps-final/sim_durations.json").read_text())
    headers = {}
    for group in expected["source_groups"]:
        for uid in group["source_uids"]:
            headers[uid] = dict(name=group["name"], grid=group["grid"], block=group["block"])
    kernelslist = traces(tmp_path, [headers[i] for i in sorted(headers)])
    result = build_mapping(
        FIXTURE / "kernel-stats.csv",
        kernelslist,
        ROOT / "results/gaps-final/corrected_timeline.json",
        1980,
        phases_path=FIXTURE / "h100_source_phases.json",
        allow_shape_transfer=True,
    )
    assert result["durations"]["source_groups"] == expected["source_groups"]
    assert result["durations"]["kernels"] == expected["kernels"]
    assert result["coverage"]["simulated_kernels"] == 5856
    mapping = save(tmp_path / "kernel_mapping.json", result["mapping"])
    assert (
        import_durations(FIXTURE / "kernel-stats.csv", mapping, 1980)["kernels"]
        == expected["kernels"]
    )


def test_pipeline_cli_and_empty(tmp_path, monkeypatch):
    durations = save(
        tmp_path / "durations.json",
        import_durations(FIXTURE / "kernel-stats.csv", FIXTURE / "mapping.json", 1980),
    )
    output = tmp_path / "batch"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cascade",
            "simulate-batch",
            str(FIXTURE / "corrected_timeline.json"),
            "--durations",
            str(durations),
            "--output",
            str(output),
        ],
    )
    main()
    real = load_timeline(FIXTURE / "corrected_timeline.json")
    for mode in ("measured-only", "accelsim"):
        for model in ("measured", "mean", "linear"):
            prediction = load_timeline(output / mode / model / "simulated_timeline.json")
            expected = simulate(
                real, {} if mode == "measured-only" else load_durations(durations), gap_model=model
            )
            expected["summary"]["simulation"]["subtract_launch_cycles"] = 0
            assert prediction == expected
            assert json.loads((output / mode / model / "compare.json").read_text()) == compare(
                real, prediction
            )
    assert json.loads((output / "sanity.json").read_text())["request_sanity_pass"]
    empty = save(tmp_path / "empty.json", dict(meta={}, steps=[], requests=[]))
    empty_durations = save(tmp_path / "empty-durations.json", {"kernels": []})
    assert not run_pipeline(empty, empty_durations, tmp_path / "empty-batch")["e2e_itl_pass"]
    with pytest.raises(ValueError, match="nonnegative"):
        run_pipeline(empty, durations, tmp_path / "invalid-batch", -1)


@pytest.fixture
def framework(tmp_path):
    base = tmp_path / "framework/gpu-simulator"
    gp = base / "gpgpu-sim/configs/tested-cfgs/GPU"
    trace = base / "configs/tested-cfgs/GPU"
    binary = base / "bin/release/accel-sim.out"
    for parent in (gp, trace, binary.parent):
        parent.mkdir(parents=True)
    binary.write_text("simulator fixture")
    binary.chmod(0o755)
    (gp / "gpgpusim.config").write_text(
        "-gpgpu_clock_domains 1980:3960:3960:3106\n"
        "-gpgpu_kernel_launch_latency 5000\n"
        "-inter_config_file fabric.icnt\n"
    )
    (gp / "fabric.icnt").write_text("fabric")
    (trace / "trace.config").write_text("# trace config\n")
    kernelslist = tmp_path / "kernelslist.g"
    kernelslist.write_text("kernel.trace\n")
    return base.parent, kernelslist, gp


def test_prepare_run_and_cli(framework, tmp_path, monkeypatch):
    root, kernelslist, gp = framework
    out = tmp_path / "run with spaces"
    plan = prepare_run(root, "GPU", kernelslist, out)
    assert (out / "fabric.icnt").read_bytes() == (gp / "fabric.icnt").read_bytes()
    assert plan["clock_mhz"] == 1980
    assert plan["launch_latency_cycles"] == "5000"
    assert "-gpgpu_kernel_launch_latency" not in plan["command"]
    assert plan["command"][2] == str(kernelslist)
    with pytest.raises(ValueError, match="empty"):
        prepare_run(root, "GPU", kernelslist, out)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cascade",
            "prepare-sim-run",
            "--framework",
            str(root),
            "--config-name",
            "GPU",
            "--kernelslist",
            str(kernelslist),
            "--run-dir",
            str(tmp_path / "cli-run"),
            "--output",
            str(tmp_path / "plan.json"),
        ],
    )
    main()
    assert json.loads((tmp_path / "plan.json").read_text())["clock_mhz"] == 1980


@pytest.mark.parametrize("name", ["", "..", "../GPU", "missing"])
def test_bad_config_name(framework, tmp_path, name):
    root, kernelslist, _ = framework
    with pytest.raises(ValueError):
        prepare_run(root, name, kernelslist, tmp_path / "run")


def test_missing_companion_and_clock(framework, tmp_path):
    root, kernelslist, gp = framework
    (gp / "fabric.icnt").unlink()
    with pytest.raises(ValueError, match="missing interconnect"):
        prepare_run(root, "GPU", kernelslist, tmp_path / "run")
    (gp / "gpgpusim.config").write_text("# no clock\n")
    with pytest.raises(ValueError, match="missing -gpgpu_clock_domains"):
        config_clock(gp / "gpgpusim.config")
