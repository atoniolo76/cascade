import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("a100_report", Path(__file__).with_name("report.py"))
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def test_native_matrix_roles_use_execution_order_not_grid():
    kernels = [{"seq": i, "name": "same_gemv_kernel", "grid": [192, 1, 1]} for i in range(49)]
    kernels += [{"seq": 49, "name": "reshape_and_cache_flash_kernel"}]
    groups = report.groups_for_step(kernels)
    assert [groups[i] for i in range(4)] == ["QKV GEMM", "Out-proj", "MLP up GEMM", "MLP down GEMM"]
    assert groups[48] == "lm_head"
    assert groups[49] == "KV write"


def test_empty_step():
    assert report.groups_for_step([]) == {}


def test_rejects_unrecognized_matrix_layout():
    with pytest.raises(ValueError, match="expected 12"):
        report.groups_for_step([{"seq": 0, "name": "gemm"}])


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("flash_fwd_kernel", "Attention"),
        ("vectorized_layer_norm_kernel", "Layer norm"),
        ("splitKreduce_kernel", "Split-K reduction"),
        ("unrolled_elementwise_kernel", "Small elementwise"),
        ("indexSelectSmallIndex", "Other setup/index/reduction"),
    ],
)
def test_non_matrix_groups(name, expected):
    kernels = [{"seq": i, "name": "gemm"} for i in range(49)]
    kernels.append({"seq": 49, "name": name})
    assert report.groups_for_step(kernels)[49] == expected


def test_group_cycles_to_microseconds():
    row = report.summarize_group(
        "Attention", [((1, 32, 0), 2)], {(1, 32, 0): {"sim_cycles": 2820, "clock_mhz": 1410}}
    )
    assert row["sim_us"] == 2 and row["signed_error_pct"] == 0
    assert row["n"] == row["modeled_n"] == 1


def test_empty_group():
    assert report.summarize_group("Attention", [], {}) is None


def test_missing_predictions_are_not_reported_as_simulated():
    row = report.summarize_group("lm_head", [((1, 32, 0), 52)], {})
    assert row["modeled_n"] == 0 and row["n"] == 1
    assert row["cycles"] is row["sim_us"] is row["signed_error_pct"] is None


def test_rejects_mixed_coverage_mean():
    with pytest.raises(ValueError, match="partial within-group coverage"):
        report.summarize_group(
            "lm_head",
            [((1, 32, 0), 52), ((1, 34, 0), 53)],
            {(1, 32, 0): {"sim_cycles": 1410, "clock_mhz": 1410}},
        )
