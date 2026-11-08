import csv
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("sweep_report", Path(__file__).with_name("report.py"))
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


@pytest.mark.parametrize(
    "y,expected",
    [(9, "QKV GEMM"), (3, "out-proj"), (12, "MLP up GEMM"), (6, "MLP down GEMM"), (66, "lm_head")],
)
def test_gemm_roles(y, expected):
    assert report.group("nvjet_sm90_hsh", [4, y, 1]) == expected


@pytest.mark.parametrize(
    "name,expected",
    [
        ("reshape_and_cache_flash_kernel", "KV write"),
        ("FlashAttnFwdSm90", "attention"),
        ("vectorized_elementwise_kernel", "small elementwise"),
        ("splitKreduce_kernel", "MLP down reduction"),
        ("vectorized_layer_norm_kernel", "layer norm"),
        ("indexSelectSmallIndex", "other setup/index/reduction"),
    ],
)
def test_named_roles(name, expected):
    assert report.group(name, []) == expected


def test_unclassified_gemm_rejected():
    with pytest.raises(ValueError, match="unclassified GEMM"):
        report.group("nvjet_unknown", [1, 99, 1])


def test_empty_csv_does_not_create_file(tmp_path):
    output = tmp_path / "empty.csv"
    report.write_csv(output, [])
    assert not output.exists()


def test_csv_round_trip(tmp_path):
    output = tmp_path / "rows.csv"
    report.write_csv(output, [{"name": "kernel,one", "cycles": 123}])
    with output.open() as stream:
        assert list(csv.DictReader(stream)) == [{"name": "kernel,one", "cycles": "123"}]
