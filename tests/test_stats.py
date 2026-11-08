import csv

import pytest

from cascade.stats import LogParser
from tests.conftest import BABELSTREAM_LOG, H100_LOG, OPT_H100_LOG, OPT_H100_STATS

BABELSTREAM_KERNELS = [
    ("_Z11init_kernelIdEvPT_S1_S1_S0_S0_S0_", 6174, 19456),
    ("_Z11copy_kernelIdEvPKT_PS0_", 5923, 11264),
    ("_Z10mul_kernelIdEvPT_PKS0_", 6216, 12288),
    ("_Z10add_kernelIdEvPKT_S2_PS0_", 6312, 14336),
    ("_Z12triad_kernelIdEvPT_PKS0_S3_", 6379, 14336),
    ("_Z10dot_kernelIdEvPKT_S2_PS0_i", 45709, 26488576),
]


def parse(path, clock, chunk):
    lines = path.read_text().splitlines()
    p = LogParser("sim-1", clock)
    stats = []
    for i in range(0, len(lines), chunk):
        stats += p.feed(lines[i : i + chunk])
    stats += p.finish()
    return p, stats


@pytest.mark.parametrize("chunk", [1, 7, 200, 100_000])
def test_babelstream_kernel_blocks(chunk):
    p, stats = parse(BABELSTREAM_LOG, 1132.0, chunk)
    assert [(s.kernel_name, s.gpu_sim_cycle, s.gpu_sim_insn) for s in stats] == BABELSTREAM_KERNELS
    assert [s.kernel_index for s in stats] == list(range(6))
    assert [s.kernel_launch_uid for s in stats] == [1, 2, 3, 4, 5, 6]
    assert stats[0].gpu_ipc == pytest.approx(3.1513)
    assert stats[0].sim_time_ns == pytest.approx(6174 / 1132.0 * 1000)
    assert p.kernels_done == 6
    assert p.current_kernel == "_Z10dot_kernelIdEvPKT_S2_PS0_i"
    assert p.latest.gpu_tot_sim_cycle == 76713
    assert p.latest.gpu_tot_sim_insn == 26560256
    assert p.latest.gpu_tot_ipc == pytest.approx(346.2289)
    assert p.latest.sim_rate_ips == 510774


def test_block_keeps_every_key_value_line():
    _, stats = parse(BABELSTREAM_LOG, 1132.0, 500)
    first = stats[0].stats
    assert first["gpu_tot_issued_cta"] == 1
    assert first["L1D_total_cache_accesses"] == 768
    assert first["L2_BW"] == "4.5060 GB/Sec"
    assert first["gpu_occupancy"] == "52.2433%"
    assert first["Row_Buffer_Locality"] == "-nan"
    assert "Row_Buffer_Locality#2" in first
    assert "gpgpu_simulation_rate" not in first


def test_progress_before_first_block_completes():
    lines = BABELSTREAM_LOG.read_text().splitlines()
    p = LogParser("sim-1", 1132.0)
    assert p.feed(lines[:305]) == []
    assert p.current_kernel == "_Z11init_kernelIdEvPT_S1_S1_S0_S0_S0_"
    assert p.kernels_done == 0
    assert p.latest.gpu_tot_sim_cycle is None


def test_h100_log():
    p, stats = parse(H100_LOG, 1980.0, 200)
    assert [(s.kernel_name, s.kernel_launch_uid, s.gpu_sim_cycle) for s in stats] == [
        ("synth_lop3", 1, 9125)
    ]
    assert stats[0].sim_time_ns == pytest.approx(9125 / 1980.0 * 1000)
    assert p.latest.sim_rate_ips == 755177


def test_unknown_clock_leaves_sim_time_empty():
    _, stats = parse(H100_LOG, None, 200)
    assert stats[0].sim_time_ns is None


def test_real_opt125m_h100_run_matches_accelsims_own_kernel_stats():
    _, stats = parse(OPT_H100_LOG, 1980.0, 4096)
    expected = [
        (int(r["uid"]), r["name"], int(r["gpu_sim_cycle"]), int(r["gpu_sim_insn"]))
        for r in csv.DictReader(OPT_H100_STATS.open())
    ]
    got = [(s.kernel_launch_uid, s.kernel_name, s.gpu_sim_cycle, s.gpu_sim_insn) for s in stats]
    assert got == expected
