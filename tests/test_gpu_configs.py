from cascade.gpu_configs import GpuConfigRegistry
from tests.conftest import ACCELSIM_DIR


def test_registry_reads_real_tested_cfgs():
    reg = GpuConfigRegistry(ACCELSIM_DIR)
    got = {c.name: (c.arch, c.core_clock_mhz) for c in reg.all()}
    assert got == {
        "SM7_QV100": ("sm_70", 1132.0),
        "SM80_A100": ("sm_80", 1410.0),
        "SM86_RTX3070": ("sm_86", 1132.0),
        "SM90_H100": ("sm_90", 1980.0),
        "SM90_H200": ("sm_90", 1980.0),
    }
    assert reg.get("SM2_GTX480") is None


def test_missing_accelsim_dir_gives_empty_registry(tmp_path):
    assert GpuConfigRegistry(tmp_path / "nope").all() == []
