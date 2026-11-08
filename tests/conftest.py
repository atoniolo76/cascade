from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
ACCELSIM_DIR = FIXTURES / "accelsim"
LOGS = FIXTURES / "logs"
BABELSTREAM_LOG = LOGS / "babelstream_sm86_rtx3070.log"
H100_LOG = LOGS / "synth_lop3_sm90_h100.log"
OPT_H100_LOG = LOGS / "opt125m_sm90_h100.log"
OPT_H100_STATS = (
    Path(__file__).parent.parent
    / "results/20261108-021007-opt125m-h100-accelsim/outputs/kernel-stats.csv"
)
