from main_cnf import *
from psdaq.slurm.config import Config

config = Config(procmgr_config)
config.select(["timing_0", "control", "teb0", "control_gui", "daqstat"])

config.add(
    {
        host: "drp-srcf-cmp001",
        id: "xtra_python",
        flags: "",
        cmd: f"python /cds/home/m/monarin/lcls2/psdaq/psdaq/slurm/test_multiproc.py",
    }
)
config.rename(["timing_0", "timing_1"], ["teb0", "teb1"])
config.extend(procmgr_ami)
config.show()
