# Deprecated: moved to psalg.slurm.parse_proc_cmdline
import sys

import psalg.slurm.parse_proc_cmdline as _new

sys.modules[__name__] = _new
