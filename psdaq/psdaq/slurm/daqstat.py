# Deprecated: moved to psalg.slurm.daqstat
import sys

import psalg.slurm.daqstat as _new

sys.modules[__name__] = _new
