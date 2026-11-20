# Deprecated: moved to psalg.slurm.ui_procStat
import sys

import psalg.slurm.ui_procStat as _new

sys.modules[__name__] = _new
