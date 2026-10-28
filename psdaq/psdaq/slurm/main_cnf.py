# Deprecated: moved to psalg.slurm.main_cnf
import sys

import psalg.slurm.main_cnf as _new

sys.modules[__name__] = _new
