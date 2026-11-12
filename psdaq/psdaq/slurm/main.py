# Deprecated: moved to psalg.slurm.main
import sys

import psalg.slurm.main as _new

sys.modules[__name__] = _new
