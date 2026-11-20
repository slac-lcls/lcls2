# Deprecated: moved to psalg.slurm.subproc
import sys

import psalg.slurm.subproc as _new

sys.modules[__name__] = _new
