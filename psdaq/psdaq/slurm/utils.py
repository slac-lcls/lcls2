# Deprecated: moved to psalg.slurm.utils
import sys

import psalg.slurm.utils as _new

sys.modules[__name__] = _new
