# Deprecated: moved to psalg.slurm.config
import sys

import psalg.slurm.config as _new

sys.modules[__name__] = _new
