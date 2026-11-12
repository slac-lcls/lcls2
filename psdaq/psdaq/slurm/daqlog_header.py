# Deprecated: moved to psalg.slurm.daqlog_header
import sys

import psalg.slurm.daqlog_header as _new

sys.modules[__name__] = _new
