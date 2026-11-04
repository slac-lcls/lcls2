# Deprecated: moved to psalg.slurm.gen_gres_conf
import sys

import psalg.slurm.gen_gres_conf as _new

sys.modules[__name__] = _new
