# Deprecated: psdaq.slurm moved to psalg.slurm (#206). Each submodule here
# aliases its psalg.slurm counterpart so old imports keep working.
import warnings

warnings.warn(
    "psdaq.slurm has moved to psalg.slurm; please update your imports",
    DeprecationWarning,
    stacklevel=2,
)
