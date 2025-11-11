# run_ctx.py
from __future__ import annotations
from psana import utils

class RunCtx:
    """
    Minimal, acyclic run context to attach to Event objects.

    Exposes only:
      - expt (str)
      - runnum (int)
      - timestamp (int)
      - intg_det (str): alias for integrating detector, if any

    Any attempt to access an unknown attribute logs a WARNING and raises AttributeError.
    """
    __slots__ = ("expt", "runnum", "timestamp", "intg_det", "_log")

    def __init__(self, expt: str, runnum: int, timestamp: int, intg_det: str | None = None) -> None:
        self.expt   = expt
        self.runnum = runnum
        self.timestamp = timestamp
        self.intg_det = intg_det
        self._log = utils.get_logger(name=utils.get_class_name(self))

    def __repr__(self) -> str:
        return f"RunCtx(expt={self.expt!r}, runnum={self.runnum!r})"

    def __getattr__(self, name: str):
        # Called only if normal attribute lookup fails
        self._log.warning(
            "RunCtx: attribute '%s' is not available. Code likely expected a full Run. "
            "Use evt.ctx.expt / evt.ctx.runnum or pass needed data explicitly.",
            name,
        )
        raise AttributeError(f"RunCtx has no attribute '{name}'")

    def __bool__(self) -> bool:
        return True
