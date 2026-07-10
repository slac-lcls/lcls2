"""
psana.gpu — GPU-accelerated Jungfrau calibration for psana2.

Minimal API
-----------
    from psana import DataSource
    from psana.gpu import prep_calib_constants, fused_calib_gpu
    import cupy as cp

    ds  = DataSource(exp='mfx101572426', run=47)
    run = next(ds.runs())
    det = run.Detector('jungfrau')

    peds_gpu, gmask_gpu = prep_calib_constants(det)   # once per run

    for evt in run.events():
        raw = det.raw.raw(evt)
        if raw is None:
            continue
        calib_gpu = fused_calib_gpu(cp.asarray(raw), peds_gpu, gmask_gpu)
        # calib_gpu: CuPy float32, same shape as raw
        # call .get() only when CPU data is needed

MPI GPU pinning
---------------
Call init_gpu_rank() BEFORE importing cupy in any multi-rank script:

    from psana.gpu import init_gpu_rank
    gpu_id = init_gpu_rank()    # sets CUDA_VISIBLE_DEVICES from SLURM_LOCALID
    import cupy as cp
"""

import logging as _logging
import os as _os
import sys as _sys

from psana.gpu.gpu_calib import prep_calib_constants, fused_calib_gpu

__all__ = [
    "prep_calib_constants",
    "fused_calib_gpu",
    "init_gpu_rank",
]

_logger = _logging.getLogger(__name__)


def init_gpu_rank(local_rank=None, n_gpus=None):
    """Pin this MPI rank to the correct GPU before importing CuPy.

    Sets CUDA_VISIBLE_DEVICES = local_rank % n_gpus so the subsequent
    CuPy import sees only one device (device 0).

    Must be called BEFORE any ``import cupy`` in the process.

    Parameters
    ----------
    local_rank : int or None  — intra-node rank; reads SLURM_LOCALID if None
    n_gpus     : int or None  — GPUs on node; reads SLURM_GPUS_ON_NODE if None

    Returns
    -------
    gpu_id : int  — physical GPU index selected
    """
    if local_rank is None:
        local_rank = int(_os.environ.get("SLURM_LOCALID", 0))
    if n_gpus is None:
        n_gpus = int(_os.environ.get("SLURM_GPUS_ON_NODE", 1))

    gpu_id = local_rank % n_gpus
    _os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if "cupy" in _sys.modules:
        _logger.warning(
            "init_gpu_rank() called after CuPy was already imported — "
            "CUDA_VISIBLE_DEVICES set to %d but device binding may be wrong. "
            "Call init_gpu_rank() before any CuPy import.",
            gpu_id,
        )
    else:
        _logger.debug("GPU pinning: local_rank=%d n_gpus=%d -> device %d",
                      local_rank, n_gpus, gpu_id)
    return gpu_id
