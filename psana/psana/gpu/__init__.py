"""Public API for psana2 GPU acceleration.

GPU processing is normally enabled through :class:`psana.DataSource`::

    from psana import DataSource

    ds = DataSource(exp='mfx100852324', run=77, gpu_det='jungfrau')
    run = next(ds.runs())

    gmd = run.Detector('gmd')
    for evt in run.events():
        calib  = evt.gpu.get('calib').on_gpu         # CuPy, stays on GPU
        energy = gmd.raw.energy(evt)                  # normal psana API
        n_hit  = int(cp.sum(calib > 5.0))
        if n_hit > 100:
            save(evt.gpu.get('jungfrau.calib').on_cpu, energy)

Only the result/context types returned by that interface and the optional
manual GPU-rank initializer are exported here.  Implementation components
should be imported from their defining ``psana.gpu`` submodules.
"""

from psana.gpu.context import GPUResult, GpuEventState
from psana.gpu.gpu_mpi import init_gpu_rank, is_calib_leader


__all__ = [
    "GPUResult",
    "GpuEventState",
    "init_gpu_rank",
    "is_calib_leader",
]
