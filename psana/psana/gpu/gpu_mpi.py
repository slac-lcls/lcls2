"""
gpu_mpi.py — MPI + GPU rank management for psana2 GPU BD ranks.

Handles MPI-specific requirements that must be satisfied before GPU work can
begin on a BD rank:

  1. GPU pinning  — CUDA_VISIBLE_DEVICES must be set from SLURM_LOCALID
                    BEFORE any CuPy import.  Wrong ordering causes rank N to
                    silently use the wrong device, producing incorrect results
                    or CUDA errors that are very hard to trace.

  2. Error handling — unhandled GPU exceptions on a BD rank cause EB ranks to
                    hang waiting for a receive that will never arrive.
                    comm.Abort(1) lets Slurm detect the failure and free the
                    allocation cleanly.

Typical usage on each BD rank
------------------------------
    # At the top of the analysis script, BEFORE any other psana or CuPy imports:
    from psana.gpu.gpu_mpi import init_gpu_rank
    gpu_id = init_gpu_rank()          # sets CUDA_VISIBLE_DEVICES

    # NOW safe to import CuPy:
    import cupy as cp

    # Then proceed with DataSource as normal:
    from psana import DataSource
    ds = DataSource(exp=..., run=..., gpu_det='jungfrau')
    ...

DataSource integration
----------------------
    When DataSource(gpu_det=...) is used with the MPI backend,
    MPIDataSource.__init__() calls init_gpu_rank() automatically for BD ranks
    (before _setup_run() which may trigger detector imports).  This covers the
    common case where the user does not explicitly call init_gpu_rank().

Reference: psana2 GPU Implementation Guide §2a (MPI Initialisation, GPU
Pinning, and Communicator Setup).
"""

import logging
import os
import sys

logger = logging.getLogger(__name__)


def bd_ranks_sharing_gpu(bd_comm, phys_gpu_id, n_gpus=None):
    """Return how many BD workers in ``bd_comm`` are pinned to ``phys_gpu_id``.

    Used to size the per-rank VRAM budget: every BD worker that shares a
    physical GPU must limit itself to roughly ``device_total / this count``,
    otherwise several ranks each believe they may commit the whole device and
    the first large allocation wins while the rest hit a CUDA OOM.

    The count is derived arithmetically from the same pinning formula that
    ``init_gpu_rank()`` applies — ``phys_gpu_id = bd_local_rank % n_gpus``
    with ``bd_local_rank = bd_rank - 1`` — so no MPI collective is needed.
    That matters because this runs only on BD ranks: a collective here would
    deadlock against the EB and smd0 ranks, which never reach this code.

    Every rank pinned to a given GPU computes the same value, so their budgets
    agree without any communication.

    Parameters
    ----------
    bd_comm     : mpi4py.MPI.Comm  (bd_rank 0 = EB, 1+ = BD workers)
    phys_gpu_id : int  (from init_gpu_rank())
    n_gpus      : int or None
        GPUs on this node.  Read from ``SLURM_GPUS_ON_NODE`` when None.

    Returns
    -------
    int — number of BD workers on ``phys_gpu_id``; always >= 1.

    Notes
    -----
    ``bd_comm`` is split per EB group when ``PS_EB_NODES > 1``, so this counts
    only the peers within this rank's own EB group.  With several EB groups on
    one node the true number of ranks per GPU is higher and the resulting
    budget is correspondingly generous; this limitation is tracked separately.
    """
    if n_gpus is None:
        try:
            n_gpus = int(os.environ.get('SLURM_GPUS_ON_NODE', 1))
        except ValueError:
            n_gpus = 1
    n_gpus = max(1, int(n_gpus))

    try:
        n_bd_total = bd_comm.Get_size() - 1   # bd_rank 0 is the EB
    except Exception:
        return 1
    if n_bd_total <= 0:
        return 1

    target = int(phys_gpu_id) % n_gpus
    # bd_local_rank k (0-indexed BD worker) is pinned to k % n_gpus.
    count = sum(1 for k in range(n_bd_total) if k % n_gpus == target)
    return max(1, count)


def log_gpu_mem(label: str, rank=None) -> None:
    """Log GPU free/used memory at a named checkpoint.

    No-op unless ``PSANA_GPU_MEM_DEBUG`` is set to a non-empty value.
    Useful for tracing which allocation step consumes GPU memory in MPI
    multi-rank runs where OOM errors give only "allocated so far: N GB".

    Usage
    -----
    Set the env var before launching::

        PSANA_GPU_MEM_DEBUG=1 sh scripts/run_mpi_perf_compare.sh ...

    Then grep the output for ``[GPU-MEM]``.

    Parameters
    ----------
    label : str   Short description of the checkpoint.
    rank  : int or None   MPI world rank; included in output when provided.
    """
    if not os.environ.get('PSANA_GPU_MEM_DEBUG'):
        return
    try:
        import cupy as cp
        free_b, total_b = cp.cuda.Device().mem_info
        used_b = total_b - free_b
        dev_id = cp.cuda.Device().id
        rank_s = f' rank={rank}' if rank is not None else ''
        pool_b = cp.get_default_memory_pool().used_bytes()
        print(
            f'[GPU-MEM]{rank_s} dev={dev_id}  '
            f'used={used_b / 1e9:.3f} GB  '
            f'free={free_b / 1e9:.3f} GB  '
            f'pool={pool_b / 1e9:.3f} GB  '
            f'| {label}',
            flush=True,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 1. GPU pinning
# ---------------------------------------------------------------------------

def init_gpu_rank(local_rank=None, n_gpus=None):
    """Pin this MPI rank to the correct GPU device.

    Sets ``os.environ['CUDA_VISIBLE_DEVICES']`` so that when CuPy is imported
    immediately afterward it sees only one device (device 0), which is the
    correct physical GPU for this rank.

    Must be called **before** any ``import cupy`` in the current process.
    If CuPy is already in ``sys.modules`` a warning is emitted but no error is
    raised — the caller is responsible for correct import ordering.

    Parameters
    ----------
    local_rank : int or None
        Intra-node rank (0-based index among tasks on this node).  If None,
        read from ``SLURM_LOCALID``.  Falls back to 0 when neither is set
        (single-GPU or non-Slurm environments).
    n_gpus : int or None
        Number of GPUs on this node.  If None, read from
        ``SLURM_GPUS_ON_NODE``.  Falls back to 1.

    Returns
    -------
    gpu_id : int
        Physical GPU index selected for this rank.  After this call,
        ``os.environ['CUDA_VISIBLE_DEVICES'] == str(gpu_id)``.
    """
    if local_rank is None:
        local_rank = int(os.environ.get('SLURM_LOCALID', 0))
    if n_gpus is None:
        n_gpus = int(os.environ.get('SLURM_GPUS_ON_NODE', 1))

    gpu_id = local_rank % n_gpus

    # Always set CUDA_VISIBLE_DEVICES so that:
    #   (a) if CuPy has not yet been imported, the subsequent import sees only
    #       the correct device (as device 0);
    #   (b) subprocesses and late imports in the same process are also pinned.
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    if 'cupy' in sys.modules:
        # CuPy already imported — CUDA_VISIBLE_DEVICES is set but it is too
        # late to restrict the current process's CUDA context.  Warn if we
        # can detect that the wrong device is active.
        try:
            import cupy as cp
            current = cp.cuda.Device().id
            # After pinning, the visible device is always 0 inside this
            # process (CUDA_VISIBLE_DEVICES remaps physical -> virtual 0).
            if current != 0:
                logger.warning(
                    'init_gpu_rank() called after CuPy was already imported '
                    '(current virtual device=%d, expected 0 after remapping).  '
                    'GPU pinning may be incorrect. Call init_gpu_rank() before '
                    'any CuPy import to guarantee correct device selection.',
                    current,
                )
            else:
                logger.debug(
                    'init_gpu_rank(): CuPy already imported; '
                    'CUDA_VISIBLE_DEVICES set to %d (device 0 in process)',
                    gpu_id,
                )
        except Exception:
            # No CUDA driver available (e.g. login node) or CuPy not
            # functional — silently skip the device check.
            logger.debug(
                'init_gpu_rank(): CuPy imported but CUDA not available; '
                'CUDA_VISIBLE_DEVICES set to %d', gpu_id,
            )
    else:
        logger.debug(
            'GPU pinning: local_rank=%d n_gpus=%d -> CUDA_VISIBLE_DEVICES=%d',
            local_rank, n_gpus, gpu_id,
        )

    return gpu_id


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class gpu_error_handler:
    """Context manager: convert GPU errors into clean ``comm.Abort(1)`` calls.

    Without this, an unhandled exception on a BD rank causes EB ranks to hang
    waiting for an MPI receive that will never arrive.  ``comm.Abort(1)``
    lets Slurm detect the failure immediately, log it cleanly, and free the
    node allocation.

    Usage
    -----
    ::

        with gpu_error_handler(comm):
            for batch_dict, gpu_batch_dict, step_dict \\
                    in eb_manager.batches_with_gpu():
                ...

    Every exception is fatal here.  Nothing is retried: by the time __exit__
    runs, the generator frame that issued the failing read is gone, so a retry
    could only skip the batch and yield silently wrong results.  Live-mode
    retry of a partially written XTC2 file belongs at the KvikIO call site.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm
        Communicator to abort on fatal GPU errors.
    """

    def __init__(self, comm):
        self._comm = comm

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is None:
            return False   # clean exit

        # GeneratorExit is Python's standard generator-cleanup signal, not an
        # error.  It is raised when a generator is GC'd or explicitly closed
        # (e.g. the user breaks out of a for-ctx-in-run.events() loop).  Let
        # it propagate naturally so Python can clean up the generator chain —
        # same behaviour as the CPU path which has no context manager at all.
        if isinstance(exc_val, GeneratorExit):
            return False

        rank = self._comm.Get_rank()

        # --- CUDARuntimeError: unrecoverable ---
        try:
            import cupy as cp
            if isinstance(exc_val, cp.cuda.runtime.CUDARuntimeError):
                print(
                    f'rank {rank}: fatal GPU error: {exc_val}',
                    flush=True,
                )
                self._comm.Abort(1)
                return True  # suppress (Abort will not return)
        except ImportError:
            pass

        # --- KvikIO read failure: fatal ---
        # Note: a context manager cannot retry the failing operation — once
        # __exit__ is called the generator frame that issued the read is gone.
        # Retrying here would silently skip the failed batch and produce
        # incorrect results.  Instead, abort cleanly so Slurm can detect the
        # failure and free the allocation.  Live-mode retry (re-opening the
        # file and re-issuing the read) must be implemented in the KvikIO call
        # site itself, not here.
        if 'KvikIO' in str(exc_val) or 'kvikio' in str(exc_val).lower():
            print(
                f'rank {rank}: fatal KvikIO read error: {exc_val}',
                flush=True,
            )
            self._comm.Abort(1)
            return True  # suppress (Abort will not return)

        # --- All other exceptions: fatal ---
        print(
            f'rank {rank}: fatal error in GPU event loop: '
            f'{exc_type.__name__}: {exc_val}',
            flush=True,
        )
        self._comm.Abort(1)
        return True   # suppress (Abort will not return)
