"""
gpu_mpi.py — MPI-side support for psana2 GPU BD ranks.

Error handling and memory diagnostics. Unhandled GPU exceptions on a BD rank
cause EB ranks to hang waiting for a receive that will never arrive;
``gpu_error_handler`` converts them into ``comm.Abort(1)`` so Slurm detects the
failure and frees the allocation.

Device selection and peer discovery live in
:mod:`psana.gpu.gpu_placement`. They used to live here as ``init_gpu_rank()``
and ``bd_ranks_sharing_gpu()``, which selected a device by
``local_rank % n_gpus`` and counted peers from the EB-local ``bd_comm``. Both
are gone: the mask write they relied on had no effect once mpi4py had
initialised CUDA, and the EB-local peer count over-committed the device
whenever ``PS_EB_NODES > 1``.
"""

import logging
import os

logger = logging.getLogger(__name__)


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
