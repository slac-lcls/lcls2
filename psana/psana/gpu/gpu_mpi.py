"""
gpu_mpi.py — MPI + GPU rank management for psana2 GPU BD ranks.

Handles MPI-specific requirements that must be satisfied before GPU work can
begin on a BD rank:

  1. GPU pinning  — CUDA_VISIBLE_DEVICES must be set from SLURM_LOCALID
                    BEFORE any CuPy import.  Wrong ordering causes rank N to
                    silently use the wrong device, producing incorrect results
                    or CUDA errors that are very hard to trace.

  2. Shared calibration — BD ranks assigned to the same physical GPU can
                          share read-only calibration buffers through CUDA IPC.

  3. Error handling — unhandled GPU exceptions on a BD rank cause EB ranks to
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


# ---------------------------------------------------------------------------
# Shared calibration constants (CUDA IPC)
# ---------------------------------------------------------------------------

def is_calib_leader(bd_comm, phys_gpu_id):
    """Return True if this BD rank should allocate calibration constants.

    Within the group of BD ranks that share ``phys_gpu_id``, only the
    lowest-rank member (the "leader") calls ``prep_calib_constants()``.
    All other ranks (followers) skip allocation entirely and later receive
    non-owning CUDA IPC views from the leader via
    ``share_calib_between_gpu_peers()``.

    This function must be called **before** ``_setup_detectors()`` so that
    followers never allocate ``peds_gpu`` / ``gmask_gpu`` at all.  Allocating
    on all ranks and then deleting on followers creates a peak-allocation window
    of ``n_bd_per_gpu × ~400 MiB`` that can exhaust GPU VRAM before sharing
    ever runs.

    **No SLURM environment variables are needed.**

    ``init_gpu_rank()`` already computed ``phys_gpu_id`` via::

        phys_gpu_id = (bd_local_rank) % n_gpus   # bd_local_rank = bd_rank - 1

    Rearranging, the leader for GPU ``g`` is the BD rank where
    ``bd_rank - 1 == g``, i.e. the first BD worker ever assigned to that GPU
    before any cycling.  No peer enumeration and no SLURM env vars are
    required — just one comparison::

        is_leader = (my_bd_rank - 1 == phys_gpu_id)

    ``share_calib_between_gpu_peers()`` still uses ``SLURM_GPUS_ON_NODE`` to
    enumerate follower bd_ranks for point-to-point IPC handle delivery, but
    that function runs *after* allocation decisions are made.

    Parameters
    ----------
    bd_comm      : mpi4py.MPI.Comm  (bd_rank 0 = EB, 1+ = BD workers)
    phys_gpu_id  : int  (from init_gpu_rank())

    Returns
    -------
    bool — True for the leader (should allocate), False for followers (skip).
    """
    try:
        from mpi4py import MPI  # noqa: F401
    except ImportError:
        return True   # no MPI — single rank, always a leader

    my_bd_rank = bd_comm.Get_rank()
    # bd_rank 0 is EB; BD workers start at 1.  The leader for phys_gpu_id g
    # is the BD worker whose zero-indexed local rank equals g, i.e. bd_rank = g+1.
    return (my_bd_rank - 1) == phys_gpu_id


_TAG_FOLLOWER_REG = 0x4750       # "GP" — follower announces itself to leader
_TAG_IPC_HANDLES  = 0x4750 + 1  # "GP+1" — leader sends IPC handle to follower


def share_calib_between_gpu_peers(gpu_detectors, bd_comm, phys_gpu_id):
    """Share peds_gpu/gmask_gpu between BD ranks that map to the same physical GPU.

    Within each group of BD ranks sharing a GPU, the lowest-bd_comm-rank
    member (the "leader") exports CUDA IPC handles for its calibration
    constant buffers.  Follower ranks receive non-owning views into the
    leader's GPU memory.

    **No SLURM environment variables required.**

    Uses a follower-first registration protocol to discover peers without
    enumerating them from ``n_gpus``:

    1. Every non-leader BD rank sends its own bd_rank to the leader
       (``dest = phys_gpu_id + 1``, tag ``_TAG_FOLLOWER_REG``).  Since all
       BD ranks enter this function at the same code point, all follower
       sends are posted to the MPI network before any leader reaches the
       drain loop.
    2. The leader drains all pending ``_TAG_FOLLOWER_REG`` messages via
       ``Iprobe + recv`` until none remain.  This gives the exact follower
       list without knowing ``n_gpus``.
    3. Leader sends CUDA IPC handles to each registered follower via
       ``_TAG_IPC_HANDLES``.  Followers receive and open the handles.

    Calibration constants are read-only during event processing and change
    only on BeginStep transitions via GPUDetector.beginstep().  Leaders
    update the shared buffer in-place; followers see the change automatically.
    Followers are marked with ``_is_calib_follower=True`` so their
    ``beginstep()`` skips the redundant H→D write and only clears caches.

    Parameters
    ----------
    gpu_detectors : dict  {det_name: (psana_det, GPUDetector)}
        From GpuEventManager.gpu_detectors — already initialised with peds/gmask
        on the leader, and with peds_gpu=gmask_gpu=None on followers
        (is_calib_leader() returned False before _setup_detectors() ran).
    bd_comm       : mpi4py.MPI.Comm
        BD-only communicator (bd_rank 0 = EB, bd_rank 1+ = BD workers).
        No collectives are used — only point-to-point sends and receives
        between the peers.  EB never participates.
    phys_gpu_id   : int
        Physical GPU index for this rank (from init_gpu_rank()).

    Returns
    -------
    is_leader : bool
        True for the rank that owns the underlying GPU buffers.
        False for followers whose peds_gpu/gmask_gpu are shared views.
    """
    try:
        import cupy as cp
        from mpi4py import MPI
    except ImportError:
        return True   # no cupy/mpi4py — nothing to share

    my_bd_rank = bd_comm.Get_rank()       # 0 = EB, 1..N = BD workers
    n_bd_total = bd_comm.Get_size() - 1   # total BD workers (excl EB)

    if n_bd_total <= 1:
        return True   # only one BD worker — nothing to share

    # Leader for GPU g = BD rank (g + 1), consistent with is_calib_leader().
    leader_bd_rank = phys_gpu_id + 1
    is_leader      = (my_bd_rank == leader_bd_rank)

    # ── Phase 1: follower self-registration ─────────────────────────────────
    # Non-leaders post their bd_rank to the leader before ANY leader logic.
    # Because all BD ranks are at the same code point, all sends are in the
    # MPI network buffer by the time the leader's drain loop executes.
    if not is_leader:
        bd_comm.send(my_bd_rank, dest=leader_bd_rank, tag=_TAG_FOLLOWER_REG)

    # ── Synchronise all BD workers before the leader drains registrations ────
    # Guarantee: all follower sends are in the MPI network buffer before the
    # leader's Iprobe loop starts.  Without this, the leader may drain an
    # empty queue (if it runs faster than followers) and return "no followers"
    # even when followers exist.
    #
    # MPI_Comm_create_group is collective over its group only, NOT over
    # bd_comm as a whole.  EB (rank 0) is excluded from the group and must
    # NOT call this — the standard guarantees it will not be involved.
    try:
        _world_grp    = bd_comm.Get_group()
        _bd_grp       = _world_grp.Excl([0])   # exclude EB at rank 0
        _bd_only_comm = bd_comm.Create_group(_bd_grp)
        _bd_only_comm.Barrier()
    except Exception as _exc:
        # Fallback if Create_group is unavailable (rare): brief sleep.
        # 50 ms is sufficient for in-process sends on the same node.
        logger.debug('share_calib: Create_group failed (%s); using sleep fallback', _exc)
        import time as _time
        _time.sleep(0.05)

    if is_leader:
        follower_bd_ranks = []
        while bd_comm.Iprobe(source=MPI.ANY_SOURCE, tag=_TAG_FOLLOWER_REG):
            rank = bd_comm.recv(source=MPI.ANY_SOURCE, tag=_TAG_FOLLOWER_REG)
            follower_bd_ranks.append(rank)
        if not follower_bd_ranks:
            return True   # solo rank on this GPU — no sharing needed

    # ── Phase 2: CUDA IPC handle exchange ───────────────────────────────────
    IPC_LAZY = cp.cuda.runtime.cudaIpcMemLazyEnablePeerAccess

    for det_name, det_info in gpu_detectors.items():
        gpu_det = det_info[1]
        if is_leader:
            peds_handle  = cp.cuda.runtime.ipcGetMemHandle(
                gpu_det.peds_gpu.data.ptr
            )
            gmask_handle = cp.cuda.runtime.ipcGetMemHandle(
                gpu_det.gmask_gpu.data.ptr
            )
            meta = (
                peds_handle,  gmask_handle,
                gpu_det.peds_gpu.shape,  gpu_det.gmask_gpu.shape,
                gpu_det.peds_gpu.nbytes, gpu_det.gmask_gpu.nbytes,
            )
            for follower_bd_rank in follower_bd_ranks:
                bd_comm.send(meta, dest=follower_bd_rank, tag=_TAG_IPC_HANDLES)
        else:
            meta = bd_comm.recv(source=leader_bd_rank, tag=_TAG_IPC_HANDLES)
            (peds_handle,  gmask_handle,
             peds_shape,   gmask_shape,
             peds_nbytes,  gmask_nbytes) = meta

            # Followers arrive here with peds_gpu=None (is_calib_leader()
            # returned False before _setup_detectors(); prep_calib_constants()
            # was never called).  Assert this to catch stale call patterns.
            assert gpu_det.peds_gpu is None and gpu_det.gmask_gpu is None, (
                "share_calib_between_gpu_peers: follower rank already has "
                "peds_gpu allocated.  Call is_calib_leader() before "
                "_setup_detectors() so followers skip prep_calib_constants()."
            )

            peds_ptr  = cp.cuda.runtime.ipcOpenMemHandle(
                peds_handle,  IPC_LAZY
            )
            gmask_ptr = cp.cuda.runtime.ipcOpenMemHandle(
                gmask_handle, IPC_LAZY
            )
            gpu_det.peds_gpu = cp.ndarray(
                peds_shape, dtype=cp.float32,
                memptr=cp.cuda.MemoryPointer(
                    cp.cuda.UnownedMemory(peds_ptr, peds_nbytes, None), 0
                ),
            )
            gpu_det.gmask_gpu = cp.ndarray(
                gmask_shape, dtype=cp.float32,
                memptr=cp.cuda.MemoryPointer(
                    cp.cuda.UnownedMemory(gmask_ptr, gmask_nbytes, None), 0
                ),
            )
            gpu_det._is_calib_follower = True
            gpu_det._stream_peds.clear()
            gpu_det._stream_gmask.clear()

    n_followers = len(follower_bd_ranks) if is_leader else 1
    logger.debug(
        'share_calib_between_gpu_peers: gpu=%d n_followers=%d role=%s',
        phys_gpu_id, n_followers, 'leader' if is_leader else 'follower',
    )
    return is_leader


# ---------------------------------------------------------------------------
# GPU memory logging utility
# ---------------------------------------------------------------------------

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

    Parameters
    ----------
    comm : mpi4py.MPI.Comm
        Communicator to abort on fatal GPU errors.
    max_kvikio_retries : int
        Number of KvikIO read retries before aborting.  Retries are intended
        for live-mode reads where the XTC2 file may still be written by the
        DAQ.  Each retry waits 100 ms × retry_count.
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
