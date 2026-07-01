"""
gpu_mpi.py — MPI + GPU rank management for psana2 GPU BD ranks.

Handles three MPI-specific requirements that must be satisfied before any
GPU kernel work can begin on a BD rank:

  1. GPU pinning  — CUDA_VISIBLE_DEVICES must be set from SLURM_LOCALID
                    BEFORE any CuPy import.  Wrong ordering causes rank N to
                    silently use the wrong device, producing incorrect results
                    or CUDA errors that are very hard to trace.

  2. Communicators — bd_comm (BD-only, for XPCS AllReduce via NCCL) and
                    node_comm (intra-node P2P / shared-memory operations).

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
import types

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared calibration constants (CUDA IPC)
# ---------------------------------------------------------------------------

def share_calib_between_gpu_peers(gpu_detectors, bd_comm, phys_gpu_id):
    """Share peds_gpu/gmask_gpu between BD ranks that map to the same physical GPU.

    Within each group of BD ranks sharing a GPU, the lowest-bd_comm-rank
    member (the "leader") exports CUDA IPC handles for its calibration
    constant buffers.  Follower ranks replace their own ~400 MB allocations
    with non-owning views into the leader's GPU memory.

    Calibration constants are read-only during event processing and change
    only on BeginStep transitions via GPUDetector.beginstep().  Leaders
    update the shared buffer in-place; followers see the change automatically.
    Followers are marked with ``_is_calib_follower=True`` so their
    ``beginstep()`` skips the redundant H→D write and only clears derived caches.

    Parameters
    ----------
    gpu_detectors : dict  {det_name: (psana_det, GPUDetector)}
        From GpuEvents.gpu_detectors — already initialised with peds/gmask.
    bd_comm       : mpi4py.MPI.Comm
        BD-only communicator (bd_rank 0 = EB, bd_rank 1+ = BD workers).
    phys_gpu_id   : int
        Physical GPU index for this rank (the value of CUDA_VISIBLE_DEVICES
        before the per-rank restriction was applied).

    Returns
    -------
    is_leader : bool
        True for the rank that owns the underlying GPU buffers.
        False for followers whose peds_gpu/gmask_gpu are shared views.

    Notes
    -----
    Memory saved: ~400 MB per follower rank (peds_gpu + gmask_gpu freed).
    For N_BD_PER_GPU=2: one follower per GPU → 400 MB × N_GPUS saved.

    beginstep() correctness:
      Leader writes new constants into the shared buffer via .set().
      Followers skip the .set() call (would race-write to shared memory)
      and only clear their _stream_peds/_stream_gmask caches so slices
      get recomputed from the updated shared arrays.
    """
    try:
        import cupy as cp
        from mpi4py import MPI
    except ImportError:
        return True   # no cupy/mpi4py — nothing to share

    # Avoid ALL collectives involving bd_comm (which includes EB at rank 0).
    # EB is in eb_node.start(), not here — any collective on bd_comm deadlocks.
    #
    # Instead, compute peers DETERMINISTICALLY from the GPU assignment formula:
    #   phys_gpu = bd_local_rank % n_gpus_per_node
    # where bd_local_rank = bd_rank - 1 (0-indexed, skipping EB at bd_rank=0).
    # Peers are all BD workers with the same phys_gpu_id, sorted by bd_rank.
    # No MPI communication is needed to discover them.
    #
    # IPC handle exchange uses point-to-point bd_comm.send/recv between the
    # specific peer bd_ranks only — no collective, no sub-communicator.
    my_bd_rank = bd_comm.Get_rank()          # 0=EB, 1..N=BD workers
    n_bd_total = bd_comm.Get_size() - 1      # total BD workers (excl EB)

    if n_bd_total <= 1:
        return True   # only one BD worker — nothing to share

    # Derive n_gpus from SLURM_GPUS_ON_NODE when available.
    # When running outside Slurm (e.g. interactive mpirun without --gres),
    # SLURM_GPUS_ON_NODE is unset and defaults to '1', which would incorrectly
    # group ALL BD ranks as peers of GPU 0 even when they are on different GPUs.
    # Guard: if all BD workers report the same phys_gpu_id (i.e. they all see
    # CUDA_VISIBLE_DEVICES=0 after pinning) and n_gpus == 1, there really is
    # only one GPU — sharing is correct.  If phys_gpu_id varies across ranks
    # (only detectable via a collective, which we avoid) we fall back to no-op.
    # The practical safe default when SLURM_GPUS_ON_NODE is missing is to use
    # the physical GPU id directly: peers are those whose init_gpu_rank() chose
    # the same phys_gpu_id, which is exactly phys_gpu_id == (bd_rank-1) % n_gpus.
    slurm_gpus_set = 'SLURM_GPUS_ON_NODE' in os.environ
    n_gpus = max(1, int(os.environ.get('SLURM_GPUS_ON_NODE', '1')))

    if not slurm_gpus_set and n_bd_total > 1:
        # Cannot safely determine GPU topology without Slurm metadata.
        # Skip IPC sharing rather than risk grouping ranks on different GPUs.
        logger.debug(
            'share_calib_between_gpu_peers: SLURM_GPUS_ON_NODE not set; '
            'skipping IPC sharing to avoid incorrect peer grouping.'
        )
        return True

    my_bd_local = my_bd_rank - 1             # 0-indexed BD worker number

    # BD workers with the same phys_gpu, sorted by bd_rank (ascending).
    peer_bd_ranks = sorted(
        r for r in range(1, bd_comm.Get_size())
        if (r - 1) % n_gpus == phys_gpu_id
    )
    is_leader = (peer_bd_ranks[0] == my_bd_rank)
    n_peers   = len(peer_bd_ranks)

    if n_peers == 1:
        return True   # only one BD worker on this GPU — nothing to share

    IPC_LAZY = cp.cuda.runtime.cudaIpcMemLazyEnablePeerAccess

    for det_name, (_, gpu_det) in gpu_detectors.items():
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
            # Send to each follower using their bd_rank (point-to-point only).
            for follower_bd_rank in peer_bd_ranks[1:]:
                bd_comm.send(meta, dest=follower_bd_rank, tag=42)
        else:
            # Receive from the leader's bd_rank.
            meta = bd_comm.recv(source=peer_bd_ranks[0], tag=42)
            (peds_handle,  gmask_handle,
             peds_shape,   gmask_shape,
             peds_nbytes,  gmask_nbytes) = meta

            peds_ptr  = cp.cuda.runtime.ipcOpenMemHandle(
                peds_handle,  IPC_LAZY
            )
            gmask_ptr = cp.cuda.runtime.ipcOpenMemHandle(
                gmask_handle, IPC_LAZY
            )

            peds_gpu = cp.ndarray(
                peds_shape, dtype=cp.float32,
                memptr=cp.cuda.MemoryPointer(
                    cp.cuda.UnownedMemory(peds_ptr, peds_nbytes, None), 0
                ),
            )
            gmask_gpu = cp.ndarray(
                gmask_shape, dtype=cp.float32,
                memptr=cp.cuda.MemoryPointer(
                    cp.cuda.UnownedMemory(gmask_ptr, gmask_nbytes, None), 0
                ),
            )

            del gpu_det.peds_gpu, gpu_det.gmask_gpu
            gpu_det.peds_gpu           = peds_gpu
            gpu_det.gmask_gpu          = gmask_gpu
            gpu_det._is_calib_follower = True
            gpu_det._stream_peds.clear()
            gpu_det._stream_gmask.clear()

    logger.debug(
        'share_calib_between_gpu_peers: gpu=%d peers=%d role=%s',
        phys_gpu_id, n_peers, 'leader' if is_leader else 'follower',
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


def verify_gpu_pinning(comm):
    """Print each rank's GPU device name to stdout, gathered at rank 0.

    Call once after ``init_gpu_rank()`` and ``import cupy`` to confirm that
    each MPI rank is on the expected device.  Remove (or gate behind a debug
    flag) before production use.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm
        MPI communicator (e.g. ``MPI.COMM_WORLD`` or ``psana_comm``).
    """
    try:
        import cupy as cp
        rank      = comm.Get_rank()
        node      = os.environ.get('SLURMD_NODENAME', 'unknown')
        local_rank = int(os.environ.get('SLURM_LOCALID', 0))
        gpu_id    = os.environ.get('CUDA_VISIBLE_DEVICES', '?')
        dev_name  = cp.cuda.Device(0).name   # always 0 after pinning

        msg = (
            f'rank {rank}: node={node} local_rank={local_rank} '
            f'CUDA_VISIBLE_DEVICES={gpu_id} device={dev_name}'
        )
        all_msgs = comm.gather(msg, root=0)
        if rank == 0 and all_msgs:
            print('\n=== GPU pinning verification ===', flush=True)
            for m in sorted(all_msgs):
                print(m, flush=True)
        comm.Barrier()
    except Exception as exc:
        logger.debug('verify_gpu_pinning failed: %s', exc)


# ---------------------------------------------------------------------------
# 2. Sub-communicators
# ---------------------------------------------------------------------------

def create_gpu_communicators(comm, bd_ranks):
    """Create sub-communicators needed by GPU BD ranks.

    Two communicators are built:

    **bd_comm** — spans all BD ranks.  Used for XPCS cross-GPU AllReduce via
    NCCL.  ``MPI.COMM_NULL`` on non-BD ranks.

    **node_comm** (``MPI.COMM_TYPE_SHARED``) — spans all ranks on the same
    physical node.  Used for intra-node GPU P2P transfers and shared-memory
    operations.

    All ranks in *comm* must call this function collectively.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm
        Global psana communicator (``MPI.COMM_WORLD`` or ``psana_comm``).
    bd_ranks : list[int]
        World ranks of all BD processes.

    Returns
    -------
    types.SimpleNamespace with fields:
        bd_comm   — BD-only communicator (``MPI.COMM_NULL`` on non-BD ranks)
        bd_rank   — rank within bd_comm  (-1 on non-BD ranks)
        node_comm — node-local communicator
        node_rank — rank within node_comm
        is_bd     — True if this rank is a BD rank
    """
    from mpi4py import MPI

    world_rank = comm.Get_rank()
    is_bd      = world_rank in bd_ranks

    # BD-only communicator.
    bd_group = comm.Get_group().Incl(list(bd_ranks))
    bd_comm  = comm.Create_group(bd_group)
    bd_rank  = bd_comm.Get_rank() if is_bd else -1

    # Node-local communicator (processes sharing host memory / NVLink domain).
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    node_rank = node_comm.Get_rank()

    logger.debug(
        'GPU communicators: world_rank=%d is_bd=%s bd_rank=%d node_rank=%d',
        world_rank, is_bd, bd_rank, node_rank,
    )

    return types.SimpleNamespace(
        bd_comm=bd_comm,
        bd_rank=bd_rank,
        node_comm=node_comm,
        node_rank=node_rank,
        is_bd=is_bd,
    )


# ---------------------------------------------------------------------------
# 3. Error handling
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
