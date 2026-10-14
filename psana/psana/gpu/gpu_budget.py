"""
psana/gpu/gpu_budget.py — GPU device-memory budget.

_GpuBudget tracks VRAM explicitly reserved by participating input, parser, and
slot-buffer owners and raises GpuMemoryPressureError before a tracked
allocation would exceed the configured per-BD limit. Admission holds reserve
future allocations before I/O; they are converted to committed bytes as the
reader, parser, and detectors allocate. Allocation tokens retain charges for
cached and detached backing until the final owner releases it.

Usage
-----
    budget = _GpuBudget(limit_bytes=15 * 1024**3)
    array = owned_empty(cp, shape, dtype, budget, category='reader')

Use gpu_allocation.owned_empty/upload_owned for pipeline arrays. They charge
the backing allocation through its last alias; trimming a cache must not
manually release its charge. Admission holds reserve future allocation bytes.

Passed to GPUDetector, KvikioGpuReader, and GpuXtcBatchPool at construction.
Created by GpuEventManager.__init__; auto-sized to device_total / n_bd_ranks
if gpu_memory_budget_gb is not configured.
"""


from functools import wraps
from threading import RLock


# Failed asynchronous uploads must survive a constructor failure that drops
# its budget. The budget/array/charge cycle alone is collectible by Python GC.
# Explicit successful draining removes this safety root; no CUDA work runs
# from a finalizer. If draining never succeeds, storage lasts until process exit.
_failed_upload_budgets = set()


def _locked(method):
    @wraps(method)
    def locked(self, *args, **kwargs):
        budget = getattr(self, 'budget', self)
        with budget._lock:
            return method(self, *args, **kwargs)
    return locked


class GpuMemoryPressureError(RuntimeError):
    """Raised before a cp.empty() call that would exceed the GPU budget.

    Tells the user which parameter to reduce rather than crashing with
    a cryptic CUDA or MPI error.
    """


class _AllocationCharge:
    """One allocation's credit; rollback and later destruction are distinct."""

    def __init__(self, budget, capacity, requested, category):
        self.budget = budget
        self.capacity = int(capacity)
        self.origin = budget._active_hold
        self.closed = False
        self.committed = False
        budget.reserve(self.capacity)
        self.allocation_id = budget._next_allocation_id
        budget._next_allocation_id += 1
        budget._allocations[self.allocation_id] = dict(
            allocation_id=self.allocation_id, category=str(category),
            requested=int(requested), capacity=self.capacity)

    @_locked
    def commit(self):
        if self.closed:
            raise RuntimeError('allocation charge is closed')
        self.committed = True
        self.origin = None

    def rollback(self):
        if self.committed:
            raise RuntimeError('cannot roll back a published allocation')
        self._return(self.origin)

    @_locked
    def release(self):
        # An owner may be destroyed while constructing its wrapper. Until
        # publication that destruction is a rollback into the originating hold.
        self._return(None if self.committed else self.origin)

    @_locked
    def _return(self, hold):
        if self.closed:
            return
        if self.capacity > self.budget._committed:
            raise RuntimeError('allocation charge exceeds committed budget')
        self.budget._committed -= self.capacity
        if hold is not None and not hold.closed:
            hold.remaining += self.capacity
            self.budget._held += self.capacity
        del self.budget._allocations[self.allocation_id]
        self.closed = True
        self.origin = None


def allocation_growth_bytes(requirements):
    """Extra capacity for every new/replacement allocation before submission.

    Each pair is (required, existing capacity). Existing allocations remain
    charged while their replacements are allocated; reusable buffers cost zero.
    Old event views can survive several replacements, so reserve all new arrays
    rather than assuming each old array disappears before the next allocation.
    """
    growing = [(need, old) for need, old in requirements if need > old]
    return sum(need for need, _ in growing)


class _GpuAdmissionHold:
    """BD-local reservation activated around input/compute allocation phases."""

    def __init__(self, budget, n):
        self.budget, self.remaining, self.closed = budget, n, False

    @_locked
    def __enter__(self):
        if self.closed or self.budget._active_hold is not None:
            raise RuntimeError("admission hold is closed or another hold is active")
        self.budget._active_hold = self
        return self

    @_locked
    def __exit__(self, *exc):
        self.budget._active_hold = None

    @_locked
    def _consume(self, n):
        if n > self.remaining:
            raise GpuMemoryPressureError(
                f"allocation exceeds admission estimate: need={n}, held={self.remaining}"
            )
        self.remaining -= n
        self.budget._held -= n

    @_locked
    def close(self):
        if self.budget._active_hold is self:
            raise RuntimeError("cannot close active admission hold")
        if not self.closed:
            self.budget._held -= self.remaining
            self.remaining = 0
            self.closed = True


class _GpuBudget:
    """Simple committed-bytes counter for GPU VRAM.

    Tracks explicit allocation ownership and pending admission in this BD rank.
    Lifetime remains enforced by input/result leases. A declared margin covers
    allocator/runtime overhead; arbitrary user CuPy allocations are not tracked.
    """

    def __init__(self, limit_bytes: int):
        """
        Parameters
        ----------
        limit_bytes : int
            Maximum committed VRAM in bytes for this BD rank.
            Typically device_total / n_bd_ranks.
        """
        self._lock = RLock()
        self._limit = limit_bytes
        self._committed = 0
        self._held = 0
        self._active_hold = None
        self._failed_allocations = []
        # Scalar inventory only: the budget must never keep an array alive.
        self._allocations = {}
        self._next_allocation_id = 0

    # ------------------------------------------------------------------

    @_locked
    def reserve_allocation(self, capacity, *, requested, category):
        if requested < 0 or capacity < requested:
            raise ValueError('invalid allocation capacity')
        return _AllocationCharge(self, capacity, requested, category)

    @_locked
    def allocation_snapshot(self):
        """Scalar inventory of all owned blocks, including detached generations."""
        return tuple(dict(record) for record in self._allocations.values())

    @_locked
    def quarantine_upload(self, stream, arrays, sources):
        """Retain unproven upload work even if setup loses its budget owner."""
        self._failed_allocations.append((stream, tuple(arrays), tuple(sources)))
        _failed_upload_budgets.add(self)

    @_locked
    def drain_failed_allocations(self):
        """Retry setup uploads whose stream completion was not established."""
        while self._failed_allocations:
            self._failed_allocations[0][0].synchronize()
            del self._failed_allocations[0]
        _failed_upload_budgets.discard(self)

    @_locked
    def reserve(self, n: int):
        """Charge a new allocation, including the full replacement on growth."""
        if n < 0:
            raise ValueError("cannot reserve negative bytes")
        if self._active_hold is not None:
            self._active_hold._consume(n)
        elif n > self.available():
            raise GpuMemoryPressureError(
                f"GPU memory budget exceeded:\n"
                f"  need      {n / 1024**3:.2f} GiB\n"
                f"  committed {self._committed / 1024**3:.2f} GiB\n"
                f"  held      {self._held / 1024**3:.2f} GiB\n"
                f"  limit     {self._limit / 1024**3:.2f} GiB\n"
                f"Reduce batch_size or n_gpu_streams, or increase "
                f"gpu_memory_budget_gb."
            )
        self._committed += n

    @_locked
    def release(self, n: int):
        """Return n bytes to the budget (called when a buffer is freed
        or replaced by a smaller/larger allocation)."""
        if n < 0:
            raise ValueError("cannot release negative bytes")
        released = min(n, self._committed)
        self._committed -= released
        if self._active_hold is not None:
            self._active_hold.remaining += released
            self._held += released

    @_locked
    def hold(self, n, *, margin=0):
        """Reserve allocation progress before issuing I/O; no CUDA allocation."""
        if n < 0 or margin < 0:
            raise ValueError("negative admission reservation")
        if n + margin > self.available():
            raise GpuMemoryPressureError(
                f"GPU admission needs {n} allocation bytes + {margin} margin; "
                f"committed={self._committed}, held={self._held}, limit={self._limit}"
            )
        self._held += n
        return _GpuAdmissionHold(self, n)

    # ------------------------------------------------------------------

    @_locked
    def available(self) -> int:
        """Bytes remaining before the limit is reached."""
        return max(0, self._limit - self._committed - self._held)

    @_locked
    def allocation_available(self):
        """Credit usable by the currently allocating phase, excluding other holds."""
        if self._active_hold is not None:
            return self._active_hold.remaining
        return self.available()

    @_locked
    def committed(self) -> int:
        """Bytes currently reserved."""
        return self._committed

    @_locked
    def limit(self) -> int:
        """Configured limit in bytes."""
        return self._limit

    @classmethod
    def auto(cls, n_bd_ranks: int = 1) -> "_GpuBudget":
        """Create a budget sized to device_total / n_bd_ranks.

        Falls back to a large sentinel (1 TiB) if CUDA is not available,
        so code paths that call reserve() still work on CPU-only nodes.
        """
        try:
            import cupy as cp

            _, total = cp.cuda.Device().mem_info
            limit = total // n_bd_ranks
        except Exception:
            limit = 1024**4  # 1 TiB sentinel — effectively unlimited
        return cls(limit_bytes=limit)
