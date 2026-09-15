"""
psana/gpu/gpu_budget.py — GPU device-memory budget.

_GpuBudget tracks VRAM explicitly reserved by participating input, parser, and
slot-buffer owners and raises GpuMemoryPressureError before a tracked
allocation would exceed the configured per-BD limit. Admission holds reserve
future allocations before I/O; they are converted to committed bytes as the
reader, parser, and detectors allocate. Cached owner capacity remains charged.

Usage
-----
    budget = _GpuBudget(limit_bytes=15 * 1024**3)
    budget.reserve(array_bytes)     # before cp.empty()
    budget.release(array_bytes)     # when replacing or freeing a buffer

Passed to GPUDetector, KvikioGpuReader, and GpuXtcBatchPool at construction.
Created by GpuEventManager.__init__; auto-sized to device_total / n_bd_ranks
if gpu_memory_budget_gb is not configured.
"""


class GpuMemoryPressureError(RuntimeError):
    """Raised before a cp.empty() call that would exceed the GPU budget.

    Tells the user which parameter to reduce rather than crashing with
    a cryptic CUDA or MPI error.
    """


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

    def __enter__(self):
        if self.closed or self.budget._active_hold is not None:
            raise RuntimeError("admission hold is closed or another hold is active")
        self.budget._active_hold = self
        return self

    def __exit__(self, *exc):
        self.budget._active_hold = None

    def _consume(self, n):
        if n > self.remaining:
            raise GpuMemoryPressureError(
                f"allocation exceeds admission estimate: need={n}, held={self.remaining}"
            )
        self.remaining -= n
        self.budget._held -= n

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
        self._limit = limit_bytes
        self._committed = 0
        self._held = 0
        self._active_hold = None
        self._failed_allocations = []

    # ------------------------------------------------------------------

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

    def available(self) -> int:
        """Bytes remaining before the limit is reached."""
        return max(0, self._limit - self._committed - self._held)

    def allocation_available(self):
        """Credit usable by the currently allocating phase, excluding other holds."""
        if self._active_hold is not None:
            return self._active_hold.remaining
        return self.available()

    def committed(self) -> int:
        """Bytes currently reserved."""
        return self._committed

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
