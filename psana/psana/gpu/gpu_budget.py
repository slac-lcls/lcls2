"""
psana/gpu/gpu_budget.py — GPU and pinned-host memory budget.

_GpuBudget tracks committed VRAM (all cp.empty() allocations) and raises
GpuMemoryPressureError before any allocation that would exceed the configured
per-BD limit.  This prevents the silent OOM crash that otherwise propagates
as a confusing MPI broken-pipe error.

Usage
-----
    budget = _GpuBudget(limit_bytes=15 * 1024**3)
    budget.reserve(array_bytes)     # before cp.empty()
    budget.release(array_bytes)     # when replacing or freeing a buffer

Passed to GPUDetector and KvikioGpuReader at construction time.
Created by GpuEvents.__init__; auto-sized to device_total / n_bd_ranks
if gpu_memory_budget_gb is not configured.
"""


class GpuMemoryPressureError(RuntimeError):
    """Raised before a cp.empty() call that would exceed the GPU budget.

    Tells the user which parameter to reduce rather than crashing with
    a cryptic CUDA or MPI error.
    """


class _GpuBudget:
    """Simple committed-bytes counter for GPU VRAM.

    Tracks the total bytes of all active cp.empty() allocations owned by
    this BD rank.  Before any new allocation, reserve() checks the limit
    and optionally frees the CuPy pool to recover cached-but-unused blocks.

    This is intentionally simple: no active-lease byte tracking (correctness
    is enforced by SlotLease.wait_until_safe_to_reuse, not the budget), no
    per-category breakdown (that is covered by GpuEvents.log_memory).
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

    # ------------------------------------------------------------------

    def reserve(self, n: int):
        """Reserve n bytes before calling cp.empty().

        If the budget would be exceeded, first attempts to free cached
        blocks from the CuPy memory pool (which holds freed arrays until
        explicitly released).  Raises GpuMemoryPressureError if there is
        still not enough room after the pool flush.

        Parameters
        ----------
        n : int
            Bytes about to be allocated via cp.empty().
        """
        if self._committed + n <= self._limit:
            self._committed += n
            return

        # Try releasing cached (but unused) CuPy pool blocks first.
        try:
            import cupy as cp

            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

        if self._committed + n > self._limit:
            raise GpuMemoryPressureError(
                f"GPU memory budget exceeded:\n"
                f"  need      {n / 1024**3:.2f} GiB\n"
                f"  committed {self._committed / 1024**3:.2f} GiB\n"
                f"  limit     {self._limit / 1024**3:.2f} GiB\n"
                f"Reduce batch_size or n_gpu_streams, or increase "
                f"gpu_memory_budget_gb."
            )
        self._committed += n

    def release(self, n: int):
        """Return n bytes to the budget (called when a buffer is freed
        or replaced by a smaller/larger allocation)."""
        self._committed = max(0, self._committed - n)

    # ------------------------------------------------------------------

    def available(self) -> int:
        """Bytes remaining before the limit is reached."""
        return max(0, self._limit - self._committed)

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
