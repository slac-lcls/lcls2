"""
Utilities for managing MPI shared-memory buffers for psana.

The helper is responsible for carving out shared arrays that can be accessed by
all ranks running on the same physical node. It hides the MPI window plumbing
and always zeroes freshly created buffers on the leader rank before exposing
them to the rest of the node.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple

import numpy as np

from psana.psexp.tools import mode

if mode == "mpi":
    from mpi4py import MPI
else:
    MPI = None  # type: ignore[assignment]


@dataclass
class SharedArrayHandle:
    """Metadata describing one shared-memory allocation."""

    name: str
    shape: Tuple[int, ...]
    dtype: np.dtype
    array: np.ndarray
    window: Any  # MPI.Win once mpi4py is available


class MPISharedMemory:
    """
    Helper that creates and tracks MPI shared-memory windows per compute node.

    Typical usage:
        helper = MPISharedMemory()
        offsets = helper.allocate_array("offsets", (n_slots, n_streams, chunk_events, 2))
        state = helper.allocate_array("state", (n_slots,), np.int32)
        ...
        helper.close()
    """

    def __init__(self, base_comm=None, shm_comm=None):
        if mode != "mpi":
            raise RuntimeError("Marching shared memory requires MPI mode")
        if shm_comm is not None:
            self.shm_comm = shm_comm
        else:
            self._base_comm = base_comm or MPI.COMM_WORLD
            rank = self._base_comm.Get_rank()
            info = MPI.INFO_NULL
            self.shm_comm = self._base_comm.Split_type(MPI.COMM_TYPE_SHARED, rank, info)
        self.rank = self.shm_comm.Get_rank()
        self.size = self.shm_comm.Get_size()
        self.is_leader = self.rank == 0
        self._handles: dict[str, SharedArrayHandle] = {}

    def allocate_array(
        self,
        name: str,
        shape: Iterable[int],
        dtype: Any = np.int64,
        zero_init: bool = True,
    ) -> np.ndarray:
        """
        Allocate a shared-memory numpy array visible to all ranks on the node.

        Parameters
        ----------
        name:
            Logical identifier for the allocation. Must be unique per helper.
        shape:
            Iterable describing the shape of the array.
        dtype:
            Any dtype understood by numpy (default: np.int64).
        zero_init:
            Whether to zero the buffer on the leader rank before use.
        """
        if name in self._handles:
            raise ValueError(f"Shared array '{name}' already exists")

        dtype = np.dtype(dtype)
        shape_tuple = tuple(int(dim) for dim in shape)
        if not shape_tuple:
            raise ValueError("Shared arrays must have at least one dimension")

        n_items = int(np.prod(shape_tuple))
        local_bytes = n_items * dtype.itemsize if self.is_leader else 0
        win = MPI.Win.Allocate_shared(
            local_bytes,
            dtype.itemsize if local_bytes else dtype.itemsize,
            comm=self.shm_comm,
        )
        buf, _ = win.Shared_query(0)
        array = np.ndarray(shape=shape_tuple, dtype=dtype, buffer=buf)
        if zero_init and self.is_leader:
            array.fill(0)
        self.shm_comm.Barrier()

        handle = SharedArrayHandle(
            name=name,
            shape=shape_tuple,
            dtype=dtype,
            array=array,
            window=win,
        )
        self._handles[name] = handle
        return array

    def get_array(self, name: str) -> np.ndarray:
        """Return the numpy view associated with an allocation."""
        return self._handles[name].array

    def has_array(self, name: str) -> bool:
        """Return True if an allocation with this name exists."""
        return name in self._handles

    def get_handle(self, name: str) -> SharedArrayHandle:
        """Return the full handle (including MPI window) for an allocation."""
        return self._handles[name]

    def free(self, name: str) -> None:
        """Release an allocation created with allocate_array()."""
        handle = self._handles.pop(name, None)
        if not handle:
            return
        handle.window.Free()

    def barrier(self) -> None:
        """Convenience wrapper around the shared-memory communicator barrier."""
        self.shm_comm.Barrier()

    def close(self) -> None:
        """Release all outstanding shared-memory windows."""
        for name in list(self._handles):
            self.free(name)
