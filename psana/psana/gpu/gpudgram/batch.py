"""Slot-owned device tables for GPU-resident XTC batches."""

from dataclasses import dataclass, field

import numpy as np

from psana.gpu.gpu_kvikio_read import (
    DESC_DEVICE_OFFSET,
    DESC_EVENT_INDEX,
    DESC_NCOLS,
    DESC_READ_SIZE,
    DESC_STREAM_ID,
)


# One row per big-data dgram.  The read path supplies the first four columns;
# the GPU walker fills the remainder in place.
DGRAM_EVENT_INDEX = 0
DGRAM_STREAM_ID = 1
DGRAM_OFFSET = 2
DGRAM_SIZE = 3
DGRAM_TIMESTAMP = 4
DGRAM_ENV = 5
DGRAM_SERVICE = 6
DGRAM_DAMAGE = 7
DGRAM_TYPE = 8
DGRAM_STATUS = 9
DGRAM_NCOLS = 10

# One row per ShapesData reference discovered by the walker.
REF_DGRAM_INDEX = 0
REF_CONFIG_NAMES_INDEX = 1
REF_OFFSET = 2
REF_EXTENT = 3
REF_DAMAGE = 4
REF_NCOLS = 5

# One row per dgram for each run-scoped field handle requested by consumers.
LOC_CONFIG_FIELD_INDEX = 0
LOC_TYPE = 1
LOC_RANK = 2
LOC_DIM0 = 3
LOC_MAX_RANK = 5
LOC_OFFSET = LOC_DIM0 + LOC_MAX_RANK
LOC_NBYTES = LOC_OFFSET + 1
LOC_STATUS = LOC_NBYTES + 1
LOC_NCOLS = LOC_STATUS + 1


def build_dgram_records(desc_table):
    """Translate dense KvikIO descriptors to walker input records.

    This is a metadata-only CPU operation.  It preserves descriptor order and
    copies the event index, stream id, packed-device offset, and byte size.
    The XTC header fields remain zero until the GPU walker fills them.
    """
    desc_table = np.asarray(desc_table)
    if desc_table.dtype != np.uint64 or desc_table.ndim != 2:
        raise TypeError("desc_table must be a 2-dimensional uint64 array")
    if desc_table.shape[1] != DESC_NCOLS:
        raise ValueError(
            f"desc_table must have shape (n, {DESC_NCOLS}), "
            f"got {desc_table.shape}"
        )

    records = np.zeros((len(desc_table), DGRAM_NCOLS), dtype=np.uint64)
    records[:, DGRAM_EVENT_INDEX] = desc_table[:, DESC_EVENT_INDEX]
    records[:, DGRAM_STREAM_ID] = desc_table[:, DESC_STREAM_ID]
    records[:, DGRAM_OFFSET] = desc_table[:, DESC_DEVICE_OFFSET]
    records[:, DGRAM_SIZE] = desc_table[:, DESC_READ_SIZE]
    return records


@dataclass
class _GpuXtcSlotBuffers:
    """Reusable parser buffers whose lifetime matches one EventPool slot."""

    cp: object
    budget: object = None
    dgram_records: object = None
    shape_counts: object = None
    shape_refs: object = None
    locators: dict = field(default_factory=dict)

    def _rows(self, existing, n_rows, row_shape):
        required_shape = (int(n_rows),) + tuple(row_shape)
        if existing is not None and existing.shape[0] >= n_rows:
            return existing, existing[:n_rows]

        old_nbytes = int(existing.nbytes) if existing is not None else 0
        required_nbytes = int(np.prod(required_shape, dtype=np.int64)) * 8
        delta = required_nbytes - old_nbytes
        if self.budget is not None:
            self.budget.reserve(delta)
        try:
            replacement = self.cp.empty(required_shape, dtype=self.cp.uint64)
        except Exception:
            if self.budget is not None:
                self.budget.release(delta)
            raise
        return replacement, replacement

    def prepare(self, desc_table, max_shapes_per_dgram, stream):
        records_host = build_dgram_records(desc_table)
        n_dgrams = len(records_host)
        self.dgram_records, records = self._rows(
            self.dgram_records, n_dgrams, (DGRAM_NCOLS,)
        )
        self.shape_counts, shape_counts = self._rows(
            self.shape_counts, n_dgrams, ()
        )
        self.shape_refs, shape_refs = self._rows(
            self.shape_refs,
            n_dgrams,
            (max_shapes_per_dgram, REF_NCOLS),
        )
        records.set(records_host, stream=stream)
        return records, shape_counts, shape_refs

    def locator_rows(self, handle, n_dgrams):
        backing, rows = self._rows(
            self.locators.get(handle), n_dgrams, (LOC_NCOLS,)
        )
        self.locators[handle] = backing
        return rows

    @property
    def memory_bytes(self):
        arrays = (self.dgram_records, self.shape_counts, self.shape_refs)
        return sum(int(array.nbytes) for array in arrays if array is not None) + sum(
            int(array.nbytes) for array in self.locators.values()
        )


class GpuXtcBatchPool:
    """Run-scoped Configure tables and reusable per-slot parser storage."""

    def __init__(
        self,
        configs,
        *,
        field_handles=(),
        n_slots=2,
        max_shapes_per_dgram=None,
        budget=None,
        cp=None,
    ):
        if cp is None:
            import cupy as cp

        self.cp = cp
        self.configs = configs
        self.field_handles = tuple(field_handles)
        self.n_slots = int(n_slots)
        if self.n_slots <= 0:
            raise ValueError("n_slots must be positive")

        if max_shapes_per_dgram is None:
            per_stream = np.diff(configs.stream_names_index)
            max_shapes_per_dgram = max(1, int(per_stream.max(initial=0)))
        self.max_shapes_per_dgram = int(max_shapes_per_dgram)
        if self.max_shapes_per_dgram <= 0:
            raise ValueError("max_shapes_per_dgram must be positive")

        self._budget = budget
        self._config_bytes = sum(
            int(table.nbytes)
            for table in (
                configs.stream_names_index,
                configs.names_table,
                configs.fields_table,
            )
        )
        if budget is not None:
            budget.reserve(self._config_bytes)
        try:
            self.device_configs = configs.to_device(cp)
            self._config_ready = cp.cuda.Event(disable_timing=True)
            self._config_ready.record(cp.cuda.get_current_stream())
        except Exception:
            if budget is not None:
                budget.release(self._config_bytes)
            raise

        self._slots = [
            _GpuXtcSlotBuffers(cp=cp, budget=budget)
            for _ in range(self.n_slots)
        ]

    def parse(self, slot_id, data_gpu, desc_table, stream):
        """Walk one completed read and locate configured fields on ``stream``."""
        from .parser import GpuEventBatch

        slot_id = int(slot_id)
        if slot_id < 0 or slot_id >= self.n_slots:
            raise IndexError(slot_id)
        slot = self._slots[slot_id]
        stream.wait_event(self._config_ready)
        records, shape_counts, shape_refs = slot.prepare(
            desc_table, self.max_shapes_per_dgram, stream
        )
        batch = GpuEventBatch(
            data_gpu,
            self.device_configs,
            records,
            max_shapes_per_dgram=self.max_shapes_per_dgram,
            stream=stream,
            shape_counts_gpu=shape_counts,
            shape_refs_gpu=shape_refs,
            locator_allocator=slot.locator_rows,
            stream_ids_by_dgram=np.array(
                desc_table[:, DESC_STREAM_ID], dtype=np.uint64, copy=True
            ),
        )
        for handle in self.field_handles:
            batch.locate(handle, stream=stream)
        return batch

    def estimate_batch_bytes(self, n_dgrams):
        """Return slot metadata bytes required for ``n_dgrams`` rows."""
        n_dgrams = int(n_dgrams)
        per_dgram = (
            DGRAM_NCOLS * 8
            + 8
            + self.max_shapes_per_dgram * REF_NCOLS * 8
            + len(self.field_handles) * LOC_NCOLS * 8
        )
        return n_dgrams * per_dgram

    def memory_bytes(self):
        per_slot = [slot.memory_bytes for slot in self._slots]
        return {
            "config": self._config_bytes,
            "batch_slots": sum(per_slot),
            "per_slot": per_slot,
            "total": self._config_bytes + sum(per_slot),
        }


__all__ = [
    "GpuXtcBatchPool",
    "build_dgram_records",
    *[name for name in globals() if name.startswith(("DGRAM_", "REF_", "LOC_"))],
]
