"""Input-owned device tables for GPU-resident XTC batches."""

from dataclasses import dataclass, field

import numpy as np

from psana.gpu.gpu_allocation import (
    owned_empty, upload_owned, allocation_requirement, backing_capacity,
)
from .config import build_field_location_tables

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
    """Reusable parser buffers leased to an input window."""

    cp: object
    budget: object = None
    dgram_records: object = None
    shape_counts: object = None
    shape_refs: object = None
    locators: dict = field(default_factory=dict)
    locator_backing: object = None
    input_bases: object = None

    def batched_locator_rows(self, n_handles, n_dgrams):
        """Return [handle, capacity, column] storage; tails keep capacity strides."""
        existing = self.locator_backing
        if existing is not None and existing.shape[1] >= n_dgrams:
            return existing
        shape = (int(n_handles), int(n_dgrams), LOC_NCOLS)
        replacement = owned_empty(self.cp, shape, self.cp.uint64,
                                  self.budget, 'parser')
        self.locator_backing = replacement
        return replacement

    def _rows(self, existing, n_rows, row_shape):
        required_shape = (int(n_rows),) + tuple(row_shape)
        if existing is not None and existing.shape[0] >= n_rows:
            return existing, existing[:n_rows]

        replacement = owned_empty(self.cp, required_shape, self.cp.uint64,
                                  self.budget, 'parser')
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
        arrays = (self.dgram_records, self.shape_counts, self.shape_refs,
                  self.locator_backing, self.input_bases)
        return sum(backing_capacity(array) for array in arrays if array is not None) + sum(
            backing_capacity(array) for array in self.locators.values()
        )


class _ParsedGroupSet:
    """Shared parser arena; raw backing still retires independently per group."""

    def __init__(self, pool, index, batch):
        self.pool, self.index, self.batch = pool, index, batch
        self.windows = {}

    def release(self, group_index):
        self.windows.pop(group_index, None)
        if not self.windows:
            self.batch.retire()
            self.pool._owners[self.index] = None

    def drain(self):
        error = None
        for window in tuple(self.windows.values()):
            try:
                window.drain()
            except BaseException as exc:
                if error is None:
                    error = exc
        if error is not None:
            raise error
        return not self.windows


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
        (self.field_handles, stream_handles, handle_table) = (
            build_field_location_tables(configs, field_handles)
        )
        self.handle_indices = {h: i for i, h in enumerate(self.field_handles)}
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
        self.device_configs = configs.to_device(cp, budget=budget)
        self.stream_handles_gpu, self.handle_table_gpu = upload_owned(
            cp, (stream_handles, handle_table), budget,
        )
        self._config_bytes = sum(backing_capacity(a) for a in (
            self.device_configs.stream_names_index, self.device_configs.names,
            self.device_configs.fields, self.stream_handles_gpu, self.handle_table_gpu))
        self._config_ready = cp.cuda.Event(disable_timing=True)
        self._config_ready.record(cp.cuda.get_current_stream())

        self._owners = [None] * self.n_slots
        self._next_window_id = 0
        self._failed_inputs = []
        self._slots = [
            _GpuXtcSlotBuffers(cp=cp, budget=budget)
            for _ in range(self.n_slots)
        ]

    def parse(self, slot_id, data_gpu, desc_table, stream, *, input_bases_gpu=None):
        """Walk one completed read and locate configured fields on ``stream``."""
        from .parser import GpuEventBatch

        slot_id = int(slot_id)
        if slot_id < 0 or slot_id >= self.n_slots:
            raise IndexError(slot_id)
        if self._owners[slot_id] is not None:
            raise RuntimeError("parser storage is owned by an input window")
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
            input_bases_gpu=input_bases_gpu,
        )
        if self.field_handles:
            with stream:
                backing = slot.batched_locator_rows(
                    len(self.field_handles), len(desc_table)
                )
                batch._locate_configured(
                    self.field_handles, self.stream_handles_gpu,
                    self.handle_table_gpu, backing, self.handle_indices,
                )
        return batch

    def parse_window(self, gpu_read, stream, *, batch_id, defer_retirement=False):
        """Lease a free input parser buffer independently of execution IDs."""
        from psana.gpu.gpu_input_window import InputWindow

        try:
            index = self._owners.index(None)
        except ValueError:
            raise RuntimeError("no free GPU input parser storage") from None
        release_raw = gpu_read.retain_input()
        try:
            batch = self.parse(index, gpu_read.data_gpu, gpu_read.desc_table, stream)
            window = InputWindow(batch_id, self._next_window_id, batch,
                                 gpu_read.desc_table, release=lambda: release(index),
                                 defer_retirement=defer_retirement)
        except BaseException:
            # Submitted parser work must finish before either raw bytes or
            # partially populated tables can be reused. Preserve ownership if
            # synchronization itself fails.
            try:
                stream.synchronize()
            except BaseException:
                self._owners[index] = stream
                self._failed_inputs.append((index, stream, release_raw))
                raise
            release_raw()
            raise

        def release(index):
            release_raw()
            self._owners[index] = None

        self._owners[index] = window
        self._next_window_id += 1
        return window

    def parse_groups(self, reads, stream, *, batch_id):
        """Three parser launches for many raw groups, with separate raw leases.

        Group views share parser rows until the last group retires. Their raw
        buffers and field offsets stay independent; no raw payload is copied.
        """
        from .parser import GpuEventBatch
        from psana.gpu.gpu_input_window import InputWindow

        reads = tuple(reads)
        if not reads:
            return ()
        try:
            index = self._owners.index(None)
        except ValueError:
            raise RuntimeError('no free GPU input parser storage') from None
        releases = []
        batch = None
        windows = []

        def release_all():
            # Some child windows may exist even if a later constructor failed.
            # Drain/detach them explicitly; their shared-arena callbacks form
            # cycles and otherwise retain allocation charges until Python GC.
            for window in windows:
                window.drain()
            windows.clear()
            for release in releases:
                release()
            if batch is not None and not getattr(batch, '_retired', False):
                batch.retire()

        try:
            for read in reads:
                releases.append(read.retain_input())
            table = np.concatenate([read.desc_table for read in reads])
            n_rows = len(table)
            bases = np.empty((n_rows, 3), dtype=np.uint64)
            cursor = 0
            for read in reads:
                count = len(read.desc_table)
                bases[cursor:cursor + count, 0] = read.data_gpu.data.ptr
                bases[cursor:cursor + count, 1] = read.data_gpu.nbytes
                bases[cursor:cursor + count, 2] = np.arange(count, dtype=np.uint64)
                cursor += count
            slot = self._slots[index]
            slot.input_bases, device_bases = slot._rows(slot.input_bases, n_rows, (3,))
            device_bases.set(bases, stream=stream)
            batch = self.parse(index, reads[0].data_gpu, table, stream,
                               input_bases_gpu=device_bases)
            shared = _ParsedGroupSet(self, index, batch)
            cursor = 0
            for group_index, (read, release_raw) in enumerate(zip(reads, releases)):
                count = len(read.desc_table)
                start, stop = cursor, cursor + count
                child = GpuEventBatch.__new__(GpuEventBatch)
                child.__dict__ = batch.__dict__.copy()
                child.data_gpu = read.data_gpu
                child.n_dgrams = count
                child.dgram_records_gpu = batch.dgram_records_gpu[start:stop]
                child.shape_counts_gpu = batch.shape_counts_gpu[start:stop]
                child.shape_refs_gpu = batch.shape_refs_gpu[start:stop]
                child.stream_ids_by_dgram = batch.stream_ids_by_dgram[start:stop]
                child._input_bases_gpu = None
                child._locators = {}
                if batch._configured_backing is not None:
                    child._configured_backing = batch._configured_backing[:, start:stop]
                child._locator_allocator = (
                    lambda handle, n, a=start, b=stop:
                    slot.locator_rows(handle, n_rows)[a:b])

                def release(i=group_index, raw=release_raw):
                    raw()
                    shared.release(i)

                window = InputWindow(batch_id, self._next_window_id, child,
                                     read.desc_table, release=release,
                                     defer_retirement=True)
                self._next_window_id += 1
                shared.windows[group_index] = window
                windows.append(window)
                cursor = stop
            self._owners[index] = shared
            # Only children need a raw-array facade after submission. Shared
            # metadata must not keep a retired first group's raw alias alive.
            batch.data_gpu = None
            return tuple(windows)
        except BaseException:
            try:
                stream.synchronize()
                release_all()
            except BaseException:
                self._owners[index] = stream
                self._failed_inputs.append((index, stream, release_all))
                raise
            raise

    def close(self):
        """Drain inputs after execution/event references have been released."""
        for index, stream, release_raw in tuple(self._failed_inputs):
            stream.synchronize()
            release_raw()
            self._owners[index] = None
            self._failed_inputs.remove((index, stream, release_raw))
        for owner in tuple(self._owners):
            if owner is not None and not owner.drain():
                raise RuntimeError("GPU input still has planned or live uses")

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

    def allocation_requirements(self, n_dgrams, *, groups=False):
        """Growth requests for the free parser slot parse_window will choose."""
        try:
            index = self._owners.index(None)
        except ValueError:
            raise RuntimeError("no free GPU input parser storage") from None
        slot = self._slots[index]
        rows = [(DGRAM_NCOLS * 8, slot.dgram_records), (8, slot.shape_counts),
                (self.max_shapes_per_dgram * REF_NCOLS * 8, slot.shape_refs)]
        if self.field_handles:
            rows.append((len(self.field_handles) * LOC_NCOLS * 8, slot.locator_backing))
        if groups:
            rows.append((3 * 8, slot.input_bases))
        return [allocation_requirement(self.cp, int(n_dgrams) * size, a)
                for size, a in rows]

    def trim_free_buffers(self):
        for index, slot in enumerate(self._slots):
            if self._owners[index] is not None:
                continue
            self._slots[index] = _GpuXtcSlotBuffers(self.cp, self._budget)
            del slot

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
