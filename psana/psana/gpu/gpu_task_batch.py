"""Aligned input context for one selected execution subbatch (Stage 3a).

All arrays/pointers are borrowed and read-only. EventPool owns their lifetime.
The single-invocation producer context wraps this input boundary.
No callback, publication, user allocation, or user kernel is scheduled here.
"""
from dataclasses import dataclass

import numpy as np

from .gpu_allocation import owned_empty


# Per event/physical segment, uint64 words. SOURCE_PRESENT means a host-known
# source dgram exists, not that the device locator has accepted the field.
FIELD_RAW_PTR, FIELD_RAW_NBYTES, FIELD_LOCATOR_PTR, FIELD_ROW = range(4)
FIELD_TYPE, FIELD_RANK, FIELD_ELEMENT_SIZE, FIELD_SOURCE_PRESENT = range(4, 8)
FIELD_WORDS = 8


def select_task_events(events, envelopes):
    """Preserve GPU order and original identity, rejecting ambiguous delivery."""
    from psana import utils
    selected = {int(utils.first_timestamp(e.dgrams)) for e in envelopes}
    result, seen = [], set()
    for event in events:
        if not len(event) or event.timestamp not in selected:
            continue
        if event.timestamp in seen:
            raise ValueError('duplicate selected GPU event timestamp')
        seen.add(event.timestamp)
        result.append(event)
    return tuple(result)


def metadata_bytes(task, bindings, n_events):
    """Two identity words plus descriptor words, before allocator rounding."""
    fields = sum(len(bindings[s[0]].canonical_segment_ids)
                 for s in task.inputs if isinstance(s, tuple))
    return int(n_events) * (2 + FIELD_WORDS * fields) * 8


@dataclass(frozen=True)
class BatchFieldDescriptor:
    """Device table (N, canonical segments, FIELD_WORDS), uint64.

    Pointer columns may refer to independent input-window bases. A kernel must
    check SOURCE_PRESENT before using pointers, then check locator status,
    type/rank and byte bounds before dereferencing the raw payload. LOCATOR_PTR
    addresses the configured field's row table (LOC_* layout), not one row.
    Host segment_ids defines the second axis; no device metadata is read here.
    """
    rows: object
    segment_ids: tuple


@dataclass(frozen=True)
class _MetadataUpload:
    host: object
    device: object
    pinned_nbytes: int


class BatchInputContext:
    """Borrowed aligned inputs, valid until closed by the execution owner.

    timestamps/batch_event_indices are immutable host tuples; their *_gpu
    counterparts are contiguous uint64 device arrays, uploaded together on first
    device-metadata access during the callback. Dense-only callbacks allocate
    and upload no task metadata. Dense data/presence and
    field rows all use exactly this event ordering. Step/run/batch IDs are host
    scalars. Registered input leases must outlive every consumer of these views.
    """
    def __init__(self, events, task, prepared, bindings, constants, stream,
                 owners, *, budget=None, batch_id=0, run=None, step_generation=0):
        self.timestamps = tuple(int(e.timestamp) for e in events)
        self.batch_event_indices = tuple(int(e.batch_event_index) for e in events)
        self.batch_id, self.run, self.step_generation = batch_id, run, step_generation
        self.size = len(events)
        self._active = True
        self._task, self._prepared, self._bindings = task, prepared, bindings
        self._constants = {key: constants.get(*key) for key in task.calibconst}
        owners.extend(self._constants.values())
        self._fields = {}
        self._timestamps_gpu = self._indices_gpu = None
        self.pinned_nbytes = 0
        self._events, self._stream, self._owners, self._budget = events, stream, owners, budget
        self._metadata_open = True
        self._metadata_started = self._metadata_ready = False
        for name in task.inputs:
            if not isinstance(name, str):
                continue
            value = prepared[name]
            if value is None:
                if events:
                    raise ValueError('task dense inputs must have aligned selected rows')
                continue
            identity = tuple((e.batch_event_index, e.timestamp) for e in value.events)
            if identity != tuple(zip(self.batch_event_indices, self.timestamps)):
                raise ValueError('task dense input event identities are not aligned')

    def _ensure_metadata(self):
        if self._metadata_ready or not self.size:
            return
        if not self._metadata_open:
            raise RuntimeError('device metadata must be requested during the callback')
        if self._metadata_started:
            raise RuntimeError('previous device metadata initialization failed')
        self._metadata_started = True
        import cupy as cp
        task, bindings, events = self._task, self._bindings, self._events
        stream, owners, budget = self._stream, self._owners, self._budget
        nbytes = metadata_bytes(task, bindings, self.size)
        pinned = cp.cuda.alloc_pinned_memory(nbytes)
        # Pinned pools may return a larger block. Only upload the logical
        # extent admitted for metadata, not allocator padding.
        host = np.frombuffer(pinned, dtype=np.uint64, count=nbytes // 8)
        host.fill(0)
        target = owned_empty(cp, host.shape, np.uint64, budget, 'task-metadata')
        # Retain both before any asynchronous upload can fail. The EventPool
        # drain/quarantine path protects sources, pointers, and device storage.
        self.pinned_nbytes = memoryview(pinned).nbytes
        owners.append(_MetadataUpload(host, target, self.pinned_nbytes))
        host[:self.size] = self.timestamps
        host[self.size:2*self.size] = self.batch_event_indices
        self._timestamps_gpu = target[:self.size]
        self._indices_gpu = target[self.size:2*self.size]
        offset = 2*self.size
        waited = set()
        for selector in task.inputs:
            if isinstance(selector, str):
                continue
            detector, algorithm, field = selector
            binding = bindings[detector].field(algorithm, field)
            segments = bindings[detector].canonical_segment_ids
            columns = {segment: i for i, segment in enumerate(segments)}
            count = self.size * len(segments) * FIELD_WORDS
            rows = host[offset:offset+count].reshape(self.size, len(segments), FIELD_WORDS)
            for i, event in enumerate(events):
                for dgram, segment, handle in binding.iter_sources(event):
                    batch = dgram._storage_batch()
                    locations = batch.configured_locations()
                    table = locations.backing[locations.handle_indices[handle]]
                    if not 0 <= dgram.dgram_index < batch.n_dgrams <= table.shape[0]:
                        raise ValueError('field descriptor row is outside configured storage')
                    if id(locations.ready) not in waited:
                        locations.wait_on(stream)
                        waited.add(id(locations.ready))
                    rows[i, columns[segment]] = (
                        batch.data_gpu.data.ptr, batch.data_gpu.nbytes,
                        table.data.ptr, dgram.dgram_index, handle.type, handle.rank,
                        handle.element_size, 1)
            self._fields[selector] = BatchFieldDescriptor(
                target[offset:offset+count].reshape(rows.shape), segments)
            offset += count
        target.set(host, stream=stream)  # One bulk upload, zero metadata kernels.
        self._metadata_ready = True

    def seal(self):
        """Forbid new GPU work after callback return, before producer completion."""
        self._metadata_open = False
        self._events = self._stream = self._owners = self._budget = None

    def _require_active(self):
        if not self._active:
            raise RuntimeError('batch input context is closed')

    @property
    def timestamps_gpu(self):
        self._require_active()
        self._ensure_metadata()
        return self._timestamps_gpu

    @property
    def batch_event_indices_gpu(self):
        self._require_active()
        self._ensure_metadata()
        return self._indices_gpu

    def _input(self, name):
        self._require_active()
        if not isinstance(name, str) or name not in self._task.inputs:
            raise KeyError(f'dense input {name!r} was not declared')
        return self._prepared[name]

    def input(self, name):
        value = self._input(name)
        return None if value is None else value.data

    def present(self, name):
        value = self._input(name)
        return None if value is None else value.present

    def field(self, detector, algorithm, field):
        self._require_active()
        selector = (detector, algorithm, field)
        if selector not in self._task.inputs:
            raise KeyError(f'field {selector!r} was not declared')
        self._ensure_metadata()
        return self._fields.get(selector)  # None only for an empty selection.

    def calibconst(self, detector, key):
        self._require_active()
        selector = (detector, key)
        if selector not in self._constants:
            raise KeyError(f'calibration constant {selector!r} was not declared')
        return self._constants[selector]

    def segment_ids(self, detector):
        self._require_active()
        declared = {s.rsplit('.', 1)[0] if isinstance(s, str) else s[0]
                    for s in self._task.inputs}
        declared.update(det for det, _ in self._task.calibconst)
        if detector not in declared:
            raise KeyError(f'detector {detector!r} was not declared')
        return self._bindings[detector].canonical_segment_ids

    def close(self):
        self.seal()
        self._active = False
        self._task = self._prepared = self._bindings = self._constants = None
        self._fields.clear()
        self._timestamps_gpu = self._indices_gpu = None
