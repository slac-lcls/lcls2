"""Dense GPU input preparation and batched field gathering."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from psana.gpu.gpu_allocation import (
    owned_empty, upload_owned, allocation_requirement, backing_capacity,
)

from psana.gpu.gpu_input import GpuDetectorBinding
from psana.gpu.gpudgram.batch import (
    LOC_DIM0,
    LOC_NBYTES,
    LOC_NCOLS,
    LOC_OFFSET,
    LOC_RANK,
    LOC_STATUS,
    LOC_TYPE,
)
from psana.gpu.gpudgram.config import GpuFieldHandle
from psana.gpu.gpudgram.parser import STATUS_FOUND

def _canonical_gather_table(binding, handle_indices):
    """Compile canonical rows as [stream column, handle index, type, rank]."""
    streams = tuple(binding.sources_by_stream)
    columns = {stream: i for i, stream in enumerate(streams)}
    rows = []
    for segment in binding.canonical_segment_ids:
        handle = binding.field_handles_by_segment[segment]
        rows.append((columns[handle.stream_id], handle_indices[handle],
                     handle.type, handle.rank))
    return streams, np.asarray(rows, dtype=np.uint64).reshape(-1, 4)


# Each event/stream entry has (owner index, owner-local dgram row). Each
# distinct owner has (raw pointer, raw bytes, locator pointer, capacity, count).
# Admission conservatively allows one owner per entry. Both tables share one
# device allocation and one pinned upload source.
_GATHER_ROW_WORDS = 2
_GATHER_OWNER_WORDS = 5
_GATHER_MAP_BYTES_PER_ENTRY = (_GATHER_ROW_WORDS + _GATHER_OWNER_WORDS) * 8


@dataclass(frozen=True)
class _GatherInputs:
    rows: object
    owners: object
    locations: tuple


class _GatherMap:
    """Slot-owned device map and pinned upload source, retired with results.

    The EventPool leases every input window through execution completion,
    including partial submission failures. Standalone callers must provide the
    same lifetime guarantee. This cache retains no parsed owners after prepare.
    """

    def __init__(self):
        self.device = None
        self.host = None

    def prepare(self, events, streams, stream, budget):
        cp = _cupy()
        nitems = len(events) * len(streams)
        required = nitems * _GATHER_MAP_BYTES_PER_ENTRY
        old_bytes = 0 if self.device is None else int(self.device.nbytes)
        if required > old_bytes:
            pinned = cp.cuda.alloc_pinned_memory(required)
            host = np.frombuffer(pinned, dtype=np.uint64, count=required // 8)
            device = owned_empty(cp, required // 8, cp.uint64, budget, 'detector')
            self.host, self.device = host, device
        row_words = nitems * _GATHER_ROW_WORDS
        rows = self.host[:row_words].reshape(len(events), len(streams), _GATHER_ROW_WORDS)
        rows.fill(np.iinfo(np.uint64).max)
        owner_indices, locations = {}, []
        for i, event in enumerate(events):
            for j, stream_id in enumerate(streams):
                dgram = event.get(stream_id)
                if dgram is None:
                    continue
                batch = dgram._storage_batch()
                if not 0 <= dgram.dgram_index < batch.n_dgrams:
                    raise ValueError("invalid gather input owner or dgram row")
                index = owner_indices.get(batch)
                if index is None:
                    index = len(locations)
                    owner_indices[batch] = index
                    location = batch.configured_locations()
                    if batch.n_dgrams > location.capacity:
                        raise ValueError("gather input count exceeds locator capacity")
                    locations.append(location)
                    start = row_words + index * _GATHER_OWNER_WORDS
                    self.host[start:start + _GATHER_OWNER_WORDS] = (
                        batch.data_gpu.data.ptr, batch.data_gpu.nbytes,
                        location.backing.data.ptr, location.capacity, batch.n_dgrams,
                    )
                rows[i, j] = (index, dgram.dgram_index)
        used = row_words + len(locations) * _GATHER_OWNER_WORDS
        # Publish slot storage before uploading; failed uploads are drained by
        # EventPool before this slot or its pinned source can be reused.
        self.device[:used].set(self.host[:used], stream=stream)
        return _GatherInputs(self.device[:row_words], self.device[row_words:used],
                             tuple(locations))


class _CanonicalGatherPlan:
    """One immutable routing upload per detector/configured handle layout."""

    def __init__(self, binding, handle_indices, budget):
        cp = _cupy()
        self.streams, self.host = _canonical_gather_table(binding, handle_indices)
        self.handle_indices = handle_indices
        self.table, = upload_owned(cp, (self.host,), budget)
        self.ready = cp.cuda.Event(disable_timing=True)
        producer = cp.cuda.get_current_stream()
        self.ready.record(producer)
        self.ordered_streams = {producer.ptr: producer}

    def gather(self, inputs, target, present, pixels, stream, shape=None):
        for locations in inputs.locations:
            if locations.handle_indices is not self.handle_indices:
                raise ValueError("gather plan does not match the configured locator layout")
        if stream.ptr not in self.ordered_streams:
            stream.wait_event(self.ready)
            self.ordered_streams[stream.ptr] = stream
        waited = set()
        for locations in inputs.locations:
            if id(locations.ready) not in waited:
                locations.wait_on(stream)
                waited.add(id(locations.ready))
        nsegments = int(self.table.shape[0])
        nrows = int(present.size)
        tiles = (pixels + 255) // 256
        _batched_gather_kernel(target.dtype)(
            (tiles * nrows,), (256,),
            (inputs.owners, np.uint64(len(inputs.locations)), self.table, inputs.rows,
             np.uint64(len(self.streams)), np.uint64(nsegments),
             np.uint64(pixels), np.uint64(tiles),
             np.uint64(shape[0] if shape else 0), np.uint64(shape[1] if shape else 0),
             target, present),
            stream=stream,
        )


@dataclass(frozen=True)
class PreparedInputBatch:
    """Borrowed dense inputs in event order; the execution lease owns storage."""

    events: tuple
    data: object       # (events, segments, rows, columns), original input dtype
    present: object    # (events, segments), uint8; parser/gather validity
    source_present: tuple = ()  # Host descriptor presence, not device field validity.


class DenseInputPreparer:
    """Batched dense field preparation without calibration or output storage.

    Shape comes from an explicit supported detector adapter, never constants.
    The caller retains input windows and retires execution consumers before
    reusing/trimming slots. Preparation queues one
    gather per nonempty subbatch; it performs no locator metadata D2H.
    """

    def __init__(self, det_shape, binding, *, dtype=np.uint16, n_slots=2,
                 budget=None):
        if not isinstance(binding, GpuDetectorBinding):
            raise TypeError("binding must be a GpuDetectorBinding")
        if (len(det_shape) != 3 or
                any(int(n) != n or n <= 0 for n in det_shape)):
            raise ValueError("det_shape must contain three positive dimensions")
        self.det_shape = tuple(int(n) for n in det_shape)
        self.binding = binding
        self._canonical_segment_ids = binding.canonical_segment_ids
        self._n_segments, self._nrows, self._ncols = self.det_shape
        if len(self._canonical_segment_ids) != self._n_segments:
            raise ValueError("canonical_segment_ids must contain one entry per detector segment")
        self._n_pix_seg = self._nrows * self._ncols
        self._dtype = np.dtype(dtype)
        if self._dtype not in (np.dtype(np.uint16), np.dtype(np.float32)):
            raise TypeError("dense preparation supports uint16 and float32")
        self._pixel_bytes = self._dtype.itemsize
        self._field_handles_by_segment = binding.field_handles_by_segment
        self._sources_by_stream = binding.sources_by_stream
        if not self._field_handles_by_segment:
            raise ValueError("dense preparation requires a field for every segment")
        for segment, handle in self._field_handles_by_segment.items():
            if handle.rank <= 0 or handle.element_size != self._pixel_bytes:
                raise ValueError(f"segment {segment}: incompatible dense field layout")
            if (handle.rank not in (2, 3) or
                                   handle.type != (1 if self._dtype == np.uint16 else 8)):
                raise ValueError(f"segment {segment}: unsupported dense field type/rank")
        if int(n_slots) != n_slots or n_slots <= 0:
            raise ValueError("n_slots must be a positive integer")
        self._n_slots = int(n_slots)
        self._budget = budget
        self._raw_slot_bufs = [None] * self._n_slots
        self._present_slot_bufs = [None] * self._n_slots
        self._gather_maps = [_GatherMap() for _ in range(self._n_slots)]
        self._gather_plan = None

    @classmethod
    def jungfrau_raw(cls, configs, binding, **kwargs):
        """Bind Jungfrau raw panels using Configure membership/type/rank.

        Jungfrau's supported panel is (1, 512, 1024) or (512, 1024), as
        specified by drp/Jungfrau.cc and SegGeometryJungfrauV2. Names provides
        type/rank, not runtime dimensions; the gather validates those on GPU.
        Rows follow binding.canonical_segment_ids. Calibration arrays retain
        their full physical-segment axis: select [:, segment_ids, ...] in user
        code rather than assuming dense row index equals physical segment ID.
        """
        for segment, handle in binding.field_handles_by_segment.items():
            names = configs.names_for_id(handle.stream_id, handle.names_id)
            field = names.fields[handle.field_index]
            if (names.det_name != binding.det_name or names.segment != segment or
                    names.det_type != "jungfrau" or names.alg_name != "raw" or
                    field.name != "raw" or
                    configs.resolve(binding.det_name, segment, "raw", "raw",
                                    stream_id=handle.stream_id) != handle):
                raise ValueError("unsupported Jungfrau raw Configure binding")
        return cls((len(binding.canonical_segment_ids), 512, 1024), binding,
                   dtype=np.uint16, **kwargs)

    @property
    def canonical_segment_ids(self):
        return self._canonical_segment_ids

    def configure_gather(self, handle_indices):
        """Upload fixed canonical routing once, after parser setup."""
        if self._gather_plan is not None:
            if self._gather_plan.handle_indices is not handle_indices:
                raise ValueError("cannot replace a live canonical gather plan")
            return
        self._gather_plan = _CanonicalGatherPlan(
            self.binding, handle_indices, self._budget,
        )

    def prepare_batch(self, gpu_events, stream=None, slot_id=None, *, aligned=False):
        """Queue one gather and return borrowed event-major data/presence.

        By default events without source dgrams are omitted. Task preparation
        uses aligned=True to preserve every selected row across detectors.
        Missing/rejected fields have zero data and presence.
        """
        events = tuple(event for event in gpu_events
                       if aligned or self.binding.has_sources(event))
        if not events:
            return None
        if slot_id is None:
            raise ValueError("dense preparation requires an EventPool slot_id")
        cp = _cupy()
        slot = int(slot_id) % self._n_slots
        shape = (len(events) * self._n_segments, self._nrows, self._ncols)
        data = self._slot_buffer(self._raw_slot_bufs,
                                 slot, shape, self._dtype, "input")
        present = self._slot_buffer(self._present_slot_bufs, slot,
                                   (len(events), self._n_segments), np.uint8,
                                   "field-presence")
        sctx = stream if stream is not None else cp.cuda.Stream.null
        if self._gather_plan is None:
            source = next((event[sid] for event in events
                           for sid in self._sources_by_stream if sid in event), None)
            if source is None:
                raise RuntimeError('aligned absent input requires configure_gather at setup')
            self.configure_gather(source._storage_batch().configured_locations().handle_indices)
        with sctx:
            inputs = self._gather_maps[slot].prepare(events, self._gather_plan.streams,
                                                     sctx, self._budget)
            self._gather_plan.gather(inputs, data, present, self._n_pix_seg, sctx,
                                     shape=self.det_shape[-2:])
        return PreparedInputBatch(events, data.reshape((len(events),) + self.det_shape), present,
                                  tuple(self.binding.has_sources(event) for event in events)
                                  if aligned else ())

    def memory_bytes(self):
        raw = (sum(backing_capacity(b) for b in self._raw_slot_bufs if b is not None)
               + sum(backing_capacity(b) for b in self._present_slot_bufs if b is not None)
               + sum(backing_capacity(m.device) for m in self._gather_maps if m.device is not None))
        routing = backing_capacity(self._gather_plan.table) if self._gather_plan else 0
        return dict(raw_slots=raw, routing=routing, total=raw + routing)

    def pinned_bytes(self) -> int:
        """Host row-map upload buffers, reported separately from device bytes."""
        return sum(int(m.host.nbytes) for m in self._gather_maps
                   if m.host is not None)

    def estimate_subbatch_bytes(self, n_events):
        return max(0, int(n_events)) * (
            int(np.prod(self.det_shape)) * self._pixel_bytes + self._n_segments
            + len(self._sources_by_stream) * _GATHER_MAP_BYTES_PER_ENTRY)

    def allocation_requirements(self, n_events, slot):
        items = [(int(n_events) * int(np.prod(self.det_shape)) * self._pixel_bytes,
                  self._raw_slot_bufs[slot]),
                 (int(n_events) * self._n_segments, self._present_slot_bufs[slot]),
                 (int(n_events) * len(self._sources_by_stream) * _GATHER_MAP_BYTES_PER_ENTRY,
                  self._gather_maps[slot].device)]
        return [allocation_requirement(_cupy(), need, a) for need, a in items]

    def trim_slot_buffers(self):
        """Caller must first retire every execution/input consumer lease."""
        self._raw_slot_bufs = [None] * self._n_slots
        self._present_slot_bufs = [None] * self._n_slots
        self._gather_maps = [_GatherMap() for _ in range(self._n_slots)]

    def _slot_buffer(self, buffers, slot, shape, dtype, label):
        """Return a reusable slot view, growing its backing array only."""
        cp = _cupy()
        nitems = int(np.prod(shape))
        needed = nitems * np.dtype(dtype).itemsize
        buf = buffers[slot]
        old_size = int(buf.nbytes) if buf is not None else 0
        if old_size < needed:
            new_buf = owned_empty(cp, nitems, dtype, self._budget, 'detector')
            buffers[slot] = new_buf
            buf = new_buf
            if __import__('os').environ.get('PSANA_GPU_MEM_DEBUG'):
                free_b, _ = cp.cuda.Device().mem_info
                print(
                    f'[GPU-MEM] {label} slot grow: '
                    f'need={needed/1e9:.1f}GB free={free_b/1e9:.1f}GB',
                    flush=True,
                )
        return buf[:nitems].reshape(shape)



@lru_cache(maxsize=2)
def _batched_gather_kernel(dtype):
    cp = _cupy()
    ctype = "unsigned short" if dtype == cp.dtype(cp.uint16) else "float"
    if dtype not in (cp.dtype(cp.uint16), cp.dtype(cp.float32)):
        raise TypeError("unsupported canonical gather dtype")
    name = "gather_canonical_u16" if ctype == "unsigned short" else "gather_canonical_f32"
    return cp.RawKernel(f"""
extern "C" __global__ void {name}(
    const unsigned long long* owners, unsigned long long n_owners,
    const unsigned long long* plan,
    const unsigned long long* rows, unsigned long long n_streams,
    unsigned long long n_segments, unsigned long long pixels,
    unsigned long long tiles, unsigned long long expected_rows,
    unsigned long long expected_cols, {ctype}* out, unsigned char* present)
{{
    const unsigned long long row = (unsigned long long)blockIdx.x / tiles;
    const unsigned long long pixel = ((unsigned long long)blockIdx.x % tiles)
                                     * blockDim.x + threadIdx.x;
    const unsigned long long* source = plan + (row % n_segments) * 4;
    const unsigned long long entry = ((row / n_segments) * n_streams + source[0]) * {_GATHER_ROW_WORDS};
    const unsigned long long owner_index = rows[entry];
    const unsigned long long dgram = rows[entry + 1];
    bool valid = owner_index < n_owners;
    const unsigned char* data = nullptr;
    const unsigned long long* locators = nullptr;
    unsigned long long data_bytes = 0, capacity = 0, offset = 0;
    if (valid) {{
        const unsigned long long* owner = owners + owner_index * {_GATHER_OWNER_WORDS};
        data = reinterpret_cast<const unsigned char*>(owner[0]);
        data_bytes = owner[1];
        locators = reinterpret_cast<const unsigned long long*>(owner[2]);
        capacity = owner[3];
        valid = dgram < owner[4] && dgram < capacity;
    }}
    if (valid) {{
        const unsigned long long* loc = locators +
            (source[1] * capacity + dgram) * {LOC_NCOLS};
        offset = loc[{LOC_OFFSET}];
        const unsigned long long nbytes = loc[{LOC_NBYTES}];
        valid = loc[{LOC_STATUS}] == {STATUS_FOUND} &&
                loc[{LOC_TYPE}] == source[2] && loc[{LOC_RANK}] == source[3] &&
                nbytes == pixels * sizeof({ctype}) &&
                offset <= data_bytes && nbytes <= data_bytes - offset;
        if (valid && expected_rows) {{
            const bool rank3 = source[3] == 3;
            const unsigned long long dim = {LOC_DIM0} + (rank3 ? 1 : 0);
            valid = (source[3] == 2 || (rank3 && loc[{LOC_DIM0}] == 1)) &&
                    loc[dim] == expected_rows && loc[dim + 1] == expected_cols;
        }}
    }}
    if (pixel == 0) present[row] = valid ? 1 : 0;
    if (pixel < pixels) {{
        out[row * pixels + pixel] = valid ?
            reinterpret_cast<const {ctype}*>(data + offset)[pixel] : ({ctype})0;
    }}
}}
""", name, options=("--std=c++17",))


@lru_cache(maxsize=1)
def _cupy():
    import cupy as cp

    return cp
