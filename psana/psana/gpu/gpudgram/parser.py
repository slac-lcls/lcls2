"""Device-resident XTC walk and field-location primitives.

Stage 1 deliberately exposes numeric device tables rather than Python dgram
views.  A run owner uploads :class:`DeviceConfigTables` once.  A batch owner
provides raw bytes and dgram records already resident on the GPU; all XTC walk
results stay there for direct consumption by later CUDA kernels.
"""

from functools import lru_cache

import numpy as np

from .batch import (
    DGRAM_DAMAGE,
    DGRAM_ENV,
    DGRAM_NCOLS,
    DGRAM_OFFSET,
    DGRAM_SERVICE,
    DGRAM_SIZE,
    DGRAM_STATUS,
    DGRAM_STREAM_ID,
    DGRAM_TIMESTAMP,
    DGRAM_TYPE,
    LOC_CONFIG_FIELD_INDEX,
    LOC_DIM0,
    LOC_MAX_RANK,
    LOC_NBYTES,
    LOC_NCOLS,
    LOC_OFFSET,
    LOC_RANK,
    LOC_STATUS,
    LOC_TYPE,
    REF_CONFIG_NAMES_INDEX,
    REF_DAMAGE,
    REF_DGRAM_INDEX,
    REF_EXTENT,
    REF_NCOLS,
    REF_OFFSET,
)
from .config import (
    FIELD_ELEMENT_SIZE,
    FIELD_NCOLS,
    FIELD_RANK,
    FIELD_SHAPE_INDEX,
    FIELD_TYPE,
    HANDLE_NAMES_INDEX,
    HANDLE_FIELD_INDEX,
    HANDLE_OUTPUT_INDEX,
    HANDLE_NCOLS,
    NAMES_FIRST_FIELD,
    NAMES_ID,
    NAMES_NCOLS,
    NAMES_N_FIELDS,
    GpuFieldHandle,
)

STATUS_OK = 0
STATUS_FOUND = 1
STATUS_NOT_PRESENT = 2
STATUS_BAD_DGRAM = 3
STATUS_BAD_XTC = 4
STATUS_CORRUPTED = 5
STATUS_CAPACITY = 6
STATUS_MISSING_SHAPES = 7
STATUS_MISSING_DATA = 8
STATUS_BAD_SHAPE = 9
STATUS_DATA_OVERFLOW = 10
STATUS_DUPLICATE = 11
STATUS_BAD_STREAM = 12
STATUS_UNKNOWN_NAMES = 13
STATUS_BAD_CONFIG = 14
STATUS_CLAIMED = 15

STATUS_NAMES = {
    STATUS_OK: "ok",
    STATUS_FOUND: "found",
    STATUS_NOT_PRESENT: "not_present",
    STATUS_BAD_DGRAM: "bad_dgram",
    STATUS_BAD_XTC: "bad_xtc",
    STATUS_CORRUPTED: "corrupted",
    STATUS_CAPACITY: "capacity",
    STATUS_MISSING_SHAPES: "missing_shapes",
    STATUS_MISSING_DATA: "missing_data",
    STATUS_BAD_SHAPE: "bad_shape",
    STATUS_DATA_OVERFLOW: "data_overflow",
    STATUS_DUPLICATE: "duplicate",
    STATUS_BAD_STREAM: "bad_stream",
    STATUS_UNKNOWN_NAMES: "unknown_names",
    STATUS_BAD_CONFIG: "bad_config",
    STATUS_CLAIMED: "claimed",
}


class DeviceFieldLocators:
    """Locator rows with an optional event access boundary."""

    def __init__(self, handle, rows_gpu, ready, lease=None):
        self.handle, self._rows_gpu, self.ready = handle, rows_gpu, ready
        self._lease = lease
        if lease is not None:
            lease.on_retire(self.retire)

    def retire(self):
        self._rows_gpu = None
        self.ready = None

    @property
    def rows_gpu(self):
        if self._lease is not None:
            self._lease.require_active()
        if self._rows_gpu is None:
            raise RuntimeError("GPU locator storage is released")
        return self._rows_gpu

    @property
    def n_dgrams(self):
        return int(self.rows_gpu.shape[0])

    def wait_on(self, stream):
        rows = self.rows_gpu
        stream.wait_event(self.ready)
        return rows


def _batch_storage(name):
    def get(self):
        if getattr(self, '_retired', False):
            raise RuntimeError("GPU batch storage is released")
        return getattr(self, '_' + name)
    def set_(self, value):
        setattr(self, '_' + name, value)
    return property(get, set_)


class ConfiguredFieldLocations:
    """Internal gather descriptor following its parsed owner's lifetime."""

    def __init__(self, owner):
        self.owner = owner

    def _active_owner(self):
        if getattr(self.owner, '_retired', False):
            raise RuntimeError("GPU batch storage is released")
        return self.owner

    @property
    def backing(self):
        return self._active_owner()._configured_backing

    @property
    def handle_indices(self):
        return self._active_owner()._configured_indices

    @property
    def ready(self):
        return self._active_owner()._configured_ready

    @property
    def capacity(self):
        # Group views slice the dgram axis while preserving the shared arena's
        # handle stride. Gather kernels need that stride, not the slice length.
        return int(self.backing.strides[0] // (LOC_NCOLS * 8))

    def wait_on(self, stream):
        # Same-stream submission is already ordered. Keep the producer stream
        # alive via owner so its identity cannot be reused before consumption.
        if stream.ptr != self._active_owner().stream.ptr:
            stream.wait_event(self.ready)


class GpuEventBatch:
    """Own device-side XTC parse state for one collection of stream dgrams.

    Parameters
    ----------
    data_gpu : cupy.ndarray, uint8[nbytes]
        Batch bytes produced by the read path.
    device_configs : DeviceConfigTables
        Run-scoped Configure tables already uploaded to the GPU.
    dgram_records_gpu : cupy.ndarray, uint64[n_dgrams, DGRAM_NCOLS]
        Device records.  Event index, stream id, byte offset, and byte size are
        inputs; the walker fills timestamp, env, service, damage, type, and
        status in place.
    stream_ids_by_dgram : numpy.ndarray, optional
        Metadata-only CPU view used to map an event's stream dgrams to dense
        locator rows. It is derived from read descriptors, not GPU parsing.

    Notes
    -----
    This object never copies parser metadata to the CPU.  Its buffers must
    remain owned until all consumers complete. The production path uses an
    InputWindow to protect raw bytes and parser tables independently of
    execution slots; standalone callers must coordinate their own reuse.
    """

    data_gpu = _batch_storage('data_gpu')
    dgram_records_gpu = _batch_storage('dgram_records_gpu')
    shape_counts_gpu = _batch_storage('shape_counts_gpu')
    shape_refs_gpu = _batch_storage('shape_refs_gpu')

    def __init__(
        self,
        data_gpu,
        device_configs,
        dgram_records_gpu,
        *,
        max_shapes_per_dgram=64,
        threads=128,
        stream=None,
        shape_counts_gpu=None,
        shape_refs_gpu=None,
        locator_allocator=None,
        stream_ids_by_dgram=None,
        input_bases_gpu=None,
    ):
        cp = _cupy()
        _require_device_array(data_gpu, cp.uint8, 1, "data_gpu")
        _require_device_array(
            dgram_records_gpu, cp.uint64, 2, "dgram_records_gpu"
        )
        if dgram_records_gpu.shape[1] != DGRAM_NCOLS:
            raise ValueError(
                "dgram_records_gpu must have shape "
                f"(n, {DGRAM_NCOLS}), got {dgram_records_gpu.shape}"
            )
        _validate_device_configs(device_configs, cp)

        self.data_gpu = data_gpu
        self._input_bases_gpu = input_bases_gpu
        self.device_configs = device_configs
        self.dgram_records_gpu = dgram_records_gpu
        self.n_dgrams = int(dgram_records_gpu.shape[0])
        if input_bases_gpu is not None:
            _require_device_array(input_bases_gpu, cp.uint64, 2, 'input_bases_gpu')
            if input_bases_gpu.shape != (self.n_dgrams, 3):
                raise ValueError('input base table must have one pointer/size/index row per dgram')
        if stream_ids_by_dgram is None:
            self.stream_ids_by_dgram = None
        else:
            stream_ids_by_dgram = np.asarray(stream_ids_by_dgram)
            if stream_ids_by_dgram.shape != (self.n_dgrams,):
                raise ValueError(
                    "stream_ids_by_dgram shape does not match n_dgrams"
                )
            self.stream_ids_by_dgram = stream_ids_by_dgram
        self.max_shapes_per_dgram = int(max_shapes_per_dgram)
        self.threads = int(threads)
        if self.max_shapes_per_dgram <= 0:
            raise ValueError("max_shapes_per_dgram must be positive")
        if self.threads <= 0:
            raise ValueError("threads must be positive")

        self.stream = stream if stream is not None else cp.cuda.get_current_stream()
        self._locator_allocator = locator_allocator
        with self.stream:
            if shape_counts_gpu is None:
                shape_counts_gpu = cp.empty(self.n_dgrams, dtype=cp.uint64)
            else:
                _require_device_array(
                    shape_counts_gpu, cp.uint64, 1, "shape_counts_gpu"
                )
                if shape_counts_gpu.shape != (self.n_dgrams,):
                    raise ValueError("shape_counts_gpu shape does not match n_dgrams")
            if shape_refs_gpu is None:
                shape_refs_gpu = cp.empty(
                    (self.n_dgrams, self.max_shapes_per_dgram, REF_NCOLS),
                    dtype=cp.uint64,
                )
            else:
                _require_device_array(
                    shape_refs_gpu, cp.uint64, 3, "shape_refs_gpu"
                )
                expected = (
                    self.n_dgrams,
                    self.max_shapes_per_dgram,
                    REF_NCOLS,
                )
                if shape_refs_gpu.shape != expected:
                    raise ValueError(
                        f"shape_refs_gpu must have shape {expected}, "
                        f"got {shape_refs_gpu.shape}"
                    )
            self.shape_counts_gpu = shape_counts_gpu
            self.shape_refs_gpu = shape_refs_gpu
            if self.n_dgrams:
                blocks = (self.n_dgrams + self.threads - 1) // self.threads
                _walk_kernel()(
                    (blocks,),
                    (self.threads,),
                    (
                        self.data_gpu,
                        np.uint64(self.data_gpu.nbytes),
                        self.dgram_records_gpu,
                        np.uint64(self.n_dgrams),
                        self.device_configs.stream_names_index,
                        self.device_configs.names,
                        np.uint64(self.device_configs.n_streams),
                        self.shape_refs_gpu,
                        self.shape_counts_gpu,
                        np.uint64(self.max_shapes_per_dgram),
                        input_bases_gpu if input_bases_gpu is not None else np.uint64(0),
                    ),
                    stream=self.stream,
                )
            self.walk_done = cp.cuda.Event()
            self.walk_done.record(self.stream)
        self._locators = {}
        self._configured_backing = None

    def _locate_configured(self, handles, stream_handles, handle_table, backing,
                          handle_indices):
        """Decode configured fields in two launches on the parser stream.

        The slot owns the backing allocation; retain it and the shared ready
        event independently of the on-demand per-handle views. Unconfigured
        requests retain their separate allocator and completion events.
        """
        cp = _cupy()
        capacity = int(backing.shape[1])
        # Retain scheduling tables through completion even if the pool is dropped.
        self._location_tables = (stream_handles, handle_table)
        if self.n_dgrams and handles:
            n_rows = len(handles) * self.n_dgrams
            _init_locators_kernel()(
                ((n_rows + self.threads - 1) // self.threads,),
                (self.threads,),
                (backing, self.dgram_records_gpu, np.uint64(self.n_dgrams),
                 np.uint64(capacity), np.uint64(len(handles))),
                stream=self.stream,
            )
            _locate_fields_kernel()(
                (self.n_dgrams,), (self.threads,),
                (self.data_gpu, self.shape_refs_gpu, self.shape_counts_gpu,
                 self.dgram_records_gpu, np.uint64(self.max_shapes_per_dgram),
                 self.device_configs.names, self.device_configs.fields,
                 np.uint64(self.device_configs.n_names),
                 np.uint64(self.device_configs.n_streams), stream_handles,
                 handle_table, np.uint64(capacity), backing,
                 self._input_bases_gpu if self._input_bases_gpu is not None else np.uint64(0)),
                stream=self.stream,
            )
        ready = cp.cuda.Event(disable_timing=True)
        ready.record(self.stream)
        self._configured_backing = backing
        self._configured_indices = handle_indices
        self._configured_ready = ready

    def configured_locations(self):
        """Return input-local combined storage without reading device metadata."""
        if getattr(self, '_retired', False):
            raise RuntimeError("GPU batch storage is released")
        if self._configured_backing is None:
            raise ValueError("canonical gathering requires configured field locations")
        return ConfiguredFieldLocations(self)

    def retire(self):
        """Detach completed window storage, including bound slot allocators."""
        for locator in self._locators.values():
            locator.retire()
        self._locators.clear()
        self._locator_allocator = None
        self._configured_backing = self._configured_ready = None
        self._configured_indices = self._location_tables = None
        self._input_bases_gpu = None
        self.data_gpu = self.dgram_records_gpu = None
        self.shape_counts_gpu = self.shape_refs_gpu = None
        self.device_configs = None
        self.walk_done = None
        self._retired = True

    def locate(self, handle, *, stream=None):
        """Return a cached view, decoding only unconfigured handles on demand.

        Configured fields were decoded on the parser stream. Their first access
        creates only a view, sharing the configured-ready event; consumers must
        still wait on that event before using the rows on another stream.
        """
        if self.data_gpu is None:
            raise RuntimeError("GPU batch storage is released")
        if not isinstance(handle, GpuFieldHandle):
            raise TypeError("handle must be a GpuFieldHandle")
        if not 0 <= handle.stream_id < self.device_configs.n_streams:
            raise ValueError("field handle has an invalid stream id")
        if not 0 <= handle.config_names_index < self.device_configs.n_names:
            raise ValueError("field handle has an invalid Configure Names index")
        if not 0 <= handle.config_field_index < self.device_configs.n_fields:
            raise ValueError("field handle has an invalid Configure field index")

        cached = self._locators.get(handle)
        if cached is not None:
            return cached

        if self._configured_backing is not None:
            index = self._configured_indices.get(handle)
            if index is not None:
                result = DeviceFieldLocators(
                    handle, self._configured_backing[index, :self.n_dgrams],
                    self._configured_ready,
                )
                self._locators[handle] = result
                return result

        cp = _cupy()
        launch_stream = stream if stream is not None else self.stream
        if launch_stream is not self.stream:
            launch_stream.wait_event(self.walk_done)
        with launch_stream:
            if self._locator_allocator is None:
                rows_gpu = cp.empty(
                    (self.n_dgrams, LOC_NCOLS), dtype=cp.uint64
                )
            else:
                rows_gpu = self._locator_allocator(handle, self.n_dgrams)
                _require_device_array(rows_gpu, cp.uint64, 2, "locator rows")
                if rows_gpu.shape != (self.n_dgrams, LOC_NCOLS):
                    raise ValueError("locator allocator returned the wrong shape")
            rows_gpu.fill(0)
            if self.n_dgrams:
                rows_gpu[:, LOC_STATUS] = STATUS_NOT_PRESENT
                n_work = self.n_dgrams * self.max_shapes_per_dgram
                blocks = (n_work + self.threads - 1) // self.threads
                _locate_kernel()(
                    (blocks,),
                    (self.threads,),
                    (
                        self.data_gpu,
                        self.shape_refs_gpu,
                        self.shape_counts_gpu,
                        self.dgram_records_gpu,
                        np.uint64(self.n_dgrams),
                        np.uint64(self.max_shapes_per_dgram),
                        self.device_configs.names,
                        self.device_configs.fields,
                        np.uint64(self.device_configs.n_names),
                        np.uint64(handle.config_names_index),
                        np.uint64(handle.config_field_index),
                        rows_gpu,
                    ),
                    stream=launch_stream,
                )
            ready = cp.cuda.Event()
            ready.record(launch_stream)

        result = DeviceFieldLocators(handle=handle, rows_gpu=rows_gpu, ready=ready)
        self._locators[handle] = result
        return result


def _require_device_array(array, dtype, ndim, name):
    if array.dtype != dtype or array.ndim != ndim:
        raise TypeError(
            f"{name} must be a {ndim}-dimensional CuPy {dtype} array"
        )
    if not array.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")


def _validate_device_configs(configs, cp):
    _require_device_array(
        configs.stream_names_index, cp.uint64, 1, "stream_names_index"
    )
    _require_device_array(configs.names, cp.uint64, 2, "names")
    _require_device_array(configs.fields, cp.uint64, 2, "fields")
    if configs.stream_names_index.shape != (configs.n_streams + 1,):
        raise ValueError("stream_names_index shape does not match n_streams")
    if configs.names.shape != (configs.n_names, NAMES_NCOLS):
        raise ValueError("names shape does not match n_names")
    if configs.fields.shape != (configs.n_fields, FIELD_NCOLS):
        raise ValueError("fields shape does not match n_fields")


@lru_cache(maxsize=1)
def _cupy():
    import cupy as cp

    return cp


@lru_cache(maxsize=1)
def _walk_kernel():
    return _cupy().RawKernel(_kernel_source(), "walk_xtc")


@lru_cache(maxsize=1)
def _locate_kernel():
    return _cupy().RawKernel(_kernel_source(), "locate_field")


@lru_cache(maxsize=1)
def _init_locators_kernel():
    return _cupy().RawKernel(_kernel_source(), "init_locators")


@lru_cache(maxsize=1)
def _locate_fields_kernel():
    return _cupy().RawKernel(_kernel_source(), "locate_fields")


@lru_cache(maxsize=1)
def _kernel_source():
    return f"""
namespace {{

static constexpr unsigned long long XTC_HEADER = 12;
static constexpr unsigned long long DGRAM_HEADER = 24;
static constexpr unsigned int TYPE_PARENT = 0;
static constexpr unsigned int TYPE_SHAPES_DATA = 1;
static constexpr unsigned int TYPE_SHAPES = 2;
static constexpr unsigned int TYPE_DATA = 3;
static constexpr unsigned int TYPE_MASK = 0x0fff;
static constexpr unsigned int DAMAGE_CORRUPTED = 1u << 3;
static constexpr unsigned int MAX_DEPTH = 16;
static constexpr unsigned long long SHAPE_BYTES = {LOC_MAX_RANK * 4};

__device__ __forceinline__ unsigned short load_u16(const unsigned char* p)
{{
    return static_cast<unsigned short>(p[0]) |
           (static_cast<unsigned short>(p[1]) << 8);
}}

__device__ __forceinline__ unsigned int load_u32(const unsigned char* p)
{{
    return static_cast<unsigned int>(p[0]) |
           (static_cast<unsigned int>(p[1]) << 8) |
           (static_cast<unsigned int>(p[2]) << 16) |
           (static_cast<unsigned int>(p[3]) << 24);
}}

__device__ __forceinline__ unsigned long long load_u64(const unsigned char* p)
{{
    return static_cast<unsigned long long>(load_u32(p)) |
           (static_cast<unsigned long long>(load_u32(p + 4)) << 32);
}}

__device__ __forceinline__ unsigned long long find_names(
    unsigned long long stream_id,
    unsigned int names_id,
    const unsigned long long* stream_names_index,
    const unsigned long long* names)
{{
    unsigned long long lo = stream_names_index[stream_id];
    unsigned long long hi = stream_names_index[stream_id + 1];
    while (lo < hi) {{
        const unsigned long long mid = lo + (hi - lo) / 2;
        const unsigned long long value =
            names[mid * {NAMES_NCOLS} + {NAMES_ID}];
        if (value < names_id) lo = mid + 1;
        else hi = mid;
    }}
    if (lo == stream_names_index[stream_id + 1] ||
        names[lo * {NAMES_NCOLS} + {NAMES_ID}] != names_id) {{
        return ~0ull;
    }}
    return lo;
}}

__device__ __forceinline__ void finish_locator(
    unsigned long long* locator,
    unsigned long long status)
{{
    atomicCAS(locator + {LOC_STATUS},
              static_cast<unsigned long long>({STATUS_CLAIMED}),
              status);
}}

}} // namespace

extern "C" __global__
void walk_xtc(const unsigned char* data,
              unsigned long long data_nbytes,
              unsigned long long* dgrams,
              unsigned long long n_dgrams,
              const unsigned long long* stream_names_index,
              const unsigned long long* names,
              unsigned long long n_streams,
              unsigned long long* refs,
              unsigned long long* counts,
              unsigned long long ref_capacity,
              const unsigned long long* input_bases)
{{
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= n_dgrams) return;
    if (input_bases) {{
        data = reinterpret_cast<const unsigned char*>(input_bases[index * 3]);
        data_nbytes = input_bases[index * 3 + 1];
    }}

    unsigned long long* dgram = dgrams + index * {DGRAM_NCOLS};
    dgram[{DGRAM_STATUS}] = {STATUS_BAD_DGRAM};
    counts[index] = 0;
    const unsigned long long dgram_offset = dgram[{DGRAM_OFFSET}];
    const unsigned long long dgram_size = dgram[{DGRAM_SIZE}];
    if (dgram_offset > data_nbytes ||
        dgram_size < DGRAM_HEADER ||
        dgram_size > data_nbytes - dgram_offset) return;

    const unsigned char* base = data + dgram_offset;
    const unsigned int root_extent = load_u32(base + 20);
    if (root_extent < XTC_HEADER || 12ull + root_extent != dgram_size) return;

    const unsigned int env = load_u32(base + 8);
    dgram[{DGRAM_TIMESTAMP}] = load_u64(base);
    dgram[{DGRAM_ENV}] = env;
    dgram[{DGRAM_SERVICE}] = (env >> 24) & 0x0f;
    dgram[{DGRAM_DAMAGE}] = load_u16(base + 16);
    dgram[{DGRAM_TYPE}] = load_u16(base + 18) & TYPE_MASK;

    const unsigned long long stream_id = dgram[{DGRAM_STREAM_ID}];
    if (stream_id >= n_streams) {{
        dgram[{DGRAM_STATUS}] = {STATUS_BAD_STREAM};
        return;
    }}
    if (load_u16(base + 16) & DAMAGE_CORRUPTED) {{
        dgram[{DGRAM_STATUS}] = {STATUS_CORRUPTED};
        return;
    }}
    if ((load_u16(base + 18) & TYPE_MASK) != TYPE_PARENT) {{
        dgram[{DGRAM_STATUS}] = {STATUS_BAD_XTC};
        return;
    }}

    unsigned long long cursor[MAX_DEPTH];
    unsigned long long end[MAX_DEPTH];
    int depth = 0;
    cursor[0] = DGRAM_HEADER;
    end[0] = dgram_size;
    unsigned long long count = 0;
    unsigned long long status = {STATUS_OK};

    while (depth >= 0) {{
        if (cursor[depth] == end[depth]) {{
            --depth;
            continue;
        }}
        if (cursor[depth] > end[depth] ||
            end[depth] - cursor[depth] < XTC_HEADER) {{
            status = {STATUS_BAD_XTC};
            break;
        }}

        const unsigned long long node_offset = cursor[depth];
        const unsigned char* node = base + node_offset;
        const unsigned int extent = load_u32(node + 8);
        if (extent < XTC_HEADER || extent > end[depth] - node_offset) {{
            status = {STATUS_BAD_XTC};
            break;
        }}
        const unsigned long long node_end = node_offset + extent;
        cursor[depth] = node_end;
        const unsigned int type = load_u16(node + 6) & TYPE_MASK;

        if (type == TYPE_PARENT) {{
            if (load_u16(node + 4) & DAMAGE_CORRUPTED) continue;
            if (depth + 1 >= static_cast<int>(MAX_DEPTH)) {{
                status = {STATUS_CAPACITY};
                break;
            }}
            ++depth;
            cursor[depth] = node_offset + XTC_HEADER;
            end[depth] = node_end;
        }} else if (type == TYPE_SHAPES_DATA) {{
            if (count == ref_capacity) {{
                status = {STATUS_CAPACITY};
                break;
            }}
            const unsigned long long names_index = find_names(
                stream_id,
                load_u32(node),
                stream_names_index,
                names);
            if (names_index == ~0ull) {{
                status = {STATUS_UNKNOWN_NAMES};
                break;
            }}
            unsigned long long* row =
                refs + (index * ref_capacity + count) * {REF_NCOLS};
            row[{REF_DGRAM_INDEX}] = input_bases ? input_bases[index * 3 + 2] : index;
            row[{REF_CONFIG_NAMES_INDEX}] = names_index;
            row[{REF_OFFSET}] = dgram_offset + node_offset;
            row[{REF_EXTENT}] = extent;
            row[{REF_DAMAGE}] = load_u16(node + 4);
            ++count;
        }}
    }}

    counts[index] = count;
    dgram[{DGRAM_STATUS}] = status;
}}

__device__ __forceinline__
void locate_field_ref(const unsigned char* data,
                  const unsigned long long* refs,
                  const unsigned long long* counts,
                  const unsigned long long* dgrams,
                  unsigned long long dgram_index,
                  unsigned long long ref_index,
                  unsigned long long ref_capacity,
                  const unsigned long long* names,
                  const unsigned long long* fields,
                  unsigned long long n_names,
                  unsigned long long target_names_index,
                  unsigned long long target_field_index,
                  unsigned long long* locators)
{{
    const unsigned long long work = dgram_index * ref_capacity + ref_index;
    unsigned long long* locator = locators + dgram_index * {LOC_NCOLS};
    const unsigned long long dgram_status =
        dgrams[dgram_index * {DGRAM_NCOLS} + {DGRAM_STATUS}];
    if (dgram_status != {STATUS_OK}) {{
        atomicExch(locator + {LOC_STATUS}, dgram_status);
        return;
    }}
    if (ref_index >= counts[dgram_index]) return;

    const unsigned long long* ref = refs + work * {REF_NCOLS};
    if (ref[{REF_CONFIG_NAMES_INDEX}] != target_names_index) return;
    const unsigned long long previous = atomicCAS(
        locator + {LOC_STATUS},
        static_cast<unsigned long long>({STATUS_NOT_PRESENT}),
        static_cast<unsigned long long>({STATUS_CLAIMED}));
    if (previous != {STATUS_NOT_PRESENT}) {{
        atomicExch(locator + {LOC_STATUS},
                   static_cast<unsigned long long>({STATUS_DUPLICATE}));
        return;
    }}
    if (target_names_index >= n_names) {{
        finish_locator(locator, {STATUS_BAD_CONFIG});
        return;
    }}
    if (ref[{REF_DAMAGE}] & DAMAGE_CORRUPTED) {{
        finish_locator(locator, {STATUS_CORRUPTED});
        return;
    }}

    const unsigned long long* names_row =
        names + target_names_index * {NAMES_NCOLS};
    const unsigned long long first_field = names_row[{NAMES_FIRST_FIELD}];
    const unsigned long long n_fields = names_row[{NAMES_N_FIELDS}];
    if (target_field_index < first_field ||
        target_field_index - first_field >= n_fields) {{
        finish_locator(locator, {STATUS_BAD_CONFIG});
        return;
    }}

    const unsigned long long node_offset = ref[{REF_OFFSET}];
    const unsigned long long node_end = node_offset + ref[{REF_EXTENT}];
    unsigned long long child = node_offset + XTC_HEADER;
    unsigned long long shapes_payload = 0;
    unsigned long long shapes_nbytes = 0;
    unsigned long long data_payload = 0;
    unsigned long long data_nbytes = 0;
    while (child < node_end) {{
        if (node_end - child < XTC_HEADER) {{
            finish_locator(locator, {STATUS_BAD_XTC});
            return;
        }}
        const unsigned char* child_ptr = data + child;
        const unsigned int child_extent = load_u32(child_ptr + 8);
        if (child_extent < XTC_HEADER || child_extent > node_end - child) {{
            finish_locator(locator, {STATUS_BAD_XTC});
            return;
        }}
        const unsigned int type = load_u16(child_ptr + 6) & TYPE_MASK;
        if (type == TYPE_SHAPES) {{
            shapes_payload = child + XTC_HEADER;
            shapes_nbytes = child_extent - XTC_HEADER;
        }} else if (type == TYPE_DATA) {{
            data_payload = child + XTC_HEADER;
            data_nbytes = child_extent - XTC_HEADER;
        }}
        child += child_extent;
    }}
    if (!data_payload) {{
        finish_locator(locator, {STATUS_MISSING_DATA});
        return;
    }}

    unsigned long long field_offset = 0;
    for (unsigned long long field_index = first_field;
         field_index <= target_field_index;
         ++field_index) {{
        const unsigned long long* field = fields + field_index * {FIELD_NCOLS};
        unsigned long long field_nbytes = field[{FIELD_ELEMENT_SIZE}];
        const unsigned long long rank = field[{FIELD_RANK}];
        if (rank > {LOC_MAX_RANK}) {{
            finish_locator(locator, {STATUS_BAD_SHAPE});
            return;
        }}
        if (rank) {{
            if (!shapes_payload) {{
                finish_locator(locator, {STATUS_MISSING_SHAPES});
                return;
            }}
            const unsigned long long shape_index = field[{FIELD_SHAPE_INDEX}];
            if (shape_index == ~0ull || shape_index >= (~0ull) / SHAPE_BYTES) {{
                finish_locator(locator, {STATUS_BAD_SHAPE});
                return;
            }}
            const unsigned long long shape_offset = shape_index * SHAPE_BYTES;
            if (shape_offset > shapes_nbytes ||
                SHAPE_BYTES > shapes_nbytes - shape_offset) {{
                finish_locator(locator, {STATUS_BAD_SHAPE});
                return;
            }}
            for (unsigned long long dim_index = 0;
                 dim_index < rank;
                 ++dim_index) {{
                const unsigned long long dim = load_u32(
                    data + shapes_payload + shape_offset + dim_index * 4);
                if (field_index == target_field_index) {{
                    locator[{LOC_DIM0} + dim_index] = dim;
                }}
                if (dim && field_nbytes > (~0ull) / dim) {{
                    finish_locator(locator, {STATUS_BAD_SHAPE});
                    return;
                }}
                field_nbytes *= dim;
            }}
        }}
        if (field_offset > data_nbytes ||
            field_nbytes > data_nbytes - field_offset) {{
            finish_locator(locator, {STATUS_DATA_OVERFLOW});
            return;
        }}
        if (field_index == target_field_index) {{
            locator[{LOC_CONFIG_FIELD_INDEX}] = field_index;
            locator[{LOC_TYPE}] = field[{FIELD_TYPE}];
            locator[{LOC_RANK}] = rank;
            locator[{LOC_OFFSET}] = data_payload + field_offset;
            locator[{LOC_NBYTES}] = field_nbytes;
            finish_locator(locator, {STATUS_FOUND});
            return;
        }}
        field_offset += field_nbytes;
    }}
    finish_locator(locator, {STATUS_BAD_CONFIG});
}}

extern "C" __global__
void locate_field(const unsigned char* data,
                  const unsigned long long* refs,
                  const unsigned long long* counts,
                  const unsigned long long* dgrams,
                  unsigned long long n_dgrams,
                  unsigned long long ref_capacity,
                  const unsigned long long* names,
                  const unsigned long long* fields,
                  unsigned long long n_names,
                  unsigned long long target_names_index,
                  unsigned long long target_field_index,
                  unsigned long long* locators)
{{
    const unsigned long long work =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (work >= n_dgrams * ref_capacity) return;
    locate_field_ref(data, refs, counts, dgrams, work / ref_capacity,
                     work % ref_capacity, ref_capacity, names, fields, n_names,
                     target_names_index, target_field_index, locators);
}}

extern "C" __global__
void init_locators(unsigned long long* locators,
                   const unsigned long long* dgrams,
                   unsigned long long n_dgrams,
                   unsigned long long capacity,
                   unsigned long long n_handles)
{{
    const unsigned long long row =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= n_handles * n_dgrams) return;
    const unsigned long long dgram = row % n_dgrams;
    unsigned long long* locator =
        locators + ((row / n_dgrams) * capacity + dgram) * {LOC_NCOLS};
    for (int column = 0; column < {LOC_NCOLS}; ++column) locator[column] = 0;
    const unsigned long long status = dgrams[dgram * {DGRAM_NCOLS} + {DGRAM_STATUS}];
    locator[{LOC_STATUS}] = status == {STATUS_OK} ? {STATUS_NOT_PRESENT} : status;
}}

extern "C" __global__
void locate_fields(const unsigned char* data,
                   const unsigned long long* refs,
                   const unsigned long long* counts,
                   const unsigned long long* dgrams,
                   unsigned long long ref_capacity,
                   const unsigned long long* names,
                   const unsigned long long* fields,
                   unsigned long long n_names,
                   unsigned long long n_streams,
                   const unsigned long long* stream_handles,
                   const unsigned long long* handles,
                   unsigned long long capacity,
                   unsigned long long* locators,
                   const unsigned long long* input_bases)
{{
    // One block per dgram; threads span only its stream's handles and actual
    // references. Device metadata stays on the GPU throughout scheduling.
    const unsigned long long dgram = blockIdx.x;
    if (input_bases)
        data = reinterpret_cast<const unsigned char*>(input_bases[dgram * 3]);
    const unsigned long long* record = dgrams + dgram * {DGRAM_NCOLS};
    if (record[{DGRAM_STATUS}] != {STATUS_OK}) return;
    const unsigned long long stream = record[{DGRAM_STREAM_ID}];
    if (stream >= n_streams) return;
    const unsigned long long begin = stream_handles[stream];
    const unsigned long long count = counts[dgram];
    const unsigned long long n_work = (stream_handles[stream + 1] - begin) * count;
    for (unsigned long long work = threadIdx.x; work < n_work; work += blockDim.x) {{
        const unsigned long long* handle = handles + (begin + work / count) * {HANDLE_NCOLS};
        locate_field_ref(data, refs, counts, dgrams, dgram, work % count,
                         ref_capacity, names, fields, n_names,
                         handle[{HANDLE_NAMES_INDEX}], handle[{HANDLE_FIELD_INDEX}],
                         locators + handle[{HANDLE_OUTPUT_INDEX}] * capacity * {LOC_NCOLS});
    }}
}}

"""


__all__ = [
    "DeviceFieldLocators",
    "GpuEventBatch",
    "LOC_CONFIG_FIELD_INDEX",
    "LOC_DIM0",
    "LOC_MAX_RANK",
    "LOC_NBYTES",
    "LOC_NCOLS",
    "LOC_OFFSET",
    "LOC_RANK",
    "LOC_STATUS",
    "LOC_TYPE",
    "REF_CONFIG_NAMES_INDEX",
    "REF_DAMAGE",
    "REF_DGRAM_INDEX",
    "REF_EXTENT",
    "REF_NCOLS",
    "REF_OFFSET",
    "STATUS_FOUND",
    "STATUS_NAMES",
    "STATUS_NOT_PRESENT",
    "STATUS_OK",
]
