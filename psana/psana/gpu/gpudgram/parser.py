"""Device-resident XTC walk and field-location primitives.

Stage 1 deliberately exposes numeric device tables rather than Python dgram
views.  A run owner uploads :class:`DeviceConfigTables` once.  A batch owner
provides raw bytes and dgram records already resident on the GPU; all XTC walk
results stay there for direct consumption by later CUDA kernels.
"""

from dataclasses import dataclass
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


@dataclass(frozen=True)
class DeviceFieldLocators:
    """One locator row per batch dgram for a resolved field handle.

    ``rows_gpu`` is a ``uint64[n_dgrams, LOC_NCOLS]`` CuPy array.  A consumer
    on another CUDA stream must call :meth:`wait_on` before launching work.
    """

    handle: GpuFieldHandle
    rows_gpu: object
    ready: object

    @property
    def n_dgrams(self):
        return int(self.rows_gpu.shape[0])

    def wait_on(self, stream):
        stream.wait_event(self.ready)
        return self.rows_gpu


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
    remain owned by its EventPool slot until that slot is safely retired.
    """

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
        self.device_configs = device_configs
        self.dgram_records_gpu = dgram_records_gpu
        self.n_dgrams = int(dgram_records_gpu.shape[0])
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
                    ),
                    stream=self.stream,
                )
            self.walk_done = cp.cuda.Event()
            self.walk_done.record(self.stream)
        self._locators = {}

    def locate(self, handle, *, stream=None):
        """Launch or return the device locator table for ``handle``."""
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
              unsigned long long ref_capacity)
{{
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= n_dgrams) return;

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
            row[{REF_DGRAM_INDEX}] = index;
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
    const unsigned long long n_work = n_dgrams * ref_capacity;
    if (work >= n_work) return;
    const unsigned long long dgram_index = work / ref_capacity;
    const unsigned long long ref_index = work - dgram_index * ref_capacity;
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
