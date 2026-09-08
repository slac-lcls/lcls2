"""One-thread-per-dgram GPU XTC walker and debug-facing field views."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .schema import (
    FIELD_ELEMENT_SIZE,
    FIELD_KEY,
    FIELD_NCOLS,
    FIELD_RANK,
    FIELD_SHAPE_INDEX,
    FIELD_TYPE,
)


DGRAM_OFFSET = 0
DGRAM_SIZE = 1
DGRAM_TIMESTAMP = 2
DGRAM_ENV = 3
DGRAM_SERVICE = 4
DGRAM_DAMAGE = 5
DGRAM_TYPE = 6
DGRAM_STATUS = 7
DGRAM_NCOLS = 8

REF_DGRAM_INDEX = 0
REF_NAMES_ID = 1
REF_OFFSET = 2
REF_EXTENT = 3
REF_DAMAGE = 4
REF_NCOLS = 5

LOC_FIELD_KEY = 0
LOC_TYPE = 1
LOC_RANK = 2
LOC_DIM0 = 3
LOC_MAX_RANK = 5
LOC_OFFSET = LOC_DIM0 + LOC_MAX_RANK
LOC_NBYTES = LOC_OFFSET + 1
LOC_STATUS = LOC_NBYTES + 1
LOC_NCOLS = LOC_STATUS + 1

STATUS_OK = 0
STATUS_FOUND = 1
STATUS_NOT_PRESENT = 2
STATUS_PARTIAL_DGRAM = 3
STATUS_BAD_DGRAM = 4
STATUS_BAD_XTC = 5
STATUS_CORRUPTED = 6
STATUS_CAPACITY = 7
STATUS_MISSING_SHAPES = 8
STATUS_MISSING_DATA = 9
STATUS_BAD_SHAPE = 10
STATUS_DATA_OVERFLOW = 11
STATUS_DUPLICATE = 12

STATUS_NAMES = {
    STATUS_OK: "ok",
    STATUS_FOUND: "found",
    STATUS_NOT_PRESENT: "not_present",
    STATUS_PARTIAL_DGRAM: "partial_dgram",
    STATUS_BAD_DGRAM: "bad_dgram",
    STATUS_BAD_XTC: "bad_xtc",
    STATUS_CORRUPTED: "corrupted",
    STATUS_CAPACITY: "capacity",
    STATUS_MISSING_SHAPES: "missing_shapes",
    STATUS_MISSING_DATA: "missing_data",
    STATUS_BAD_SHAPE: "bad_shape",
    STATUS_DATA_OVERFLOW: "data_overflow",
    STATUS_DUPLICATE: "duplicate",
}

_TYPE_DTYPES = {
    0: np.dtype("u1"),
    1: np.dtype("<u2"),
    2: np.dtype("<u4"),
    3: np.dtype("<u8"),
    4: np.dtype("i1"),
    5: np.dtype("<i2"),
    6: np.dtype("<i4"),
    7: np.dtype("<i8"),
    8: np.dtype("<f4"),
    9: np.dtype("<f8"),
    10: np.dtype("u1"),
    11: np.dtype("<u4"),
    12: np.dtype("<u4"),
}


@dataclass(frozen=True)
class GpuFieldView:
    name: str
    type: int
    shape: tuple
    device_offset: int
    nbytes: int
    array: object


class GpuAlgView:
    def __init__(self, schema, fields):
        self.det_name = schema.det_name
        self.det_type = schema.det_type
        self.det_id = schema.det_id
        self.segment = schema.segment
        self.alg_name = schema.alg_name
        self.alg_version = schema.alg_version
        self.names_id = schema.names_id
        self.fields = dict(fields)

    def get(self, field_name):
        return self.fields[str(field_name)]

    def __getitem__(self, field_name):
        return self.get(field_name)


class GPUDgram:
    """One indexed dgram within a :class:`GPUDgramBatch`."""

    def __init__(self, batch, index):
        self._batch = batch
        self.index = int(index)

    @property
    def info(self):
        return self._batch.dgram_info[self.index]

    @property
    def timestamp(self):
        return int(self.info[DGRAM_TIMESTAMP])

    @property
    def service(self):
        return int(self.info[DGRAM_SERVICE])

    def get(self, det_name, segment, alg_name):
        """Return GPU field views for one detector/segment/algorithm."""
        return self._batch._get(self.index, det_name, segment, alg_name)


class GPUDgramBatch:
    """Parse consecutive GPU-resident dgrams without CPU event parsing.

    ``data_gpu`` begins at a dgram boundary and may end with an incomplete
    trailing dgram.  A one-thread index pass discovers complete dgram extents;
    one CUDA thread per complete dgram then walks nested Parent records and
    records every ShapesData location.
    """

    def __init__(
        self,
        data_gpu,
        schema,
        *,
        max_dgrams=4096,
        max_shapes_per_dgram=None,
        threads=128,
    ):
        cp = _cupy()
        if data_gpu.dtype != cp.uint8 or data_gpu.ndim != 1:
            raise TypeError("data_gpu must be a one-dimensional CuPy uint8 array")
        self.data_gpu = data_gpu
        self.schema = schema
        self.device_schema = schema.to_device(cp)
        self.max_dgrams = int(max_dgrams)
        if self.max_dgrams <= 0:
            raise ValueError("max_dgrams must be positive")
        if max_shapes_per_dgram is None:
            max_shapes_per_dgram = max(1, len(schema.names))
        self.max_shapes_per_dgram = int(max_shapes_per_dgram)
        if self.max_shapes_per_dgram <= 0:
            raise ValueError("max_shapes_per_dgram must be positive")
        self.threads = int(threads)
        if self.threads <= 0:
            raise ValueError("threads must be positive")

        dgram_info_gpu = cp.zeros(
            (self.max_dgrams, DGRAM_NCOLS), dtype=cp.uint64
        )
        index_meta_gpu = cp.zeros(3, dtype=cp.uint64)
        _index_kernel()(
            (1,),
            (1,),
            (
                data_gpu,
                np.uint64(data_gpu.nbytes),
                dgram_info_gpu,
                np.uint64(self.max_dgrams),
                index_meta_gpu,
            ),
        )
        index_meta = cp.asnumpy(index_meta_gpu)
        self.n_dgrams = int(index_meta[0])
        self.index_status = int(index_meta[1])
        self.indexed_nbytes = int(index_meta[2])
        self.trailing_nbytes = int(data_gpu.nbytes) - self.indexed_nbytes
        self.dgram_info_gpu = dgram_info_gpu[: self.n_dgrams]
        self.dgram_info = cp.asnumpy(self.dgram_info_gpu)

        self.shape_counts_gpu = cp.zeros(self.n_dgrams, dtype=cp.uint64)
        self.walk_status_gpu = cp.zeros(self.n_dgrams, dtype=cp.uint64)
        self.shape_refs_gpu = cp.zeros(
            (self.n_dgrams, self.max_shapes_per_dgram, REF_NCOLS),
            dtype=cp.uint64,
        )
        if self.n_dgrams:
            blocks = (self.n_dgrams + self.threads - 1) // self.threads
            _walk_kernel()(
                (blocks,),
                (self.threads,),
                (
                    data_gpu,
                    self.dgram_info_gpu,
                    np.uint64(self.n_dgrams),
                    self.shape_refs_gpu,
                    self.shape_counts_gpu,
                    self.walk_status_gpu,
                    np.uint64(self.max_shapes_per_dgram),
                ),
            )
        self.shape_counts = cp.asnumpy(self.shape_counts_gpu)
        self.walk_status = cp.asnumpy(self.walk_status_gpu)
        self._decoded = {}

    def __len__(self):
        return self.n_dgrams

    def __getitem__(self, index):
        index = int(index)
        if index < 0:
            index += self.n_dgrams
        if index < 0 or index >= self.n_dgrams:
            raise IndexError(index)
        return GPUDgram(self, index)

    def __iter__(self):
        for index in range(self.n_dgrams):
            yield GPUDgram(self, index)

    def _decode(self, names_schema):
        cached = self._decoded.get(names_schema.names_id)
        if cached is not None:
            return cached

        cp = _cupy()
        n_fields = len(names_schema.fields)
        locators_gpu = cp.zeros(
            (self.n_dgrams, n_fields, LOC_NCOLS), dtype=cp.uint64
        )
        if n_fields:
            locators_gpu[:, :, LOC_STATUS] = STATUS_NOT_PRESENT
        n_refs = self.n_dgrams * self.max_shapes_per_dgram
        if n_refs and n_fields:
            blocks = (n_refs + self.threads - 1) // self.threads
            _decode_kernel()(
                (blocks,),
                (self.threads,),
                (
                    self.data_gpu,
                    self.shape_refs_gpu,
                    self.shape_counts_gpu,
                    np.uint64(self.n_dgrams),
                    np.uint64(self.max_shapes_per_dgram),
                    self.device_schema.fields,
                    np.uint64(names_schema.first_field),
                    np.uint64(n_fields),
                    np.uint64(names_schema.names_id),
                    locators_gpu,
                ),
            )
        locators = cp.asnumpy(locators_gpu)
        cached = (locators_gpu, locators)
        self._decoded[names_schema.names_id] = cached
        return cached

    def _get(self, dgram_index, det_name, segment, alg_name):
        walk_status = int(self.walk_status[dgram_index])
        if walk_status != STATUS_OK:
            raise RuntimeError(
                f"GPU XTC walk failed for dgram {dgram_index}: "
                f"{STATUS_NAMES.get(walk_status, walk_status)}"
            )
        matches = []
        for names_schema in self.schema.find_all(det_name, segment, alg_name):
            result = self._get_names_id(dgram_index, names_schema)
            if result is not None:
                matches.append(result)
        if len(matches) > 1:
            names_ids = ", ".join(f"0x{match.names_id:x}" for match in matches)
            raise RuntimeError(
                f"dgram contains multiple NamesIds for "
                f"{det_name}[{segment}].{alg_name}: {names_ids}"
            )
        return matches[0] if matches else None

    def _get_names_id(self, dgram_index, names_schema):
        _, locators = self._decode(names_schema)
        rows = locators[dgram_index]
        statuses = rows[:, LOC_STATUS]
        if np.all(statuses == STATUS_NOT_PRESENT):
            return None

        fields = {}
        cp = _cupy()
        for field_schema, row in zip(names_schema.fields, rows):
            status = int(row[LOC_STATUS])
            if status != STATUS_FOUND:
                raise RuntimeError(
                    f"GPU field parse failed for {names_schema.det_name}"
                    f"[{names_schema.segment}].{names_schema.alg_name}."
                    f"{field_schema.name}: "
                    f"{STATUS_NAMES.get(status, status)}"
                )
            rank = int(row[LOC_RANK])
            shape = tuple(
                int(value) for value in row[LOC_DIM0 : LOC_DIM0 + rank]
            )
            offset = int(row[LOC_OFFSET])
            nbytes = int(row[LOC_NBYTES])
            dtype = _TYPE_DTYPES[int(row[LOC_TYPE])]
            expected = int(np.prod(shape, dtype=np.uint64) if shape else 1)
            expected *= dtype.itemsize
            if expected != nbytes:
                raise RuntimeError(
                    f"locator size {nbytes} does not match shape {shape} "
                    f"and dtype {dtype} ({expected} bytes)"
                )
            array = self.data_gpu[offset : offset + nbytes].view(dtype).reshape(
                shape
            )
            fields[field_schema.name] = GpuFieldView(
                name=field_schema.name,
                type=int(row[LOC_TYPE]),
                shape=shape,
                device_offset=offset,
                nbytes=nbytes,
                array=array,
            )
        return GpuAlgView(names_schema, fields)


@lru_cache(maxsize=1)
def _cupy():
    import cupy as cp

    return cp


@lru_cache(maxsize=1)
def _index_kernel():
    return _cupy().RawKernel(_kernel_source(), "index_dgrams")


@lru_cache(maxsize=1)
def _walk_kernel():
    return _cupy().RawKernel(_kernel_source(), "walk_xtc")


@lru_cache(maxsize=1)
def _decode_kernel():
    return _cupy().RawKernel(_kernel_source(), "decode_fields")


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

__device__ __forceinline__ unsigned long long load_timestamp(
    const unsigned char* p)
{{
    return static_cast<unsigned long long>(load_u32(p)) |
           (static_cast<unsigned long long>(load_u32(p + 4)) << 32);
}}

}} // namespace

extern "C" __global__
void index_dgrams(const unsigned char* data,
                  unsigned long long nbytes,
                  unsigned long long* dgrams,
                  unsigned long long capacity,
                  unsigned long long* meta)
{{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    unsigned long long offset = 0;
    unsigned long long count = 0;
    unsigned long long status = {STATUS_OK};
    while (offset < nbytes) {{
        if (nbytes - offset < DGRAM_HEADER) {{
            status = {STATUS_PARTIAL_DGRAM};
            break;
        }}
        if (count == capacity) {{
            status = {STATUS_CAPACITY};
            break;
        }}

        const unsigned char* dgram = data + offset;
        const unsigned int extent = load_u32(dgram + 20);
        if (extent < XTC_HEADER) {{
            status = {STATUS_BAD_DGRAM};
            break;
        }}
        const unsigned long long total = 12ull + extent;
        if (total > nbytes - offset) {{
            status = {STATUS_PARTIAL_DGRAM};
            break;
        }}

        unsigned long long* row = dgrams + count * {DGRAM_NCOLS};
        const unsigned int env = load_u32(dgram + 8);
        row[{DGRAM_OFFSET}] = offset;
        row[{DGRAM_SIZE}] = total;
        row[{DGRAM_TIMESTAMP}] = load_timestamp(dgram);
        row[{DGRAM_ENV}] = env;
        row[{DGRAM_SERVICE}] = (env >> 24) & 0x0f;
        row[{DGRAM_DAMAGE}] = load_u16(dgram + 16);
        row[{DGRAM_TYPE}] = load_u16(dgram + 18) & TYPE_MASK;
        row[{DGRAM_STATUS}] = {STATUS_OK};
        ++count;
        offset += total;
    }}

    meta[0] = count;
    meta[1] = status;
    meta[2] = offset;
}}

extern "C" __global__
void walk_xtc(const unsigned char* data,
              const unsigned long long* dgrams,
              unsigned long long n_dgrams,
              unsigned long long* refs,
              unsigned long long* counts,
              unsigned long long* statuses,
              unsigned long long ref_capacity)
{{
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= n_dgrams) return;

    const unsigned long long* dgram = dgrams + index * {DGRAM_NCOLS};
    const unsigned long long dgram_offset = dgram[{DGRAM_OFFSET}];
    const unsigned long long dgram_size = dgram[{DGRAM_SIZE}];
    const unsigned char* base = data + dgram_offset;
    if (dgram_size < DGRAM_HEADER) {{
        statuses[index] = {STATUS_BAD_DGRAM};
        return;
    }}
    if (load_u16(base + 16) & DAMAGE_CORRUPTED) {{
        statuses[index] = {STATUS_CORRUPTED};
        return;
    }}
    if ((load_u16(base + 18) & TYPE_MASK) != TYPE_PARENT) {{
        statuses[index] = {STATUS_BAD_XTC};
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
            unsigned long long* row =
                refs + (index * ref_capacity + count) * {REF_NCOLS};
            row[{REF_DGRAM_INDEX}] = index;
            row[{REF_NAMES_ID}] = load_u32(node);
            row[{REF_OFFSET}] = dgram_offset + node_offset;
            row[{REF_EXTENT}] = extent;
            row[{REF_DAMAGE}] = load_u16(node + 4);
            ++count;
        }}
    }}

    counts[index] = count;
    statuses[index] = status;
}}

extern "C" __global__
void decode_fields(const unsigned char* data,
                   const unsigned long long* refs,
                   const unsigned long long* counts,
                   unsigned long long n_dgrams,
                   unsigned long long ref_capacity,
                   const unsigned long long* fields,
                   unsigned long long first_field,
                   unsigned long long n_fields,
                   unsigned long long target_names_id,
                   unsigned long long* locators)
{{
    const unsigned long long work =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long n_work = n_dgrams * ref_capacity;
    if (work >= n_work) return;
    const unsigned long long dgram_index = work / ref_capacity;
    const unsigned long long ref_index = work - dgram_index * ref_capacity;
    if (ref_index >= counts[dgram_index]) return;

    const unsigned long long* ref = refs + work * {REF_NCOLS};
    if (ref[{REF_NAMES_ID}] != target_names_id) return;
    unsigned long long* first_loc =
        locators + dgram_index * n_fields * {LOC_NCOLS};
    const unsigned long long previous = atomicCAS(
        first_loc + {LOC_STATUS},
        static_cast<unsigned long long>({STATUS_NOT_PRESENT}),
        static_cast<unsigned long long>({STATUS_OK})
    );
    if (previous != {STATUS_NOT_PRESENT}) {{
        first_loc[{LOC_STATUS}] = {STATUS_DUPLICATE};
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
            first_loc[{LOC_STATUS}] = {STATUS_BAD_XTC};
            return;
        }}
        const unsigned char* child_ptr = data + child;
        const unsigned int child_extent = load_u32(child_ptr + 8);
        if (child_extent < XTC_HEADER || child_extent > node_end - child) {{
            first_loc[{LOC_STATUS}] = {STATUS_BAD_XTC};
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
        first_loc[{LOC_STATUS}] = {STATUS_MISSING_DATA};
        return;
    }}

    unsigned long long field_offset = 0;
    for (unsigned long long i = 0; i < n_fields; ++i) {{
        const unsigned long long* field =
            fields + (first_field + i) * {FIELD_NCOLS};
        unsigned long long* loc = first_loc + i * {LOC_NCOLS};
        loc[{LOC_FIELD_KEY}] = field[{FIELD_KEY}];
        loc[{LOC_TYPE}] = field[{FIELD_TYPE}];
        loc[{LOC_RANK}] = field[{FIELD_RANK}];
        unsigned long long field_nbytes = field[{FIELD_ELEMENT_SIZE}];
        const unsigned long long rank = field[{FIELD_RANK}];
        if (rank > {LOC_MAX_RANK}) {{
            loc[{LOC_STATUS}] = {STATUS_BAD_SHAPE};
            return;
        }}
        if (rank) {{
            if (!shapes_payload) {{
                loc[{LOC_STATUS}] = {STATUS_MISSING_SHAPES};
                return;
            }}
            const unsigned long long shape_index = field[{FIELD_SHAPE_INDEX}];
            if (shape_index == ~0ull ||
                shape_index >= (~0ull) / SHAPE_BYTES) {{
                loc[{LOC_STATUS}] = {STATUS_BAD_SHAPE};
                return;
            }}
            const unsigned long long shape_offset = shape_index * SHAPE_BYTES;
            if (shape_offset > shapes_nbytes ||
                SHAPE_BYTES > shapes_nbytes - shape_offset) {{
                loc[{LOC_STATUS}] = {STATUS_BAD_SHAPE};
                return;
            }}
            for (unsigned long long dim_index = 0;
                 dim_index < rank;
                 ++dim_index) {{
                const unsigned long long dim = load_u32(
                    data + shapes_payload + shape_offset + dim_index * 4
                );
                loc[{LOC_DIM0} + dim_index] = dim;
                if (dim && field_nbytes > (~0ull) / dim) {{
                    loc[{LOC_STATUS}] = {STATUS_BAD_SHAPE};
                    return;
                }}
                field_nbytes *= dim;
            }}
        }}
        if (field_offset > data_nbytes ||
            field_nbytes > data_nbytes - field_offset) {{
            loc[{LOC_STATUS}] = {STATUS_DATA_OVERFLOW};
            return;
        }}
        loc[{LOC_OFFSET}] = data_payload + field_offset;
        loc[{LOC_NBYTES}] = field_nbytes;
        loc[{LOC_STATUS}] = {STATUS_FOUND};
        field_offset += field_nbytes;
    }}
}}
"""
