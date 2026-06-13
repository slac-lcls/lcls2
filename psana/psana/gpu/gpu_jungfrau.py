from functools import lru_cache
from pathlib import Path
from dataclasses import dataclass

import numpy as np

from psana.gpu.config.gpu_jungfrau_config import (
    GPU_JF_CFG_DIM0,
    GPU_JF_CFG_DIM1,
    GPU_JF_CFG_DIM2,
    GPU_JF_CFG_DTYPE_SIZE,
    GPU_JF_CFG_NAMES_ID_VALUE,
    GPU_JF_CFG_NCOLS,
    GPU_JF_CFG_RAW_DATA_OFFSET,
    GPU_JF_CFG_SEGMENT,
    GPU_JF_CFG_STREAM_ID,
)
from psana.gpu.gpu_kvikio_read import (
    DESC_DEVICE_OFFSET,
    DESC_EVENT_INDEX,
    DESC_NCOLS,
    DESC_READ_SIZE,
    DESC_STREAM_ID,
    DESC_TIMESTAMP,
)


JF_LOC_DESC_INDEX = 0
JF_LOC_CONFIG_INDEX = 1
JF_LOC_EVENT_INDEX = 2
JF_LOC_STREAM_ID = 3
JF_LOC_TIMESTAMP = 4
JF_LOC_NAMES_ID_VALUE = 5
JF_LOC_SEGMENT = 6
JF_LOC_RAW_DEVICE_OFFSET = 7
JF_LOC_RAW_NBYTES = 8
JF_LOC_DIM0 = 9
JF_LOC_DIM1 = 10
JF_LOC_DIM2 = 11
JF_LOC_DTYPE_SIZE = 12
JF_LOC_STATUS = 13
JF_LOC_NCOLS = 14

JF_LOC_STATUS_SKIPPED = 0
JF_LOC_STATUS_FOUND = 1
JF_LOC_STATUS_NOT_FOUND = 2
JF_LOC_STATUS_BAD_DGRAM = 3
JF_LOC_STATUS_BAD_XTC = 4
JF_LOC_STATUS_DATA_TOO_SMALL = 5


LOCATE_KERNEL_NAME = "locate_jungfrau_raw_kernel"
ASSEMBLE_KERNEL_NAME = "assemble_jungfrau_raw_kernel"
INVALID_SEGMENT_ROW = np.iinfo(np.uint64).max


@dataclass(frozen=True)
class JungfrauRawLayout:
    segments: np.ndarray
    segment_to_row_gpu: object
    raw_shape: tuple
    dim0: int
    dim1: int
    dim2: int

    @property
    def n_segments(self):
        return int(self.segments.shape[0])

    @property
    def pixels_per_segment(self):
        return int(self.dim0 * self.dim1 * self.dim2)


def prepare_jungfrau_raw_layout(jungfrau_config_table, cp=None):
    if not jungfrau_config_table:
        return None

    if cp is None:
        cp = _cupy()

    rows = jungfrau_config_table.rows
    segments = np.asarray(sorted(set(rows["segment"].astype(np.uint64))), dtype=np.uint64)
    if segments.size == 0:
        return None

    dim0 = _single_config_value(rows, "dim0")
    dim1 = _single_config_value(rows, "dim1")
    dim2 = _single_config_value(rows, "dim2")
    dtype_size = _single_config_value(rows, "dtype_size")
    if dtype_size != np.dtype(np.uint16).itemsize:
        raise ValueError(f"Jungfrau raw dtype_size must be 2, got {dtype_size}")

    segment_to_row = np.full(
        int(segments.max()) + 1,
        INVALID_SEGMENT_ROW,
        dtype=np.uint64,
    )
    segment_to_row[segments] = np.arange(segments.size, dtype=np.uint64)

    return JungfrauRawLayout(
        segments=segments,
        segment_to_row_gpu=cp.asarray(segment_to_row),
        raw_shape=(int(segments.size) * dim0, dim1, dim2),
        dim0=dim0,
        dim1=dim1,
        dim2=dim2,
    )


def assemble_jungfrau_raw(data_gpu, loc_gpu, layout, n_events, threads=256):
    """
    Assemble located Jungfrau raw payloads into det.raw.raw-compatible order.

    Returns a CuPy uint16 array with shape (n_events,) + layout.raw_shape.
    For Jungfrau this is normally (n_events, n_segments, 512, 1024).
    """
    cp = _cupy()

    if data_gpu.dtype != cp.uint8:
        raise TypeError(f"data_gpu must have dtype uint8, got {data_gpu.dtype}")

    if loc_gpu.dtype != cp.uint64:
        raise TypeError(f"loc_gpu must have dtype uint64, got {loc_gpu.dtype}")

    if loc_gpu.ndim != 2 or loc_gpu.shape[1] != JF_LOC_NCOLS:
        raise ValueError(
            f"loc_gpu must have shape (n, {JF_LOC_NCOLS}), got {loc_gpu.shape}"
        )

    n_events = int(n_events)
    raw_gpu = cp.zeros((n_events,) + layout.raw_shape, dtype=cp.uint16)
    n_loc = int(loc_gpu.shape[0])
    if n_events == 0 or n_loc == 0 or layout.n_segments == 0:
        return raw_gpu

    n_work = n_loc * layout.pixels_per_segment
    blocks = (n_work + threads - 1) // threads
    kernel = _assemble_jungfrau_raw_kernel()
    kernel(
        (blocks,),
        (threads,),
        (
            data_gpu,
            loc_gpu,
            layout.segment_to_row_gpu,
            raw_gpu,
            np.uint64(n_loc),
            np.uint64(n_events),
            np.uint64(layout.n_segments),
            np.uint64(layout.segment_to_row_gpu.shape[0]),
            np.uint64(layout.pixels_per_segment),
        ),
    )
    return raw_gpu


def _single_config_value(rows, name):
    values = set(int(value) for value in rows[name])
    if len(values) != 1:
        raise ValueError(f"Jungfrau config has inconsistent {name}: {sorted(values)}")
    return values.pop()


def locate_jungfrau_raw(
    data_gpu,
    desc_table_gpu,
    jungfrau_config_gpu,
    n_desc=None,
    threads=128,
):
    """
    Locate Jungfrau raw array payloads in GPU-resident dgrams.

    The returned CuPy uint64 table has one row for each
    (read descriptor, Jungfrau config row) pair.  Rows with
    JF_LOC_STATUS_FOUND contain a device offset into data_gpu.
    """
    cp = _cupy()

    if data_gpu.dtype != cp.uint8:
        raise TypeError(f"data_gpu must have dtype uint8, got {data_gpu.dtype}")

    if desc_table_gpu.dtype != cp.uint64:
        raise TypeError(
            f"desc_table_gpu must have dtype uint64, got {desc_table_gpu.dtype}"
        )

    if desc_table_gpu.ndim != 2 or desc_table_gpu.shape[1] != DESC_NCOLS:
        raise ValueError(
            f"desc_table_gpu must have shape (n, {DESC_NCOLS}), "
            f"got {desc_table_gpu.shape}"
        )

    if jungfrau_config_gpu.dtype != cp.uint64:
        raise TypeError(
            "jungfrau_config_gpu must have dtype uint64, "
            f"got {jungfrau_config_gpu.dtype}"
        )

    if (
        jungfrau_config_gpu.ndim != 2
        or jungfrau_config_gpu.shape[1] != GPU_JF_CFG_NCOLS
    ):
        raise ValueError(
            "jungfrau_config_gpu must have shape "
            f"(n, {GPU_JF_CFG_NCOLS}), got {jungfrau_config_gpu.shape}"
        )

    if n_desc is None:
        n_desc = int(desc_table_gpu.shape[0])
    else:
        n_desc = int(n_desc)
    n_config = int(jungfrau_config_gpu.shape[0])

    loc_gpu = cp.empty((n_desc * n_config, JF_LOC_NCOLS), dtype=cp.uint64)
    if n_desc == 0 or n_config == 0:
        return loc_gpu

    n_work = n_desc * n_config
    blocks = (n_work + threads - 1) // threads
    kernel = _locate_jungfrau_raw_kernel()
    kernel(
        (blocks,),
        (threads,),
        (
            data_gpu,
            desc_table_gpu,
            jungfrau_config_gpu,
            loc_gpu,
            np.uint64(n_desc),
            np.uint64(n_config),
        ),
    )
    return loc_gpu


@lru_cache(maxsize=1)
def _cupy():
    import cupy as cp

    return cp


@lru_cache(maxsize=1)
def _locate_jungfrau_raw_kernel():
    cp = _cupy()
    return cp.RawKernel(
        _kernel_source(),
        LOCATE_KERNEL_NAME,
        options=("--std=c++17",),
    )


@lru_cache(maxsize=1)
def _assemble_jungfrau_raw_kernel():
    cp = _cupy()
    return cp.RawKernel(
        _kernel_source(),
        ASSEMBLE_KERNEL_NAME,
        options=("--std=c++17",),
    )


@lru_cache(maxsize=1)
def _kernel_source():
    header_path = Path(__file__).with_name("cuda") / "gpudgramlite.cuh"
    header = header_path.read_text()

    return header + f"""

namespace {{

static constexpr unsigned int TYPE_PARENT = 0;
static constexpr unsigned int TYPE_SHAPES_DATA = 1;
static constexpr unsigned int TYPE_DATA = 3;
static constexpr unsigned int TYPE_ID_MASK = 0x0fff;

static constexpr unsigned long long STATUS_SKIPPED = {JF_LOC_STATUS_SKIPPED};
static constexpr unsigned long long STATUS_FOUND = {JF_LOC_STATUS_FOUND};
static constexpr unsigned long long STATUS_NOT_FOUND = {JF_LOC_STATUS_NOT_FOUND};
static constexpr unsigned long long STATUS_BAD_DGRAM = {JF_LOC_STATUS_BAD_DGRAM};
static constexpr unsigned long long STATUS_BAD_XTC = {JF_LOC_STATUS_BAD_XTC};
static constexpr unsigned long long STATUS_DATA_TOO_SMALL =
    {JF_LOC_STATUS_DATA_TOO_SMALL};

__device__ unsigned int type_id(const psana_gpu::GpuXtcLite& xtc)
{{
    return xtc.contains() & TYPE_ID_MASK;
}}

__device__ bool find_data_payload(const unsigned char* shapes_data,
                                  const unsigned char* shapes_data_end,
                                  const unsigned char** data_payload,
                                  unsigned long long* data_payload_nbytes)
{{
    const unsigned char* child = shapes_data + psana_gpu::XTC_HEADER_NBYTES;

    while (child + psana_gpu::XTC_HEADER_NBYTES <= shapes_data_end) {{
        const psana_gpu::GpuXtcLite child_xtc(child);
        const unsigned int child_extent = child_xtc.extent();
        if (child_extent < psana_gpu::XTC_HEADER_NBYTES ||
            child + child_extent > shapes_data_end) {{
            return false;
        }}

        if (type_id(child_xtc) == TYPE_DATA) {{
            *data_payload = child_xtc.payload();
            *data_payload_nbytes = child_xtc.payload_size();
            return true;
        }}

        child += child_extent;
    }}

    return false;
}}

__device__ unsigned long long scan_children_for_raw(
    const unsigned char* dgram,
    unsigned long long device_offset,
    const unsigned char* begin,
    const unsigned char* end,
    unsigned long long names_id_value,
    unsigned long long raw_data_offset,
    unsigned long long raw_nbytes,
    unsigned long long* raw_device_offset)
{{
    const unsigned char* child = begin;
    while (child + psana_gpu::XTC_HEADER_NBYTES <= end) {{
        const psana_gpu::GpuXtcLite child_xtc(child);
        const unsigned int child_extent = child_xtc.extent();
        if (child_extent < psana_gpu::XTC_HEADER_NBYTES ||
            child + child_extent > end) {{
            return STATUS_BAD_XTC;
        }}

        if (type_id(child_xtc) == TYPE_SHAPES_DATA &&
            static_cast<unsigned long long>(child_xtc.src()) == names_id_value) {{
            const unsigned char* data_payload = nullptr;
            unsigned long long data_payload_nbytes = 0;
            if (!find_data_payload(
                    child,
                    child + child_extent,
                    &data_payload,
                    &data_payload_nbytes)) {{
                return STATUS_BAD_XTC;
            }}

            if (raw_data_offset + raw_nbytes > data_payload_nbytes) {{
                return STATUS_DATA_TOO_SMALL;
            }}

            *raw_device_offset =
                device_offset +
                static_cast<unsigned long long>(data_payload - dgram) +
                raw_data_offset;
            return STATUS_FOUND;
        }}

        child += child_extent;
    }}

    return STATUS_NOT_FOUND;
}}

}} // namespace

extern "C" __global__
void {LOCATE_KERNEL_NAME}(const unsigned char* data,
                          const unsigned long long* desc_table,
                          const unsigned long long* config_table,
                          unsigned long long* loc_table,
                          unsigned long long n_desc,
                          unsigned long long n_config)
{{
    const unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long n_work = n_desc * n_config;
    if (i >= n_work) {{
        return;
    }}

    const unsigned long long desc_index = i / n_config;
    const unsigned long long config_index = i - desc_index * n_config;
    const unsigned long long* desc = desc_table + desc_index * {DESC_NCOLS};
    const unsigned long long* config =
        config_table + config_index * {GPU_JF_CFG_NCOLS};
    unsigned long long* loc = loc_table + i * {JF_LOC_NCOLS};

    const unsigned long long stream_id = desc[{DESC_STREAM_ID}];
    const unsigned long long config_stream_id = config[{GPU_JF_CFG_STREAM_ID}];
    const unsigned long long dim0 = config[{GPU_JF_CFG_DIM0}];
    const unsigned long long dim1 = config[{GPU_JF_CFG_DIM1}];
    const unsigned long long dim2 = config[{GPU_JF_CFG_DIM2}];
    const unsigned long long dtype_size = config[{GPU_JF_CFG_DTYPE_SIZE}];
    const unsigned long long raw_nbytes = dim0 * dim1 * dim2 * dtype_size;

    loc[{JF_LOC_DESC_INDEX}] = desc_index;
    loc[{JF_LOC_CONFIG_INDEX}] = config_index;
    loc[{JF_LOC_EVENT_INDEX}] = desc[{DESC_EVENT_INDEX}];
    loc[{JF_LOC_STREAM_ID}] = stream_id;
    loc[{JF_LOC_TIMESTAMP}] = desc[{DESC_TIMESTAMP}];
    loc[{JF_LOC_NAMES_ID_VALUE}] = config[{GPU_JF_CFG_NAMES_ID_VALUE}];
    loc[{JF_LOC_SEGMENT}] = config[{GPU_JF_CFG_SEGMENT}];
    loc[{JF_LOC_RAW_DEVICE_OFFSET}] = 0;
    loc[{JF_LOC_RAW_NBYTES}] = raw_nbytes;
    loc[{JF_LOC_DIM0}] = dim0;
    loc[{JF_LOC_DIM1}] = dim1;
    loc[{JF_LOC_DIM2}] = dim2;
    loc[{JF_LOC_DTYPE_SIZE}] = dtype_size;
    loc[{JF_LOC_STATUS}] = STATUS_SKIPPED;

    if (stream_id != config_stream_id) {{
        return;
    }}

    const unsigned long long read_size = desc[{DESC_READ_SIZE}];
    const unsigned long long device_offset = desc[{DESC_DEVICE_OFFSET}];
    if (read_size < psana_gpu::DGRAM_XTC_OFFSET +
                    psana_gpu::XTC_HEADER_NBYTES) {{
        loc[{JF_LOC_STATUS}] = STATUS_BAD_DGRAM;
        return;
    }}

    const unsigned char* dgram = data + device_offset;
    const psana_gpu::GpuDgramLite dg(dgram);
    const psana_gpu::GpuXtcLite root = dg.xtc();
    const unsigned int root_extent = root.extent();
    if (root_extent < psana_gpu::XTC_HEADER_NBYTES ||
        psana_gpu::DGRAM_XTC_OFFSET +
            static_cast<unsigned long long>(root_extent) > read_size) {{
        loc[{JF_LOC_STATUS}] = STATUS_BAD_DGRAM;
        return;
    }}

    const unsigned char* root_begin = root.payload();
    const unsigned char* root_end =
        dgram + psana_gpu::DGRAM_XTC_OFFSET + root_extent;
    const unsigned long long status = scan_children_for_raw(
        dgram,
        device_offset,
        root_begin,
        root_end,
        config[{GPU_JF_CFG_NAMES_ID_VALUE}],
        config[{GPU_JF_CFG_RAW_DATA_OFFSET}],
        raw_nbytes,
        loc + {JF_LOC_RAW_DEVICE_OFFSET});

    loc[{JF_LOC_STATUS}] = status;
}}

extern "C" __global__
void {ASSEMBLE_KERNEL_NAME}(const unsigned char* data,
                            const unsigned long long* loc_table,
                            const unsigned long long* segment_to_row,
                            unsigned short* raw,
                            unsigned long long n_loc,
                            unsigned long long n_events,
                            unsigned long long n_segment_rows,
                            unsigned long long segment_to_row_size,
                            unsigned long long pixels_per_segment)
{{
    const unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long n_work = n_loc * pixels_per_segment;
    if (i >= n_work) {{
        return;
    }}

    const unsigned long long loc_index = i / pixels_per_segment;
    const unsigned long long pixel_index = i - loc_index * pixels_per_segment;
    const unsigned long long* loc = loc_table + loc_index * {JF_LOC_NCOLS};

    if (loc[{JF_LOC_STATUS}] != STATUS_FOUND) {{
        return;
    }}

    const unsigned long long event_index = loc[{JF_LOC_EVENT_INDEX}];
    const unsigned long long segment = loc[{JF_LOC_SEGMENT}];
    if (event_index >= n_events || segment >= segment_to_row_size) {{
        return;
    }}

    const unsigned long long segment_row = segment_to_row[segment];
    if (segment_row == {INVALID_SEGMENT_ROW} ||
        segment_row >= n_segment_rows) {{
        return;
    }}

    const unsigned long long raw_device_offset = loc[{JF_LOC_RAW_DEVICE_OFFSET}];
    const unsigned short* src =
        reinterpret_cast<const unsigned short*>(data + raw_device_offset);
    unsigned short* dst =
        raw + (event_index * n_segment_rows + segment_row) * pixels_per_segment;
    dst[pixel_index] = src[pixel_index];
}}
"""
