from functools import lru_cache
from dataclasses import dataclass

import numpy as np

from psana.gpu.gpu_kvikio_read import (
    DESC_DEVICE_OFFSET,
    DESC_EVENT_INDEX,
    DESC_STREAM_ID,
    DESC_TIMESTAMP,
)


JF_LOC_DESC_INDEX = 0
JF_LOC_RAW_ROW_INDEX = 1
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

JF_LOC_STATUS_FOUND = 1


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


def prepare_jungfrau_raw_layout(raw_offset_source, cp=None):
    if raw_offset_source is None:
        return None

    if cp is None:
        cp = _cupy()

    rows = raw_offset_source.rows
    if rows.size == 0:
        return None

    segments = np.asarray(
        sorted(set(rows["segment"].astype(np.uint64))),
        dtype=np.uint64,
    )
    if segments.size == 0:
        return None

    dim0 = _single_value(rows, "dim0")
    dim1 = _single_value(rows, "dim1")
    dim2 = _single_value(rows, "dim2")
    dtype_size = _single_value(rows, "dtype_size")
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


def build_jungfrau_raw_loc_table(desc_table, raw_offset_cache):
    """
    Build the Jungfrau raw locator table on CPU from cached dgram offsets.

    The GPU assembly kernel only consumes this table and copies raw bytes; it
    does not parse Dgram/XTC headers.
    """
    desc_table = np.asarray(desc_table, dtype=np.uint64)
    if raw_offset_cache is None or desc_table.size == 0:
        return np.empty((0, JF_LOC_NCOLS), dtype=np.uint64)

    loc_rows = []
    for desc_index, desc in enumerate(desc_table):
        stream_id = int(desc[DESC_STREAM_ID])
        device_offset = int(desc[DESC_DEVICE_OFFSET])
        raw_rows = raw_offset_cache.rows_for_stream(stream_id)

        for raw_row_index, raw_row in enumerate(raw_rows):
            loc_rows.append(
                (
                    desc_index,
                    raw_row_index,
                    int(desc[DESC_EVENT_INDEX]),
                    stream_id,
                    int(desc[DESC_TIMESTAMP]),
                    int(raw_row["names_id_value"]),
                    int(raw_row["segment"]),
                    device_offset + int(raw_row["raw_rel_offset"]),
                    int(raw_row["raw_nbytes"]),
                    int(raw_row["dim0"]),
                    int(raw_row["dim1"]),
                    int(raw_row["dim2"]),
                    int(raw_row["dtype_size"]),
                    JF_LOC_STATUS_FOUND,
                )
            )

    return np.asarray(loc_rows, dtype=np.uint64).reshape(-1, JF_LOC_NCOLS)


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


def _single_value(rows, name):
    values = set(int(value) for value in rows[name])
    if len(values) != 1:
        raise ValueError(
            f"Jungfrau raw offsets have inconsistent {name}: {sorted(values)}"
        )
    return values.pop()


@lru_cache(maxsize=1)
def _cupy():
    import cupy as cp

    return cp


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
    return f"""

namespace {{

static constexpr unsigned long long STATUS_FOUND = {JF_LOC_STATUS_FOUND};

}} // namespace

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
