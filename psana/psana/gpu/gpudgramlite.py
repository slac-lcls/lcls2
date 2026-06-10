from functools import lru_cache
from pathlib import Path

import numpy as np

from psana.gpu.gpu_kvikio_read import (
    DESC_DEVICE_OFFSET,
    DESC_EVENT_INDEX,
    DESC_NCOLS,
    DESC_READ_SIZE,
    DESC_STREAM_ID,
)


INFO_EVENT_INDEX = 0
INFO_STREAM_ID = 1
INFO_TIMESTAMP = 2
INFO_SERVICE = 3
INFO_EXTENT = 4
INFO_PAYLOAD_SIZE = 5
INFO_DEVICE_OFFSET = 6
INFO_PAYLOAD_DEVICE_OFFSET = 7
INFO_READ_SIZE = 8
INFO_NCOLS = 9


_KERNEL_NAME = "extract_dgram_info_kernel"


def extract_dgram_info(data_gpu, desc_table_gpu, n_desc=None, threads=128):
    """
    Parse top-level Dgram/Xtc header fields on GPU.

    Returns a CuPy uint64 array with shape (n_desc, INFO_NCOLS).  The returned
    table stays on GPU; call .get() only for validation/debugging.
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

    if n_desc is None:
        n_desc = int(desc_table_gpu.shape[0])
    else:
        n_desc = int(n_desc)

    info_gpu = cp.empty((n_desc, INFO_NCOLS), dtype=cp.uint64)
    if n_desc == 0:
        return info_gpu

    blocks = (n_desc + threads - 1) // threads
    kernel = _extract_dgram_info_kernel()
    kernel(
        (blocks,),
        (threads,),
        (
            data_gpu,
            desc_table_gpu,
            info_gpu,
            np.uint64(n_desc),
        ),
    )
    return info_gpu


@lru_cache(maxsize=1)
def _cupy():
    import cupy as cp

    return cp


@lru_cache(maxsize=1)
def _extract_dgram_info_kernel():
    cp = _cupy()
    return cp.RawKernel(
        _kernel_source(),
        _KERNEL_NAME,
        options=("--std=c++17",),
    )


@lru_cache(maxsize=1)
def _kernel_source():
    header_path = Path(__file__).with_name("cuda") / "gpudgramlite.cuh"
    header = header_path.read_text()

    return header + f"""

extern "C" __global__
void {_KERNEL_NAME}(const unsigned char* data,
                    const unsigned long long* desc_table,
                    unsigned long long* info_table,
                    unsigned long long n_desc)
{{
    const unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_desc) {{
        return;
    }}

    const unsigned long long* desc = desc_table + i * {DESC_NCOLS};
    const unsigned long long device_offset = desc[{DESC_DEVICE_OFFSET}];
    const unsigned char* dgram = data + device_offset;
    const psana_gpu::GpuDgramLite dg(dgram);

    unsigned long long* info = info_table + i * {INFO_NCOLS};
    info[{INFO_EVENT_INDEX}] = desc[{DESC_EVENT_INDEX}];
    info[{INFO_STREAM_ID}] = desc[{DESC_STREAM_ID}];
    info[{INFO_TIMESTAMP}] = dg.timestamp();
    info[{INFO_SERVICE}] = dg.service();
    info[{INFO_EXTENT}] = dg.extent();
    info[{INFO_PAYLOAD_SIZE}] = dg.payload_size();
    info[{INFO_DEVICE_OFFSET}] = device_offset;
    info[{INFO_PAYLOAD_DEVICE_OFFSET}] =
        device_offset + psana_gpu::DGRAM_HEADER_NBYTES;
    info[{INFO_READ_SIZE}] = desc[{DESC_READ_SIZE}];
}}
"""
