"""
gpu_calib.py — GPU-accelerated Jungfrau calibration.

Public API
----------
prep_calib_constants(det) -> (peds_gpu, gmask_gpu)

    Extract pedestals and gain*mask from a psana Detector object,
    compute gmask = (1/pixel_gain) * mask on CPU, transfer both to GPU.
    Call once at BeginRun; re-call after BeginStep if constants change.

fused_calib_gpu(raw_gpu, peds_gpu, gmask_gpu) -> calib_gpu

    Run the Jungfrau calibration kernel.  raw_gpu must already be on device.
    Returns a float32 CuPy array of the same shape.

Calibration constant layout (peds_gpu and gmask_gpu):
    flat float32, length 3 * npixels, mode-major C order.
    Index: mode * npixels + pixel_index
    where npixels = nsegs * nrows * ncols.

Gain-bit to mode mapping (top 2 bits of each raw uint16):
    0b00 -> mode 0  (g0 / low-gain)
    0b01 -> mode 1  (g1 / medium-gain)
    0b11 -> mode 2  (g2 / high-gain)
    0b10 -> bad pixel -> output 0.0
"""

from functools import lru_cache
from pathlib import Path

import numpy as np

_KERNEL_NAME = "jungfrau_calib_kernel"


def prep_calib_constants(det):
    """Transfer Jungfrau calibration constants to GPU.

    Reads pedestals, pixel_gain, and pixel_mask from det.calibconst,
    computes gmask = (1/gain) * mask on CPU, copies both to device.

    Parameters
    ----------
    det : psana Detector object (calibconst already loaded)

    Returns
    -------
    peds_gpu  : cp.ndarray float32, flat, length 3 * npixels
    gmask_gpu : cp.ndarray float32, flat, same length
    """
    import cupy as cp
    peds_flat, gmask_flat = _compute_calib_constants(det)
    return cp.asarray(peds_flat), cp.asarray(gmask_flat)


def fused_calib_gpu(raw_gpu, peds_gpu, gmask_gpu, threads=256):
    """Run Jungfrau 3-gain-mode calibration on GPU.

    Parameters
    ----------
    raw_gpu   : cp.ndarray uint16, any shape
    peds_gpu  : cp.ndarray float32, flat, length 3 * raw_gpu.size
    gmask_gpu : cp.ndarray float32, flat, length 3 * raw_gpu.size
    threads   : CUDA threads per block (default 256)

    Returns
    -------
    cp.ndarray float32, same shape as raw_gpu
    """
    cp = _cupy()
    npixels = int(raw_gpu.size)

    if raw_gpu.dtype != cp.uint16:
        raise TypeError(f"raw_gpu must be uint16, got {raw_gpu.dtype}")
    if peds_gpu.size != 3 * npixels:
        raise ValueError(
            f"peds_gpu length {peds_gpu.size} != 3 * npixels ({3 * npixels})"
        )
    if gmask_gpu.size != 3 * npixels:
        raise ValueError(
            f"gmask_gpu length {gmask_gpu.size} != 3 * npixels ({3 * npixels})"
        )

    calib_gpu = cp.empty(npixels, dtype=cp.float32)
    blocks = (npixels + threads - 1) // threads
    _jungfrau_calib_kernel()(
        (blocks,),
        (threads,),
        (
            raw_gpu.ravel(),
            peds_gpu.ravel(),
            gmask_gpu.ravel(),
            calib_gpu,
            np.uint64(npixels),
        ),
    )
    return calib_gpu.reshape(raw_gpu.shape)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_calib_constants(det):
    """Build flat CPU arrays for pedestals and gmask.

    Returns
    -------
    peds_flat  : np.ndarray float32, contiguous, length 3 * npixels
    gmask_flat : np.ndarray float32, contiguous, same length
    """
    cc   = det.calibconst
    peds = cc["pedestals"][0].astype(np.float32)   # (3, n_segs, nrows, ncols)
    gain = cc["pixel_gain"][0].astype(np.float32)

    # Apply pixel_offset when present (mirrors CPU DetCache behaviour).
    try:
        offset = cc.get("pixel_offset", [None])[0]
        if offset is not None:
            peds = peds + offset.astype(np.float32)
    except Exception:
        pass

    # Build mask. Priority:
    #   1. det.raw._mask(all_segs=True) — uses all calibration quality flags
    #   2. pixel_status calibconst mode-0 (0 = good)
    #   3. All-ones (no masking)
    expected_shape = peds.shape[1:]
    mask = None
    try:
        m = det.raw._mask(all_segs=True)
        if m is not None and m.shape == expected_shape:
            mask = m
    except Exception:
        pass
    if mask is None:
        try:
            status = cc["pixel_status"][0]
            mask = (status[0] == 0).astype(np.float32)
        except Exception:
            pass

    gfac = np.where(gain != 0, np.float32(1.0) / gain, np.float32(0.0))
    gmask = (gfac * mask[np.newaxis]).astype(np.float32) if mask is not None else gfac.astype(np.float32)

    return (np.ascontiguousarray(peds.ravel()),
            np.ascontiguousarray(gmask.ravel()))


@lru_cache(maxsize=1)
def _cupy():
    import cupy as cp
    return cp


@lru_cache(maxsize=1)
def _jungfrau_calib_kernel():
    cp = _cupy()
    return cp.RawKernel(
        _kernel_source(),
        _KERNEL_NAME,
        options=("--std=c++17",),
    )


@lru_cache(maxsize=1)
def _kernel_source():
    header_path = Path(__file__).with_name("cuda") / "fused_calib.cuh"
    return header_path.read_text() + f"""

extern "C" __global__
void {_KERNEL_NAME}(
    const unsigned short* raw,
    const float*          peds,
    const float*          gmask,
    float*                calib,
    unsigned long long    npixels)
{{
    const unsigned long long i =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npixels)
        return;
    calib[i] = psana_gpu::jungfrau_calib_pixel(raw[i], peds, gmask, i, npixels);
}}
"""
