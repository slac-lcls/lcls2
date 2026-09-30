"""Jungfrau GPU calibration and calibration-constant preparation."""

from functools import lru_cache
from pathlib import Path

import numpy as np


_KERNEL_NAME = "jungfrau_calib_kernel"


def fused_calib_gpu(raw_gpu, peds_gpu, gmask_gpu, threads=256, out=None):
    """Calibrate a canonical Jungfrau raw array on the GPU."""
    cp = _cupy()
    npixels = int(raw_gpu.size)

    if raw_gpu.dtype != cp.uint16:
        raise TypeError(f"raw_gpu must be uint16, got {raw_gpu.dtype}")
    if peds_gpu.dtype != cp.float32:
        raise TypeError(f"peds_gpu must be float32, got {peds_gpu.dtype}")
    if gmask_gpu.dtype != cp.float32:
        raise TypeError(f"gmask_gpu must be float32, got {gmask_gpu.dtype}")
    if peds_gpu.size != 3 * npixels:
        raise ValueError(
            f"peds_gpu length {peds_gpu.size} != 3 * npixels ({3 * npixels})"
        )
    if gmask_gpu.size != 3 * npixels:
        raise ValueError(
            f"gmask_gpu length {gmask_gpu.size} != 3 * npixels ({3 * npixels})"
        )

    if out is not None and out.size >= npixels and out.dtype == cp.float32:
        calib_gpu = out.ravel()[:npixels]
    else:
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


def _compute_calib_constants_cpu(det, canonical_segment_ids=None):
    """Return contiguous pedestal and gain/mask arrays in canonical order."""
    cc = det.calibconst
    peds = cc["pedestals"][0].astype(np.float32)
    gain = cc["pixel_gain"][0].astype(np.float32)

    try:
        offset = cc.get("pixel_offset", [None])[0]
        if offset is not None:
            peds = peds + offset.astype(np.float32)
    except Exception:
        pass

    expected_shape = peds.shape[1:]
    mask = None
    try:
        candidate = det.raw._mask(all_segs=True)
        if candidate is not None and candidate.shape == expected_shape:
            mask = candidate
    except Exception:
        pass
    if mask is None:
        try:
            status = cc["pixel_status"][0]
            mask = (status[0] == 0).astype(np.float32)
        except Exception:
            pass

    gfac = np.where(gain != 0, np.float32(1.0) / gain, np.float32(0.0))
    if mask is not None:
        gmask = (gfac * mask[np.newaxis]).astype(np.float32)
    else:
        gmask = gfac.astype(np.float32)

    if canonical_segment_ids is not None:
        segment_ids = tuple(int(segment_id) for segment_id in canonical_segment_ids)
        if len(set(segment_ids)) != len(segment_ids):
            raise ValueError("canonical_segment_ids contains duplicates")
        if segment_ids and (
            min(segment_ids) < 0 or max(segment_ids) >= peds.shape[1]
        ):
            raise ValueError(
                "canonical segment IDs are outside the calibration-constant "
                f"axis: ids={segment_ids}, n_segments={peds.shape[1]}"
            )
        peds = peds[:, segment_ids]
        gmask = gmask[:, segment_ids]

    return (
        np.ascontiguousarray(peds.ravel()),
        np.ascontiguousarray(gmask.ravel()),
    )


def prep_calib_constants(det, canonical_segment_ids=None):
    """Prepare canonical calibration constants and transfer them to the GPU."""
    cp = _cupy()
    peds_flat, gmask_flat = _compute_calib_constants_cpu(
        det, canonical_segment_ids=canonical_segment_ids
    )
    return cp.asarray(peds_flat), cp.asarray(gmask_flat)


def prepare_geometry(det, canonical_segment_ids):
    """Prepare GPU image-scatter metadata from a psana detector."""
    try:
        ix_all, iy_all = det.raw._pixel_coord_indexes(all_segs=True)
    except Exception as exc:
        import warnings

        warnings.warn(
            "GPUDetector.setup_geometry: could not load pixel coordinate "
            f'indices ({exc}). evt.gpu.get("*.image") will fail.'
        )
        return None
    return prepare_geometry_from_arrays(
        ix_all,
        iy_all,
        canonical_segment_ids,
        source="setup_geometry",
    )


def prepare_geometry_from_arrays(
    ix_all,
    iy_all,
    canonical_segment_ids,
    source="setup_geometry_from_arrays",
):
    """Prepare GPU image-scatter metadata from coordinate-index arrays."""
    cp = _cupy()
    try:
        segment_ids = list(canonical_segment_ids)
        ix = ix_all[segment_ids].astype(np.int64)
        iy = iy_all[segment_ids].astype(np.int64)
    except IndexError as exc:
        import warnings

        warnings.warn(
            f"GPUDetector.{source}: segment index out of range "
            f'({exc}). evt.gpu.get("*.image") will fail.'
        )
        return None

    image_shape = (int(ix.max()) + 1, int(iy.max()) + 1)
    try:
        return (
            cp.asarray(np.ascontiguousarray(ix.ravel())),
            cp.asarray(np.ascontiguousarray(iy.ravel())),
            image_shape,
        )
    except Exception as exc:
        import warnings

        warnings.warn(
            f"GPUDetector.{source}: could not transfer scatter indices to GPU "
            f'({exc}). evt.gpu.get("*.image") will fail.'
        )
        return None


def assemble_image(calib_gpu, scatter_ix, scatter_iy, image_shape, stream=None):
    """Scatter calibrated detector segments into a 2-D GPU image."""
    if scatter_ix is None or image_shape is None:
        return None

    cp = _cupy()
    context = stream if stream is not None else cp.cuda.Stream.null
    try:
        with context:
            image_gpu = cp.zeros(image_shape, dtype=cp.float32)
            image_gpu[scatter_ix, scatter_iy] = calib_gpu.ravel()
        return image_gpu
    except Exception:
        return None


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
    header = header_path.read_text()
    return header + f"""

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
