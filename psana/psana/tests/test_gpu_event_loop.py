"""
test_gpu_event_loop.py — pytest test for the GPU/CPU BD rank event loop.

Implements the validation from psana2 GPU BD prototype design document §9:

    for ctx in run.events():   # DataSource(gpu_det='jungfrau')
        calib = ctx.get('calib').on_gpu   # GpuEventContext, stays on GPU
        assert calib.dtype == float32
        assert calib.shape = (n_segs, 512, 1024)
        assert no NaN values

Uses the public DataSource(gpu_det=) integration so the test exercises the
complete path:

    SMD Configure dgrams → auto-discover GPU streams
    EventBuilder GPU split → GPUBAT1
    KvikIO GDS reads (issue_batch / wait_batch double-buffering)
    EventPool (N batches in flight, non-blocking CUDA streams)
    GPUDetector.process_batch() → fused_calib_gpu kernel
    GpuEventContext.get('calib') → GPUResult

Requires a CUDA GPU and real MFX data at:
    /sdf/data/lcls/ds/prj/public01/xtc/smalldata/mfx100852324-r0077*

Marked @pytest.mark.slow — excluded from normal CI runs.

Run
---
    # On a GPU node:
    pytest psana/psana/tests/test_gpu_event_loop.py -m slow
"""

import glob
import os

import numpy as np
import pytest

_SMD_GLOB = os.environ.get('PSANA_GPU_TEST_SMD_GLOB', '')
_DET_NAME   = 'jungfrau'
_BATCH_SIZE = 5
_MAX_EVENTS = 10


def _gpu_available() -> bool:
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _mfx_data_available() -> bool:
    return bool(glob.glob(_SMD_GLOB))


requires_gpu = pytest.mark.skipif(
    not _gpu_available(), reason='no CUDA device available'
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
@requires_gpu
def test_gpu_event_loop_pdf_section9():
    """PDF §9 event loop: GPU calibration from real MFX Jungfrau data.

    Validates the complete GPU BD pipeline end-to-end using the public
    DataSource(gpu_det=) API:

        ds = DataSource(files=smd_files, gpu_det='jungfrau')
        for run in ds.runs():
            for ctx in run.events():
                calib = ctx.get('calib').on_gpu

    Checks per-event:
      - dtype is float32
      - shape is (n_segs, 512, 1024)  where n_segs >= 1
      - no NaN values

    Pixel-exact comparison against CPU det.raw.calib() is a future
    refinement (requires verified calibconst segment mapping).
    """
    if not _mfx_data_available():
        pytest.skip(f'test data not found (set PSANA_GPU_TEST_SMD_GLOB): {_SMD_GLOB}')

    import cupy as cp
    from psana import DataSource

    smd_files = sorted(glob.glob(_SMD_GLOB))
    smd_files = list(dict.fromkeys(smd_files))

    ds = DataSource(
        files=smd_files,
        gpu_det=_DET_NAME,
        batch_size=_BATCH_SIZE,
        max_events=_MAX_EVENTS,
    )

    n_events = 0
    for run in ds.runs():
        for ctx in run.events():
            calib = ctx.get('calib').on_gpu    # GpuEventContext → GPUResult → cp.ndarray

            assert calib.dtype == cp.float32, (
                f'evt={n_events}: expected float32, got {calib.dtype}'
            )
            assert calib.ndim == 3, (
                f'evt={n_events}: expected 3-D (n_segs, nrows, ncols), '
                f'got shape {calib.shape}'
            )
            n_segs, nrows, ncols = calib.shape
            assert n_segs >= 1, (
                f'evt={n_events}: n_segs must be >= 1, got {n_segs}'
            )
            assert (nrows, ncols) == (512, 1024), (
                f'evt={n_events}: expected (512, 1024) panel size, '
                f'got {(nrows, ncols)}'
            )
            assert not bool(cp.any(cp.isnan(calib))), (
                f'evt={n_events}: NaN values in GPU calib '
                f'(shape={calib.shape})'
            )

            n_events += 1

    assert n_events > 0, (
        'No events were produced.  '
        'Check that the DataSource GPU routing is working.'
    )


@pytest.mark.slow
@requires_gpu
def test_gpu_event_loop_raw_and_image():
    """ctx.get('raw') and ctx.get('image') are populated alongside 'calib'.

    - ctx.get('raw')   → uint16 raw ADC values, same shape as calib
    - ctx.get('image') → float32 assembled 2-D image (if geometry loaded)
    """
    if not _mfx_data_available():
        pytest.skip(f'test data not found (set PSANA_GPU_TEST_SMD_GLOB): {_SMD_GLOB}')

    import cupy as cp
    from psana import DataSource

    smd_files = sorted(glob.glob(_SMD_GLOB))
    smd_files = list(dict.fromkeys(smd_files))

    ds = DataSource(
        files=smd_files,
        gpu_det=_DET_NAME,
        batch_size=_BATCH_SIZE,
        max_events=3,
    )

    n_events = 0
    for run in ds.runs():
        for ctx in run.events():
            # Raw ADC values
            raw = ctx.get('raw').on_gpu
            assert raw.dtype == cp.uint16, (
                f'evt={n_events}: raw should be uint16, got {raw.dtype}'
            )
            assert raw.shape == ctx.get('calib').on_gpu.shape, (
                f'evt={n_events}: raw shape {raw.shape} != '
                f'calib shape {ctx.get("calib").on_gpu.shape}'
            )

            # All raw gain-bit patterns should be representable (0-3 in top 2 bits).
            gbits = cp.asnumpy(raw >> 14)
            assert set(gbits.ravel()).issubset({0, 1, 2, 3}), (
                f'evt={n_events}: unexpected gain bit values: '
                f'{set(gbits.ravel()) - {0,1,2,3}}'
            )

            # Assembled 2-D image (optional — geometry may not be loaded for all runs)
            try:
                img = ctx.get('image').on_gpu
                assert img.dtype == cp.float32
                assert img.ndim == 2
                assert img.shape[0] > 0 and img.shape[1] > 0
            except KeyError:
                pass   # geometry not available — skip silently

            n_events += 1

    assert n_events > 0


@pytest.mark.slow
@requires_gpu
def test_gpu_event_loop_batch_size_5():
    """Pipeline with batch_size=5 produces same event count as batch_size=1.

    Tests that the EventPool correctly flushes all batches regardless of size
    and that the batch_event_index logic is correct for batch_size > 1.
    """
    if not _mfx_data_available():
        pytest.skip(f'test data not found (set PSANA_GPU_TEST_SMD_GLOB): {_SMD_GLOB}')

    from psana import DataSource

    smd_files = sorted(glob.glob(_SMD_GLOB))
    smd_files = list(dict.fromkeys(smd_files))

    def count_events(batch_size):
        ds = DataSource(
            files=smd_files,
            gpu_det=_DET_NAME,
            batch_size=batch_size,
            max_events=_MAX_EVENTS,
        )
        n = 0
        for run in ds.runs():
            for _ in run.events():
                n += 1
        return n

    n_bs1 = count_events(1)
    n_bs5 = count_events(5)

    assert n_bs1 == n_bs5, (
        f'batch_size=1 produced {n_bs1} events but '
        f'batch_size=5 produced {n_bs5}'
    )
    assert n_bs1 > 0, 'no events processed'


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'slow or not slow'])
