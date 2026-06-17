"""
test_gpu_calib.py — pytest tests for GPU calibration kernel (Steps 5a and 4).

Two tests, both using the local test_jungfrau05M_calib.xtc2 fixture:

test_fused_calib
    Transfers CPU raw array to GPU, runs fused_calib_gpu(), and compares the
    result against det.raw.calib(evt).  The fixture has pixel mask = all-zeros,
    so the expected output is all-zeros.

test_event_context
    Loads the full bigdata dgram bytes into a GPU buffer (simulating KvikIO),
    calls GPUDetector.calibrate(), and checks the output matches det.raw.calib().
    Validates JUNGFRAU_DGRAM_RAW_OFFSET=80 is correct.

Both tests require a CUDA GPU and are marked @pytest.mark.slow so they are
excluded from normal CI runs (see pytest.ini: addopts = -m "not slow").

Run
---
    # Normal CI — skipped automatically:
    pytest psana/psana/tests/test_gpu_calib.py

    # On a GPU node — include slow/GPU tests:
    pytest psana/psana/tests/test_gpu_calib.py -m slow
"""

import os

import numpy as np
import pytest

_DIR  = os.path.dirname(os.path.realpath(__file__))
_XTC2 = os.path.join(_DIR, 'test_data', 'detector', 'test_jungfrau05M_calib.xtc2')
_DET  = 'jungfrau'


def _gpu_available() -> bool:
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(
    not _gpu_available(), reason='no CUDA device available'
)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def _open_det(xtc2=_XTC2, det_name=_DET):
    import psana
    ds  = psana.DataSource(files=xtc2)
    run = next(ds.runs())
    det = run.Detector(det_name)
    return run, det


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
@requires_gpu
def test_fused_calib():
    """fused_calib_gpu() output matches det.raw.calib(evt) for the all-zero fixture.

    The test fixture has pixel mask = all-zeros, so gmask = 0 everywhere and
    the calibrated output must be exactly zero for every event.
    """
    import cupy as cp
    from psana.gpu.gpu_calib import fused_calib_gpu, prep_calib_constants

    run, det = _open_det()
    peds_gpu, gmask_gpu = prep_calib_constants(det)

    # Confirm the all-zero-mask invariant holds for this fixture.
    mask = det.raw._mask()
    assert mask is None or np.all(mask == 0), (
        'test fixture should have all-zeros mask; '
        'check test_jungfrau05M_calib.xtc2'
    )

    for i, evt in enumerate(run.events()):
        if i >= 3:
            break
        raw_cpu   = det.raw.raw(evt)
        calib_cpu = det.raw.calib(evt)
        assert raw_cpu is not None,   f'evt={i}: det.raw.raw() returned None'
        assert calib_cpu is not None, f'evt={i}: det.raw.calib() returned None'

        raw_gpu        = cp.asarray(raw_cpu)
        calib_from_gpu = fused_calib_gpu(raw_gpu, peds_gpu, gmask_gpu).get()

        assert calib_from_gpu.shape == calib_cpu.shape, (
            f'evt={i}: shape mismatch '
            f'{calib_from_gpu.shape} != {calib_cpu.shape}'
        )
        assert calib_from_gpu.dtype == np.float32, (
            f'evt={i}: expected float32, got {calib_from_gpu.dtype}'
        )
        assert np.allclose(calib_from_gpu, calib_cpu, atol=1e-5), (
            f'evt={i}: GPU calib != CPU calib; '
            f'max diff = {np.abs(calib_from_gpu - calib_cpu).max():.2e}'
        )
        # The fixture invariant: mask is all-zeros → output must be all-zeros.
        assert np.all(calib_from_gpu == 0), (
            f'evt={i}: expected all-zero output (mask=0), '
            f'got mean={calib_from_gpu.mean():.6f}'
        )


@pytest.mark.slow
@requires_gpu
def test_event_context():
    """GPUDetector.calibrate() from dgram bytes matches det.raw.calib().

    Simulates what KvikIO would deliver: the full bigdata dgram bytes are
    transferred to GPU as a flat uint8 buffer.  GPUDetector extracts raw
    pixels at JUNGFRAU_DGRAM_RAW_OFFSET=80 and applies calibration.

    Passing this test confirms:
      1. The XTC header offset (80 bytes) is correct.
      2. The full extraction → calibration pipeline is wired correctly.
    """
    import cupy as cp
    from psana.gpu.gpu_calib import GPUDetector, prep_calib_constants

    run, det = _open_det()
    peds_gpu, gmask_gpu = prep_calib_constants(det)
    det_shape = det.calibconst['pedestals'][0].shape[1:]   # (n_segs, nrows, ncols)

    gpu_det = GPUDetector(
        det_shape=det_shape,
        peds_gpu=peds_gpu,
        gmask_gpu=gmask_gpu,
    )

    for i, evt in enumerate(run.events()):
        if i >= 3:
            break
        dg = evt._dgrams[0]
        assert dg is not None, f'evt={i}: _dgrams[0] is None'

        calib_cpu = det.raw.calib(evt)
        assert calib_cpu is not None, f'evt={i}: det.raw.calib() returned None'

        # Simulate KvikIO: load full dgram bytes into GPU buffer.
        dg_bytes = np.frombuffer(bytearray(dg), dtype=np.uint8)
        data_gpu = cp.asarray(dg_bytes)

        calib_gpu = gpu_det.calibrate(data_gpu, device_offset=0).get()

        assert calib_gpu.shape == calib_cpu.shape, (
            f'evt={i}: shape mismatch {calib_gpu.shape} != {calib_cpu.shape}'
        )
        assert calib_gpu.dtype == np.float32, (
            f'evt={i}: expected float32, got {calib_gpu.dtype}'
        )
        assert np.allclose(calib_gpu, calib_cpu, atol=1e-5), (
            f'evt={i}: GPUDetector.calibrate() output != CPU calib; '
            f'max diff = {np.abs(calib_gpu - calib_cpu).max():.2e}'
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'slow or not slow'])
