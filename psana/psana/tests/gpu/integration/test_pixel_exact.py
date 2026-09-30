"""Real-data pixel checks for input-only GPU processing and CPU hybrid calibration.

Historical calibration kernels are explicit test references, outside the runtime.
Callback numerical acceptance will be added with public task support.
"""
import glob
import os

import numpy as np
import pytest


_EXP = os.environ.get("PSANA_GPU_TEST_EXP", "mfx100848724")
_RUN = int(os.environ.get("PSANA_GPU_TEST_RUN", "51"))
_DIR = os.environ.get(
    "PSANA_GPU_TEST_DIR",
    "/sdf/data/lcls/ds/prj/public01/xtc",
)
_DET_NAME = "jungfrau"
_N_EVENTS = 13


def _gpu_available():
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _data_available():
    prefix = f"{_EXP}-r{_RUN:04d}"
    smd_files = glob.glob(os.path.join(_DIR, "smalldata", f"{prefix}*.smd.xtc2"))
    xtc_files = glob.glob(os.path.join(_DIR, f"{prefix}*.xtc2"))
    return bool(smd_files and xtc_files)


requires_gpu = pytest.mark.skipif(
    not _gpu_available(),
    reason="no CUDA device available",
)
requires_data = pytest.mark.skipif(
    not _data_available(),
    reason=f"test data not found: exp={_EXP} run={_RUN} dir={_DIR}",
)


@pytest.fixture(scope="module")
def cpu_reference():
    """Return timestamp-keyed CPU calibration arrays for the public run."""
    from psana import DataSource

    ds = DataSource(
        exp=_EXP,
        run=_RUN,
        dir=_DIR,
        max_events=_N_EVENTS,
    )
    run = next(ds.runs())
    det = run.Detector(_DET_NAME)

    reference = {}
    gain_modes = set()
    has_nonzero_calib = False

    for evt in run.events():
        raw = det.raw.raw(evt)
        calib = det.raw.calib(evt, cmpars=None)
        if raw is None or calib is None:
            continue

        timestamp = int(evt.timestamp)
        assert timestamp not in reference, f"duplicate CPU timestamp {timestamp}"

        # Canonicalize to the GPU result dtype and detach from psana's event
        # buffers before the iterator advances.
        calib = np.asarray(calib, dtype=np.float32).copy()
        reference[timestamp] = {
            "raw": np.asarray(raw, dtype=np.uint16).copy(),
            "calib": calib,
        }
        gain_modes.update(int(value) for value in np.unique(raw >> 14))
        has_nonzero_calib = has_nonzero_calib or bool(np.any(calib != 0))

    assert len(reference) == _N_EVENTS, (
        f"CPU reference produced {len(reference)} usable events; "
        f"expected {_N_EVENTS}"
    )
    assert has_nonzero_calib, "reference calibration is entirely zero"
    assert len(gain_modes) >= 2, (
        f"reference data exercise only gain-bit values {sorted(gain_modes)}"
    )
    return reference


@pytest.mark.gpu
@requires_gpu
def test_explicit_calibration_reference_gain_modes():
    import cupy as cp
    from psana.tests.gpu.calibration_reference import fused_calib_gpu
    raw = cp.asarray([10, 0x400A, 0xC00A], dtype=cp.uint16)
    peds = cp.repeat(cp.asarray([1, 2, 3], dtype=cp.float32), 3)
    gain = cp.repeat(cp.asarray([2, 3, 4], dtype=cp.float32), 3)
    result = fused_calib_gpu(raw, peds, gain)
    np.testing.assert_array_equal(result.get(), [18, 24, 28])


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.data
@requires_gpu
@requires_data
@pytest.mark.parametrize('detector_kw,batch_size,pool_depth,bulk_read', [
    pytest.param('gpu_det', 1, 1, False, id='single-event'),
    pytest.param('gpu_det', 5, 2, False, id='slot-reuse-partial-tail'),
    pytest.param('hybrid_det', 5, 2, False, id='hybrid'),
    pytest.param('gpu_det', 5, 2, True, id='bulk-exclusive'),
    pytest.param('hybrid_det', 5, 2, True, id='bulk-hybrid'),
])
def test_integrated_jungfrau_inputs_pixel_exact(
        cpu_reference, detector_kw, batch_size, pool_depth, bulk_read):
    from psana import DataSource
    ds = DataSource(exp=_EXP, run=_RUN, dir=_DIR, max_events=_N_EVENTS,
                    batch_size=batch_size, n_gpu_streams=pool_depth,
                    gpu_bulk_read=bulk_read,
                    skip_calib_load=[_DET_NAME] if detector_kw == 'gpu_det' else [],
                    **{detector_kw: _DET_NAME})
    run = next(ds.runs())
    det = run.Detector(_DET_NAME)
    if detector_kw == 'gpu_det':
        assert not det.calibconst
    seen = set()
    saved = []
    for evt in run.events():
        timestamp = int(evt.timestamp)
        assert timestamp not in seen and timestamp in cpu_reference
        assert not evt.gpu._gpu_results
        for name in ('calib', 'raw', 'image', 'jungfrau.calib'):
            with pytest.raises(KeyError):
                evt.gpu.get(name)
        fields = evt.gpu.detector(_DET_NAME)
        result = fields.field('raw', 'raw')
        values = result.on_cpu
        actual = np.stack([values[s].reshape(512, 1024) for s in values.segment_ids])
        np.testing.assert_array_equal(actual, cpu_reference[timestamp]['raw'])
        if not seen:
            counter = fields.field('raw', 'frame_cnt', segment=values.segment_ids[0]).on_cpu.only()
            assert counter.shape == () and counter.dtype == np.uint64
            import cupy as cp
            stream = cp.cuda.Stream(non_blocking=True)
            with result.on_gpu_view(stream) as views:
                with stream:
                    copied = views[values.segment_ids[0]].copy()
            stream.synchronize()
            np.testing.assert_array_equal(copied.get().reshape(512, 1024), actual[0])
        if detector_kw == 'hybrid_det':
            np.testing.assert_array_equal(det.raw.raw(evt), actual)
            np.testing.assert_array_equal(det.raw.calib(evt, cmpars=None),
                                          cpu_reference[timestamp]['calib'])
        saved.append((result, actual))
        seen.add(timestamp)
    assert seen == set(cpu_reference)
    # Host copies remain independent after all input slots retire.
    for result, expected in saved:
        values = result.on_cpu
        np.testing.assert_array_equal(np.stack([values[s].reshape(512, 1024)
                                               for s in values.segment_ids]), expected)
