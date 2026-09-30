"""Pixel-exact validation of the integrated psana2 GPU event path.

The existing kernel tests start from ``det.raw.raw(evt)`` and therefore do
not exercise EventBuilder GPU splitting, GPUBAT1 descriptors, KvikIO reads,
raw-payload extraction, segment ordering, EventPool slot reuse, or timestamp
joining. This test compares the final result from
``DataSource(gpu_det="jungfrau")`` with the normal psana CPU calibration for
the same event timestamps.

The default dataset is public MFX Lysozyme Jungfrau ``mfx100848724`` run 51.
Override it with ``PSANA_GPU_TEST_EXP``, ``PSANA_GPU_TEST_RUN``, and
``PSANA_GPU_TEST_DIR``.
Common-mode correction is disabled explicitly because the current GPU kernel
implements pedestal, gain, pixel-offset, and mask calibration only.

The public ``mfx100852324`` runs 77 and 78 are intentionally not the defaults:
their effective Jungfrau masks are all zero, so both CPU and GPU calibration
produce trivial all-zero arrays even though the raw data are nonzero.
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
        reference[timestamp] = calib
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


def _assert_pixel_exact(timestamp, cpu_calib, gpu_calib):
    assert gpu_calib.dtype == np.float32, (
        f"timestamp={timestamp}: expected GPU float32, got {gpu_calib.dtype}"
    )
    assert gpu_calib.shape == cpu_calib.shape, (
        f"timestamp={timestamp}: shape mismatch "
        f"GPU={gpu_calib.shape} CPU={cpu_calib.shape}"
    )

    # Treat NaNs at the same pixel as equal, but otherwise require exact
    # float32 equality. No numerical tolerance is intentionally applied.
    equal = np.equal(cpu_calib, gpu_calib)
    equal |= np.isnan(cpu_calib) & np.isnan(gpu_calib)
    if bool(np.all(equal)):
        return

    mismatch = ~equal
    first_flat = int(np.argmax(mismatch))
    first_index = tuple(
        int(index) for index in np.unravel_index(first_flat, cpu_calib.shape)
    )
    n_mismatch = int(np.count_nonzero(mismatch))
    cpu_value = cpu_calib[first_index]
    gpu_value = gpu_calib[first_index]
    max_abs_diff = float(
        np.nanmax(np.abs(cpu_calib.astype(np.float64) - gpu_calib))
    )
    pytest.fail(
        f"timestamp={timestamp}: {n_mismatch} pixel mismatches; "
        f"first_index={first_index} CPU={cpu_value!r} GPU={gpu_value!r}; "
        f"max_abs_diff={max_abs_diff:.9g}"
    )


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.data
@requires_gpu
@requires_data
@pytest.mark.parametrize(
    "batch_size,pool_depth",
    [
        pytest.param(1, 1, id="single-event"),
        pytest.param(5, 2, id="batched-slot-reuse-partial-tail"),
    ],
)
def test_integrated_jungfrau_pixel_exact(cpu_reference, batch_size, pool_depth):
    """Integrated GPU calibration exactly matches normal psana by timestamp."""
    from psana import DataSource

    ds = DataSource(
        exp=_EXP,
        run=_RUN,
        dir=_DIR,
        gpu_det=_DET_NAME,
        batch_size=batch_size,
        n_gpu_streams=pool_depth,
        max_events=_N_EVENTS,
    )
    run = next(ds.runs())

    seen = set()
    for ctx in run.events():
        timestamp = int(ctx.timestamp)
        assert timestamp not in seen, f"duplicate GPU timestamp {timestamp}"
        assert timestamp in cpu_reference, (
            f"GPU produced timestamp {timestamp} absent from CPU reference"
        )

        # Copy immediately, before advancing the iterator can recycle the
        # EventPool slot that owns this result.
        gpu_calib = np.asarray(ctx.get("calib").on_cpu).copy()
        _assert_pixel_exact(timestamp, cpu_reference[timestamp], gpu_calib)
        seen.add(timestamp)

    expected = set(cpu_reference)
    assert seen == expected, (
        f"timestamp set mismatch: missing={sorted(expected - seen)} "
        f"extra={sorted(seen - expected)}"
    )
