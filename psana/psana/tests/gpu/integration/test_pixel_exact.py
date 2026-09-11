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


def _assert_result_is_slot_backed(run, arr, result_type="calib"):
    """The integrated result must remain a view of a budgeted slot buffer.

    Only meaningful while the event still owns its device slot.  With
    gpu_d2h_chunk_size > 0 the manager may free the slot before yielding, in
    which case there is deliberately no device array left to check — see
    _result_still_on_device().
    """
    manager = getattr(run._evt_iter, "gpu_manager", run._evt_iter)
    gpu_detector = manager.gpu_detectors[_DET_NAME][1]
    result_start = int(arr.data.ptr)
    result_end = result_start + int(arr.nbytes)

    slot_buffers = getattr(gpu_detector, f"_{result_type}_slot_bufs")
    for slot_buf in slot_buffers:
        if slot_buf is None:
            continue
        slot_start = int(slot_buf.data.ptr)
        slot_end = slot_start + int(slot_buf.nbytes)
        if slot_start <= result_start and result_end <= slot_end:
            return

    pytest.fail(
        f"GPU {result_type} result is not backed by a budgeted slot buffer"
    )


def _result_still_on_device(result):
    """True when this result kept its device slot through delivery.

    False in the automatic-D2H early-release case, where _arr is cleared and
    the host copy is the only valid view.
    """
    return getattr(result, "_arr", None) is not None


@pytest.mark.gpu
@requires_gpu
def test_locator_raw_gather_precedes_ordinary_calibration():
    """A parser locator, rather than an inferred stride, addresses raw data."""
    import cupy as cp

    from psana.gpu.gpu_calib import fused_calib_gpu
    from psana.gpu.gpu_detector import _gather_locator_field_gpu
    from psana.gpu.gpudgram.batch import (
        LOC_NBYTES,
        LOC_NCOLS,
        LOC_OFFSET,
        LOC_RANK,
        LOC_STATUS,
        LOC_TYPE,
    )
    from psana.gpu.gpudgram.config import GpuFieldHandle
    from psana.gpu.gpudgram.parser import STATUS_FOUND

    src = cp.asarray([90, 91, 1, 2, 3], dtype=cp.uint16).view(cp.uint8)
    handle = GpuFieldHandle(0, 10, 0, 0, 0, 1, 2, 1, 0)
    locators = cp.zeros((1, LOC_NCOLS), dtype=cp.uint64)
    locators[0, LOC_TYPE] = handle.type
    locators[0, LOC_RANK] = handle.rank
    locators[0, LOC_OFFSET] = 2 * 2
    locators[0, LOC_NBYTES] = 3 * 2
    locators[0, LOC_STATUS] = STATUS_FOUND
    raw = cp.full((3, 1, 3), 0, dtype=cp.uint16)
    present = cp.zeros(3, dtype=cp.uint8)

    result = _gather_locator_field_gpu(
        src,
        locators,
        dgram_index=0,
        handle=handle,
        output_row=2,
        pixels_per_segment=3,
        out=raw,
        present=present,
    )
    peds = cp.zeros(3 * raw.size, dtype=cp.float32)
    gmask = cp.ones(3 * raw.size, dtype=cp.float32)
    calib = fused_calib_gpu(raw, peds, gmask)
    cp.cuda.Stream.null.synchronize()

    assert result is raw
    np.testing.assert_array_equal(
        cp.asnumpy(raw[:, 0]),
        [[0, 0, 0], [0, 0, 0], [1, 2, 3]],
    )
    np.testing.assert_array_equal(cp.asnumpy(present), [0, 0, 1])
    np.testing.assert_array_equal(cp.asnumpy(calib), cp.asnumpy(raw))


@pytest.mark.gpu
@requires_gpu
def test_locator_passthrough_copy_writes_canonical_row():
    """A float32 field locator feeds the passthrough destination directly."""
    import cupy as cp

    from psana.gpu.gpu_detector import _gather_locator_field_gpu
    from psana.gpu.gpudgram.batch import (
        LOC_NBYTES,
        LOC_NCOLS,
        LOC_OFFSET,
        LOC_RANK,
        LOC_STATUS,
        LOC_TYPE,
    )
    from psana.gpu.gpudgram.config import GpuFieldHandle
    from psana.gpu.gpudgram.parser import STATUS_FOUND

    src = cp.asarray([99, 1, 2, 3], dtype=cp.float32).view(cp.uint8)
    handle = GpuFieldHandle(0, 10, 0, 0, 0, 2, 4, 1, 0)
    locators = cp.zeros((1, LOC_NCOLS), dtype=cp.uint64)
    locators[0, LOC_TYPE] = handle.type
    locators[0, LOC_RANK] = handle.rank
    locators[0, LOC_OFFSET] = 4
    locators[0, LOC_NBYTES] = 3 * 4
    locators[0, LOC_STATUS] = STATUS_FOUND
    out = cp.full((3, 3), -1, dtype=cp.float32)
    present = cp.zeros(3, dtype=cp.uint8)

    result = _gather_locator_field_gpu(
        src,
        locators,
        dgram_index=0,
        handle=handle,
        output_row=2,
        pixels_per_segment=3,
        out=out,
        present=present,
    )
    cp.cuda.Stream.null.synchronize()

    assert result is out
    np.testing.assert_array_equal(cp.asnumpy(out[2]), [1, 2, 3])
    np.testing.assert_array_equal(cp.asnumpy(out[0]), [-1, -1, -1])
    np.testing.assert_array_equal(cp.asnumpy(out[1]), [-1, -1, -1])


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.data
@requires_gpu
@requires_data
@pytest.mark.parametrize(
    "batch_size,pool_depth,d2h_chunk_size",
    [
        pytest.param(1, 1, 0, id="single-event"),
        pytest.param(5, 2, 0, id="batched-slot-reuse-partial-tail"),
        # gpu_d2h_chunk_size > 0 activates _D2hPipeline: results are copied to
        # pinned host memory on a separate stream. Parsed field access now
        # keeps the input slot through the event yield window. Exercising the
        # D2H here is the only
        # check of that path against a real CUDA stream — the unit tests fake
        # cupy with a synchronous memmove, which cannot detect a missing
        # synchronization because the data has already landed.
        pytest.param(5, 2, 1, id="d2h-chunk-per-event"),
        pytest.param(5, 2, 3, id="d2h-chunk-spans-events"),
        # Chunk larger than the batch: exercises the partial-chunk path.
        pytest.param(5, 2, 8, id="d2h-chunk-exceeds-batch"),
    ],
)
def test_integrated_jungfrau_pixel_exact(
    cpu_reference, batch_size, pool_depth, d2h_chunk_size
):
    """Integrated GPU calibration exactly matches normal psana by timestamp."""
    from psana import DataSource

    ds = DataSource(
        exp=_EXP,
        run=_RUN,
        dir=_DIR,
        gpu_det=_DET_NAME,
        batch_size=batch_size,
        n_gpu_streams=pool_depth,
        gpu_d2h_chunk_size=d2h_chunk_size,
        max_events=_N_EVENTS,
    )
    run = next(ds.runs())

    seen = set()
    validated_xtc_batches = set()
    validated_general_field_access = False
    for evt in run.events():
        timestamp = int(evt.timestamp)
        assert timestamp not in seen, f"duplicate GPU timestamp {timestamp}"
        assert timestamp in cpu_reference, (
            f"GPU produced timestamp {timestamp} absent from CPU reference"
        )

        # Verify the canonical result is still the budgeted slot view, then
        # copy before advancing the iterator can recycle that slot.
        calib_result = evt.gpu.get("calib")
        raw_result = evt.gpu.get("raw")
        manager = getattr(run._evt_iter, "gpu_manager", run._evt_iter)
        if d2h_chunk_size == 0:
            # While the slot is exposed between the two retirement phases,
            # verify the parser decoded every routed dgram and located the
            # arrays now consumed by GPUDetector.
            slot_record = manager.event_pool._retiring
            if slot_record is None:
                # End-of-input/max-events delivery comes from flush(), whose
                # yield window retains the record in _slots without using the
                # incremental-retirement latch.
                slot_record = next(
                    (
                        record
                        for record in manager.event_pool._slots
                        if record is not None
                        and timestamp in record.gpu_results_by_ts
                    ),
                    None,
                )
            assert slot_record is not None
            xtc_batch = slot_record.xtc_batch
            assert xtc_batch is not None
            if id(xtc_batch) not in validated_xtc_batches:
                import cupy as cp

                from psana.gpu.gpudgram.batch import DGRAM_STATUS, LOC_STATUS
                from psana.gpu.gpudgram.parser import STATUS_FOUND, STATUS_OK

                dgram_status = cp.asnumpy(
                    xtc_batch.dgram_records_gpu[:, DGRAM_STATUS]
                )
                assert np.all(dgram_status == STATUS_OK)
                found = np.zeros(xtc_batch.n_dgrams, dtype=bool)
                for handle in manager.gpu_xtc_parser.field_handles:
                    locator_status = cp.asnumpy(
                        xtc_batch.locate(handle).rows_gpu[:, LOC_STATUS]
                    )
                    found |= locator_status == STATUS_FOUND
                assert np.all(found), "a routed dgram had no located detector array"
                validated_xtc_batches.add(id(xtc_batch))
        if _result_still_on_device(calib_result):
            _assert_result_is_slot_backed(run, calib_result._arr)
        else:
            # Early release only happens once every result has a host handoff,
            # so on_cpu must still work and the device view must be refused
            # rather than silently returning a recycled buffer.
            assert d2h_chunk_size > 0, (
                "device slot was released without the D2H pipeline enabled"
            )
            with pytest.raises(RuntimeError):
                calib_result.on_gpu
        if _result_still_on_device(raw_result):
            _assert_result_is_slot_backed(run, raw_result._arr, result_type="raw")
        gpu_raw = np.asarray(raw_result.on_cpu).copy()
        gpu_calib = np.asarray(calib_result.on_cpu).copy()
        if not validated_general_field_access:
            import cupy as cp

            detector = evt.gpu.detector(_DET_NAME)
            parsed_raw_result = detector.field("raw", "raw")
            parsed_raw = parsed_raw_result.on_cpu
            binding = manager.gpu_detector_bindings[_DET_NAME]
            assert parsed_raw.segment_ids == binding.canonical_segment_ids
            for row, segment in enumerate(binding.canonical_segment_ids):
                np.testing.assert_array_equal(
                    np.asarray(parsed_raw[segment]).reshape(gpu_raw[row].shape),
                    gpu_raw[row],
                )
            frame_count = detector.field(
                "raw", "frame_cnt", segment=binding.canonical_segment_ids[0]
            ).on_cpu.only()
            assert frame_count.shape == ()
            assert frame_count.dtype == np.uint64

            first_segment = binding.canonical_segment_ids[0]
            one_segment = detector.field(
                "raw", "raw", segment=first_segment
            )
            gpu_copy = one_segment.on_gpu.only()
            np.testing.assert_array_equal(
                cp.asnumpy(gpu_copy).reshape(gpu_raw[0].shape),
                gpu_raw[0],
            )
            user_stream = cp.cuda.Stream(non_blocking=True)
            with one_segment.on_gpu_view(user_stream) as views:
                with user_stream:
                    view_copy = views.only().copy()
            np.testing.assert_array_equal(
                cp.asnumpy(view_copy).reshape(gpu_raw[0].shape),
                gpu_raw[0],
            )
            validated_general_field_access = True
        np.testing.assert_array_equal(
            gpu_raw,
            cpu_reference[timestamp]["raw"],
            err_msg=f"timestamp={timestamp}: canonical raw mismatch",
        )
        _assert_pixel_exact(
            timestamp, cpu_reference[timestamp]["calib"], gpu_calib
        )
        seen.add(timestamp)

    expected = set(cpu_reference)
    assert seen == expected, (
        f"timestamp set mismatch: missing={sorted(expected - seen)} "
        f"extra={sorted(seen - expected)}"
    )
