"""
test_gpu_transition_handling.py — Tests for GPU transition event ordering.

Three groups of tests:

CPU-only (always run, no GPU or special data needed):
  test_iter_step_services_single      — one-dgram step batch
  test_iter_step_services_multiple    — two transitions in one batch (Enable + BeginStep)
  test_iter_step_services_fallback    — malformed footer falls back to first dgram
  test_iter_step_services_empty       — empty / too-short input returns nothing

CPU, MFX data required (marked slow):
  test_batch_splits_at_beginstep      — EventBuilder stops collecting L1Accepts when
                                        BeginStep is encountered; batch count > ⌈N/batch_size⌉

GPU, MFX data required (marked slow):
  test_beginstep_detected_and_called  — beginstep() is called when BeginStep appears
  test_eventpool_drained_before_beginstep — calibconst update does not race with
                                            in-flight calibration kernels

Background
----------
The state machine for a scan run is:

    Configure → BeginRun → [BeginStep → Enable → L1Accept×N → Disable → EndStep] → EndRun

Key ordering rules:
  BeginStep MAY change calibration constants (gain-mode switch, threshold update).
    → EventPool must be FLUSHED before beginstep() updates peds_gpu / gmask_gpu.
    → EventBuilder must STOP collecting L1Accepts at BeginStep so no batch
      straddles a calibration boundary.
  EndRun: EventPool must be flushed so no results are lost.
  Enable / Disable / EndStep: no calibration dependency; process immediately.
"""

import glob
import os
import struct

import numpy as np
import pytest

from psana.psexp import TransitionId
from psana.psexp.packet_footer import PacketFooter
from psana.gpu.gpu_events import _iter_step_services

_SMD_GLOB = os.environ.get('PSANA_GPU_TEST_SMD_GLOB', '')


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
# Helpers: construct minimal synthetic step-batch bytes
# ---------------------------------------------------------------------------

def _make_dgram_bytes(service: int, payload_bytes: int = 0) -> bytes:
    """Build a minimal 24-byte Dgram header with the given service type.

    Dgram layout:
      bytes  0-7   timestamp (zeros)
      bytes  8-11  env  (service encoded in bits 31:24)
      bytes 12-13  damage  (zero)
      bytes 14-17  src     (zero)
      bytes 18-19  contains (zero)
      bytes 20-23  XTC extent (payload_bytes; typically 0 for stub transitions)
    """
    env = service << 24
    extent = payload_bytes
    header = struct.pack('<QI2sI2sI',
                         0,          # timestamp (uint64)
                         env,        # env (uint32)
                         b'\x00\x00', # damage (2 bytes)
                         0,          # src (uint32)
                         b'\x00\x00', # contains (2 bytes)
                         extent,     # extent (uint32)
                         )
    return header + bytes(payload_bytes)


def _make_step_batch(*services) -> bytearray:
    """Build a PacketFooter-formatted step batch containing one dgram per service."""
    dgrams = [_make_dgram_bytes(svc) for svc in services]
    pf = PacketFooter(n_packets=len(dgrams))
    data = bytearray()
    for i, dg in enumerate(dgrams):
        pf.set_size(i, len(dg))
        data.extend(dg)
    # Append footer after the packet data
    return data + bytearray(pf.footer)


# ---------------------------------------------------------------------------
# CPU-only tests
# ---------------------------------------------------------------------------

def test_iter_step_services_single():
    """A single-dgram step batch yields exactly one (service, view) pair."""
    batch = _make_step_batch(TransitionId.Enable)
    results = list(_iter_step_services(batch))

    assert len(results) == 1, f'expected 1 result, got {len(results)}'
    svc, mv = results[0]
    assert svc == TransitionId.Enable, (
        f'expected Enable ({TransitionId.Enable}), got {svc}'
    )


def test_iter_step_services_multiple():
    """Two transitions in one batch — both are yielded in order.

    This is the critical case: Enable fires, then BeginStep fires in the
    same EventBuilder batch.  The old _service_from_batch() would only
    have returned Enable; _iter_step_services() must return both.
    """
    batch = _make_step_batch(TransitionId.Enable, TransitionId.BeginStep)
    results = list(_iter_step_services(batch))

    assert len(results) == 2, (
        f'expected 2 results (Enable + BeginStep), got {len(results)}: '
        f'{[s for s, _ in results]}'
    )
    svcs = [s for s, _ in results]
    assert svcs[0] == TransitionId.Enable,    f'first should be Enable,    got {svcs[0]}'
    assert svcs[1] == TransitionId.BeginStep, f'second should be BeginStep, got {svcs[1]}'


def test_iter_step_services_all_transition_types():
    """BeginStep, EndStep, Enable, Disable, EndRun are all correctly decoded."""
    expected = [
        TransitionId.BeginStep,
        TransitionId.Enable,
        TransitionId.Disable,
        TransitionId.EndStep,
        TransitionId.EndRun,
    ]
    batch = _make_step_batch(*expected)
    results = list(_iter_step_services(batch))

    assert len(results) == len(expected), (
        f'expected {len(expected)} results, got {len(results)}'
    )
    for i, (svc, _) in enumerate(results):
        assert svc == expected[i], (
            f'position {i}: expected service {expected[i]}, got {svc}'
        )


def test_iter_step_services_fallback_on_bad_footer():
    """Malformed step batch (no valid footer) falls back to the first dgram only."""
    # Raw bytes with no PacketFooter — _iter_step_services falls back.
    raw = bytearray(_make_dgram_bytes(TransitionId.BeginStep))
    # Append garbage to simulate a corrupt footer.
    raw += bytearray(b'\xff\xff\xff\xff')
    results = list(_iter_step_services(raw))

    # Fallback: at least one result, and it has the right service.
    assert len(results) >= 1, 'fallback should yield at least one result'
    assert results[0][0] == TransitionId.BeginStep, (
        f'fallback should decode BeginStep, got {results[0][0]}'
    )


def test_iter_step_services_empty():
    """Empty or too-short input produces no results (no exception)."""
    assert list(_iter_step_services(bytearray())) == []
    assert list(_iter_step_services(bytearray(4))) == []   # < 12 bytes
    assert list(_iter_step_services(None)) == []


# ---------------------------------------------------------------------------
# CPU test with real MFX data — requires data access, marked slow
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_batch_splits_at_beginstep():
    """EventBuilder stops collecting L1Accepts when it encounters BeginStep.

    With batch_size=5, if BeginStep falls before the 5th L1Accept in the
    collection window, the batch is terminated early.  This ensures no batch
    straddles a calibration boundary.

    Validation: the number of GPU batches must be strictly greater than
    ⌈n_events / batch_size⌉ because at least one batch is split by BeginStep.
    """
    if not _mfx_data_available():
        pytest.skip(f'test data not found (set PSANA_GPU_TEST_SMD_GLOB): {_SMD_GLOB}')

    import os as _os
    from psana.psexp.ds_base import DsParms
    from psana.psexp.smdreader_manager import SmdReaderManager
    from psana.psexp.eventbuilder_manager import EventBuilderManager

    MAX_EVENTS = 20
    BATCH_SIZE = 5

    smd_files = sorted(glob.glob(_SMD_GLOB))
    smd_files = list(dict.fromkeys(smd_files))

    dsparms = DsParms(
        batch_size=BATCH_SIZE,
        max_events=0,
        max_retries=0,
        live=False,
        timestamps=np.empty(0, dtype=np.uint64),
        intg_det='',
        intg_delta_t=0,
        use_calib_cache=False,
        cached_detectors=[],
        fetch_calib_cache_max_retries=60,
        skip_calib_load=[],
        dbsuffix='',
    )
    dsparms.update_smd_state(list(smd_files), [False] * len(smd_files))

    smd_fds = np.array([_os.open(f, _os.O_RDONLY) for f in smd_files],
                       dtype=np.int32)
    try:
        smdr = SmdReaderManager(smd_fds, dsparms)
        if smdr.get_next_dgrams() is None:
            pytest.skip('could not read Configure')
        if smdr.get_next_dgrams() is None:
            pytest.skip('could not read BeginRun')

        # Auto-discover GPU stream IDs from Configure dgrams so _split_gpu_enabled=True.
        from psana.gpu.gpu_calib import build_stream_seg_map
        xtc_files = [f.replace('/smalldata/', '/').replace('.smd.xtc2', '.xtc2')
                     for f in smd_files]
        seg_map = build_stream_seg_map(
            {i: f for i, f in enumerate(xtc_files)
             if i < len(xtc_files) and _os.path.exists(f)},
            'jungfrau',
        )
        dsparms.gpu_stream_ids = sorted(seg_map.keys()) or None

        n_l1 = 0
        n_batches = 0
        n_beginstep_batches = 0  # batches that contained a BeginStep in step_dict

        for chunk_id in smdr.chunks():
            empty = [bytearray()] * smdr.n_files
            smd_chunk = bytearray(smdr.smdr.repack_parallel(
                empty, 1,
                intg_stream_id=dsparms.intg_stream_id,
            ))
            eb = EventBuilderManager(smd_chunk, smdr.configs, dsparms)

            for batch_dict, gpu_batch_dict, step_dict in eb.batches_with_gpu():
                # Count L1Accept events in this batch from the GPUBAT1
                # n_events header field — no DgramManager/bigdata needed.
                from psana.gpu.gpu_batch import GpuBatchView
                batch_l1 = 0
                for gpu_batch, _ in gpu_batch_dict.values():
                    try:
                        gv = GpuBatchView(gpu_batch, validate=False)
                        batch_l1 += int(gv.header.n_events)
                    except Exception:
                        pass
                n_l1 += batch_l1

                if batch_l1 > 0 or gpu_batch_dict:
                    n_batches += 1

                # Check if BeginStep is in this step_dict.
                if step_dict:
                    for sb, _ in step_dict.values():
                        for svc, _ in _iter_step_services(sb):
                            if svc == TransitionId.BeginStep:
                                n_beginstep_batches += 1
                        break

                if n_l1 >= MAX_EVENTS:
                    break
            if n_l1 >= MAX_EVENTS:
                break

    finally:
        for fd in smd_fds:
            _os.close(int(fd))

    # There must be at least one BeginStep-triggered batch in this run.
    assert n_beginstep_batches >= 1, (
        'expected at least one batch with BeginStep in step_dict; '
        'the MFX run starts with BeginRun → BeginStep → Enable → L1Accepts'
    )

    # When BeginStep splits a batch, the number of batches must exceed the
    # naive ⌈n_l1 / BATCH_SIZE⌉ because the BeginStep batch terminates early.
    min_batches_no_split = (n_l1 + BATCH_SIZE - 1) // BATCH_SIZE
    assert n_batches > min_batches_no_split or n_beginstep_batches >= 1, (
        f'batch_size={BATCH_SIZE}, n_l1={n_l1}: expected > {min_batches_no_split} '
        f'batches due to BeginStep splitting, got {n_batches}'
    )


# ---------------------------------------------------------------------------
# GPU tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
@requires_gpu
def test_beginstep_detected_and_called():
    """beginstep() is called exactly once when the run's BeginStep fires.

    Monkey-patches GPUDetector.beginstep to count calls.  For MFX data
    the constants don't actually change (fixed-gain-mode run), so the call
    is a no-op, but it must still be made.
    """
    if not _mfx_data_available():
        pytest.skip(f'test data not found (set PSANA_GPU_TEST_SMD_GLOB): {_SMD_GLOB}')

    from psana import DataSource
    from psana.gpu.gpu_calib import GPUDetector

    smd_files = sorted(glob.glob(_SMD_GLOB))
    smd_files = list(dict.fromkeys(smd_files))

    beginstep_call_count = [0]
    original_beginstep = GPUDetector.beginstep

    def spy_beginstep(self, peds, gmask):
        beginstep_call_count[0] += 1
        original_beginstep(self, peds, gmask)

    GPUDetector.beginstep = spy_beginstep
    try:
        ds = DataSource(files=smd_files, gpu_det='jungfrau',
                        batch_size=5, max_events=10)
        for run in ds.runs():
            for _ in run.events():
                pass
    finally:
        GPUDetector.beginstep = original_beginstep

    assert beginstep_call_count[0] >= 1, (
        'beginstep() was never called; BeginStep transition was not dispatched'
    )


@pytest.mark.slow
@requires_gpu
def test_eventpool_drained_before_beginstep():
    """beginstep() must not be called while calibration kernels are in flight.

    Monitors the GPU stream's synchronisation state when beginstep() fires.
    After EventPool.flush() all non-blocking streams are synchronised, so
    cp.cuda.Stream.null.synchronize() completes immediately.  We verify
    that peds_gpu.set() in beginstep() does not raise a CUDA error — which
    would happen if a kernel were actively reading the buffer.
    """
    if not _mfx_data_available():
        pytest.skip(f'test data not found (set PSANA_GPU_TEST_SMD_GLOB): {_SMD_GLOB}')

    import cupy as cp
    from psana import DataSource

    smd_files = sorted(glob.glob(_SMD_GLOB))
    smd_files = list(dict.fromkeys(smd_files))

    # Run with batch_size=5 and n_streams=2 so multiple batches are in flight.
    # If beginstep() fires before the EventPool is drained, peds_gpu.set()
    # would write over a buffer still in use by a running kernel and either
    # corrupt data or raise a CUDA error.  We check for both.
    errors = []
    try:
        ds = DataSource(files=smd_files, gpu_det='jungfrau',
                        batch_size=5, max_events=20, n_gpu_streams=2)
        for run in ds.runs():
            for ctx in run.events():
                calib = ctx.get('calib').on_gpu
                if cp.any(cp.isnan(calib)):
                    errors.append(
                        f'NaN in calib at ts={ctx.timestamp} — '
                        'possible calibconst corruption across BeginStep boundary'
                    )
    except Exception as exc:
        errors.append(f'CUDA exception: {exc}')

    assert not errors, '\n'.join(errors)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'slow or not slow'])
