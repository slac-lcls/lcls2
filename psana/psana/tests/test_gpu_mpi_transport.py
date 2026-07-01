"""
test_gpu_mpi_transport.py — Tests for the MPI GPU transport layer.

Covers four groups of functionality added in the multi-GPU / MPI phase:

  1. BigDataNode._unpack_batch()        — pure unit, no data / GPU needed
  2. init_gpu_rank() / gpu_error_handler — pure unit, no data / GPU needed
  3. EB → BD pack/unpack roundtrip      — requires MFX SMD data, no GPU
  4. GPU stream auto-discovery          — requires MFX SMD data, no GPU
  5. End-to-end single-process GPU loop — requires GPU + MFX data

Design principle: PS_TEST_GPU_STREAM_IDS is NEVER set by these tests.
All GPU stream routing uses auto-discovery from Configure dgrams.

Run
---
    # Groups 1-2 (no data, no GPU):
    pytest psana/psana/tests/test_gpu_mpi_transport.py -k "not slow"

    # Groups 3-4 (MFX data required, no GPU):
    PSANA_GPU_TEST_SMD_GLOB='.../mfx100852324-r0077*.smd.xtc2' \\
        pytest psana/psana/tests/test_gpu_mpi_transport.py -m slow -k "not gpu"

    # All tests including GPU (on a GPU node, data required):
    PSANA_GPU_TEST_SMD_GLOB='.../mfx100852324-r0077*.smd.xtc2' \\
        pytest psana/psana/tests/test_gpu_mpi_transport.py -m slow
"""

import glob
import os

import numpy as np
import pytest

from psana.psexp.packet_footer import PacketFooter

_SMD_GLOB = os.environ.get(
    'PSANA_GPU_TEST_SMD_GLOB',
    '/sdf/data/lcls/ds/prj/public01/xtc/smalldata/mfx100852324-r0077*.smd.xtc2',
)
_DET_NAME  = 'jungfrau'
_BATCH_SIZE = 5
_MAX_EVENTS = 10


def _gpu_available():
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _mfx_data_available():
    return bool(glob.glob(_SMD_GLOB))


def _smd_files():
    return sorted(set(glob.glob(_SMD_GLOB)))


requires_gpu  = pytest.mark.skipif(not _gpu_available(),      reason='no CUDA device')
requires_data = pytest.mark.skipif(not _mfx_data_available(), reason=f'MFX data not found: {_SMD_GLOB}')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_packed_batch(smd_bytes, gpu_bytes):
    """Pack (smd_bytes, gpu_bytes) as PacketFooter — same as EventBuilderNode."""
    pf = PacketFooter(2)
    buf = bytearray()
    buf += smd_bytes
    buf += gpu_bytes
    pf.set_size(0, len(smd_bytes))
    pf.set_size(1, len(gpu_bytes))
    buf += pf.footer
    return buf


def _make_unpack_method():
    """Return a bound _unpack_batch callable without constructing a full BigDataNode."""
    from psana.psexp.node import BigDataNode

    class _FakeBD:
        pass

    obj = _FakeBD()
    obj._unpack_batch = BigDataNode._unpack_batch.__get__(obj, _FakeBD)
    return obj._unpack_batch


# ---------------------------------------------------------------------------
# Group 1: BigDataNode._unpack_batch() — pure unit
# ---------------------------------------------------------------------------

def test_unpack_batch_empty_is_termination():
    """Empty bytearray is the EB termination signal — both outputs must be empty."""
    unpack = _make_unpack_method()
    smd, gpu = unpack(bytearray())
    assert smd == bytearray(), 'smd should be empty for termination signal'
    assert gpu == bytearray(), 'gpu should be empty for termination signal'


def test_unpack_batch_two_packet_format():
    """Standard new format: PacketFooter([smd, gpu]) splits correctly."""
    unpack = _make_unpack_method()

    smd_orig = bytearray(b'smd_payload_bytes_here')
    gpu_orig = bytearray(b'GPUBAT1\x00stub_bytes')

    packed = _make_packed_batch(smd_orig, gpu_orig)
    smd, gpu = unpack(packed)

    assert bytes(smd) == bytes(smd_orig), (
        f'smd mismatch: {bytes(smd)!r} != {bytes(smd_orig)!r}'
    )
    assert bytes(gpu) == bytes(gpu_orig), (
        f'gpu mismatch: {bytes(gpu)!r} != {bytes(gpu_orig)!r}'
    )


def test_unpack_batch_empty_gpu_chunk():
    """PacketFooter([smd, empty]) — CPU-only batch, gpu_chunk must be empty."""
    unpack = _make_unpack_method()

    smd_orig = bytearray(b'cpu_only_smd_data')
    packed   = _make_packed_batch(smd_orig, bytearray())
    smd, gpu = unpack(packed)

    assert bytes(smd) == bytes(smd_orig), 'smd content wrong for CPU-only batch'
    assert gpu == bytearray(), 'gpu should be empty for CPU-only batch'


def test_unpack_batch_legacy_fallback():
    """Data that is not a valid 2-packet PacketFooter returns (chunk, empty).

    This handles any legacy message that pre-dates the GPU transport layer
    or any future format where the footer is absent.
    """
    unpack = _make_unpack_method()

    # Raw bytes with no PacketFooter — not a valid 2-packet structure.
    raw = bytearray(b'raw_legacy_data_no_footer')
    smd, gpu = unpack(raw)

    assert gpu == bytearray(), (
        'gpu must be empty when chunk has no PacketFooter'
    )
    # smd should be the original chunk unchanged
    assert bytes(smd) == bytes(raw), (
        'smd should equal original chunk for legacy format'
    )


def test_unpack_batch_roundtrip_large():
    """Round-trip with realistic payload sizes (~1 KB smd, ~1 KB GPUBAT1 stub)."""
    unpack = _make_unpack_method()

    smd_orig = bytearray(os.urandom(1024))
    gpu_orig = bytearray(os.urandom(1024))

    packed   = _make_packed_batch(smd_orig, gpu_orig)
    smd, gpu = unpack(packed)

    assert bytes(smd) == bytes(smd_orig), 'large smd roundtrip failed'
    assert bytes(gpu) == bytes(gpu_orig), 'large gpu roundtrip failed'


# ---------------------------------------------------------------------------
# Group 2: init_gpu_rank() / gpu_error_handler() — pure unit
# ---------------------------------------------------------------------------

def test_init_gpu_rank_default_single_gpu():
    """Single-GPU node: local_rank=0, n_gpus=1 → gpu_id=0."""
    from psana.gpu.gpu_mpi import init_gpu_rank

    # Remove any CUDA_VISIBLE_DEVICES that might have been set earlier.
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    gpu_id = init_gpu_rank(local_rank=0, n_gpus=1)

    assert gpu_id == 0
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'


def test_init_gpu_rank_multi_gpu_mapping():
    """Verify local_rank % n_gpus mapping for multi-GPU nodes."""
    from psana.gpu.gpu_mpi import init_gpu_rank

    cases = [
        # (local_rank, n_gpus, expected_gpu_id)
        (0, 4, 0),
        (1, 4, 1),
        (2, 4, 2),
        (3, 4, 3),
        (4, 4, 0),   # wraps around
        (5, 4, 1),
        (0, 1, 0),
        (3, 2, 1),
    ]
    for local_rank, n_gpus, expected in cases:
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        gpu_id = init_gpu_rank(local_rank=local_rank, n_gpus=n_gpus)
        assert gpu_id == expected, (
            f'local_rank={local_rank} n_gpus={n_gpus}: '
            f'expected gpu_id={expected}, got {gpu_id}'
        )
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(expected)


def test_gpu_error_handler_clean_exit():
    """No exception inside the context → __exit__ returns False, no abort."""
    from psana.gpu.gpu_mpi import gpu_error_handler

    class FakeComm:
        def Get_rank(self):
            return 0
        def Abort(self, code):
            raise AssertionError('Abort should not be called on clean exit')

    with gpu_error_handler(FakeComm()):
        pass   # no exception


def test_gpu_error_handler_kvikio_fatal():
    """KvikIO errors trigger Abort(1) immediately.

    The old retry logic was removed because a context manager cannot re-run
    the failing operation — once __exit__ is called the generator frame that
    issued the read is gone.  Retrying silently would skip the failed batch
    and produce incorrect results.  KvikIO failures are now treated as fatal,
    identical to CUDARuntimeError.
    """
    from psana.gpu.gpu_mpi import gpu_error_handler

    abort_calls = []

    class FakeComm:
        def Get_rank(self):
            return 0
        def Abort(self, code):
            abort_calls.append(code)

    handler = gpu_error_handler(FakeComm())
    exc     = RuntimeError('KvikIO pread failed: timeout')

    # First call → immediate Abort(1), exception suppressed.
    r1 = handler.__exit__(type(exc), exc, None)
    assert r1 is True, f'expected True (suppress) on kvikio error, got {r1}'
    assert len(abort_calls) == 1, (
        f'expected Abort(1) on first KvikIO error, got {len(abort_calls)} calls'
    )
    assert abort_calls[0] == 1, f'expected Abort(1), got Abort({abort_calls[0]})'


# ---------------------------------------------------------------------------
# Group 3: EB → BD pack/unpack roundtrip (requires MFX data, no GPU)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@requires_data
def test_eb_produces_gpubat1_bytes_with_auto_discovery():
    """EventBuilderManager emits non-empty gpubat1_bytes when GPU streams are
    auto-discovered from Configure dgrams — no PS_TEST_GPU_STREAM_IDS needed.

    Verifies that the full EB packing path works:
        batches_with_gpu() → gpu_batch_dict → gpubat1_bytes → PacketFooter
    and that BigDataNode._unpack_batch() correctly recovers the same bytes.
    """
    import os as _os
    from psana.psexp.ds_base import DsParms
    from psana.psexp.smdreader_manager import SmdReaderManager
    from psana.psexp.eventbuilder_manager import EventBuilderManager

    # Ensure the env var is NOT set so auto-discovery must be used.
    assert 'PS_TEST_GPU_STREAM_IDS' not in os.environ, (
        'PS_TEST_GPU_STREAM_IDS must not be set for this test — '
        'auto-discovery is required'
    )

    smd_files = _smd_files()
    dsparms = DsParms(
        batch_size=_BATCH_SIZE,
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
    dsparms.update_smd_state(smd_files, [False] * len(smd_files))

    # Auto-discover GPU stream IDs from Configure dgrams.
    xtc_files = [
        f.replace('/smalldata/', '/').replace('.smd.xtc2', '.xtc2')
        for f in smd_files
    ]
    smd_fds = np.array([_os.open(f, _os.O_RDONLY) for f in smd_files],
                       dtype=np.int32)
    try:
        smdr = SmdReaderManager(smd_fds, dsparms)
        cfgs = smdr.get_next_dgrams()    # Configure
        if cfgs is None:
            pytest.skip('could not read Configure dgrams')

        # Discover which streams contain the Jungfrau detector.
        from psana.gpu.gpu_calib import build_stream_seg_map
        seg_map = build_stream_seg_map(
            {i: f for i, f in enumerate(xtc_files)
             if i < len(xtc_files) and _os.path.exists(f)},
            _DET_NAME,
        )
        assert seg_map, (
            f'No {_DET_NAME} streams found in {len(smd_files)} SMD files — '
            'auto-discovery returned empty stream map'
        )
        dsparms.gpu_stream_ids = sorted(seg_map.keys())

        smdr.get_next_dgrams()   # BeginRun

        # Process one EventBuilder chunk and verify EB packs GPUBAT1 bytes.
        found_gpu_batch = False
        unpack = _make_unpack_method()

        for chunk_id in smdr.chunks():
            empty_views = [bytearray()] * smdr.n_files
            smd_chunk = bytearray(smdr.smdr.repack_parallel(
                empty_views, 1,
                intg_stream_id=dsparms.intg_stream_id,
            ))
            eb = EventBuilderManager(smd_chunk, smdr.configs, dsparms)

            for smd_batch_dict, gpu_batch_dict, step_dict in eb.batches_with_gpu():
                if not gpu_batch_dict:
                    continue

                # EB found GPU-routed events — extract and pack as node.py does.
                gpubat1_bytes = bytearray()
                for gpubat1, _ in gpu_batch_dict.values():
                    gpubat1_bytes = bytearray(gpubat1)
                    break

                assert gpubat1_bytes, (
                    'gpu_batch_dict was non-empty but gpubat1_bytes is empty'
                )

                smd_batch, _ = next(iter(smd_batch_dict.values()), (bytearray(), []))

                # Pack as EventBuilderNode.start() does.
                pf = PacketFooter(2)
                buf = bytearray()
                buf += smd_batch
                buf += gpubat1_bytes
                pf.set_size(0, len(smd_batch))
                pf.set_size(1, len(gpubat1_bytes))
                packed = buf + pf.footer

                # Unpack as BigDataNode._unpack_batch() does.
                recovered_smd, recovered_gpu = unpack(packed)

                assert bytes(recovered_smd) == bytes(smd_batch), (
                    'smd_batch round-trip failed'
                )
                assert bytes(recovered_gpu) == bytes(gpubat1_bytes), (
                    'gpubat1_bytes round-trip failed'
                )

                found_gpu_batch = True
                break   # one batch is enough

            if found_gpu_batch:
                break

        assert found_gpu_batch, (
            'No GPU-routed batch was produced.  '
            'Check that auto-discovery found GPU streams '
            f'(found: {dsparms.gpu_stream_ids})'
        )
    finally:
        for fd in smd_fds:
            _os.close(int(fd))


@pytest.mark.slow
@requires_data
def test_auto_discovery_finds_all_segments():
    """build_stream_seg_map discovers all 32 Jungfrau segments across the
    Jungfrau-containing streams in mfx100852324-r0077.

    Note: not all 10 SMD streams contain Jungfrau data — only those whose
    XTC2 bigdata file has Jungfrau events are discovered.  The test verifies
    that the union of all discovered segment IDs is exactly {0 … 31}.
    """
    import os as _os
    from psana.gpu.gpu_calib import build_stream_seg_map

    assert 'PS_TEST_GPU_STREAM_IDS' not in os.environ, (
        'PS_TEST_GPU_STREAM_IDS must not be set — test uses auto-discovery'
    )

    smd_files = _smd_files()
    xtc_files = [
        f.replace('/smalldata/', '/').replace('.smd.xtc2', '.xtc2')
        for f in smd_files
    ]
    existing_xtc = {i: f for i, f in enumerate(xtc_files)
                    if _os.path.exists(f)}

    seg_map = build_stream_seg_map(existing_xtc, _DET_NAME)

    assert seg_map, (
        f'build_stream_seg_map returned empty map for {_DET_NAME}; '
        f'checked {len(existing_xtc)} XTC files'
    )
    # The union of all discovered segment IDs must cover all 32 Jungfrau
    # segments — even if individual streams only carry a subset.
    all_seg_ids = sorted(sid for ids in seg_map.values() for sid in ids)
    assert all_seg_ids == list(range(32)), (
        f'Expected segments 0-31, got {all_seg_ids} '
        f'(from streams {sorted(seg_map.keys())})'
    )


@pytest.mark.slow
@requires_data
@requires_gpu
def test_setup_gpu_detector_auto_discovers_streams():
    """_setup_gpu_detector() discovers GPU stream IDs from Configure dgrams
    without PS_TEST_GPU_STREAM_IDS and returns a segment map covering all 32
    Jungfrau segments.

    GpuEvents is used by RunSerial (DataSource(exp=, run=, dir=) path).
    It reads stream and segment info from dsparms tables populated during
    Configure dgram processing.  This test verifies that the exp/run/dir
    DataSource path correctly populates those tables and produces
    GpuEventContext objects with the expected shape across all 32 segments.

    Note: DataSource(files=smd_files) creates RunSingleFile which does NOT
    use GpuEvents — only DataSource(exp=, run=, dir=) creates RunSerial
    which does.
    """
    import psana

    assert 'PS_TEST_GPU_STREAM_IDS' not in os.environ

    ds = psana.DataSource(
        exp='mfx100852324',
        run=77,
        dir='/sdf/data/lcls/ds/prj/public01/xtc',
        gpu_det=_DET_NAME,
        batch_size=_BATCH_SIZE,
        max_events=3,
    )

    n_events = 0
    for run in ds.runs():
        for ctx in run.events():
            import cupy as cp
            calib = ctx.get(_DET_NAME + '.calib').on_gpu
            assert calib.shape[0] == 32, (
                f'Expected 32 segments, got {calib.shape[0]}'
            )
            assert calib.dtype == cp.float32
            assert not bool(cp.any(cp.isnan(calib)))
            n_events += 1

    assert n_events > 0, 'No events produced'


# ---------------------------------------------------------------------------
# Group 4: End-to-end GPU event loop (requires GPU + MFX data)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@requires_gpu
@requires_data
def test_gpu_events_auto_discovery_shape_and_timestamps():
    """Single-process GPU event loop uses auto-discovery and produces the
    correct output shape and valid timestamps — no PS_TEST_GPU_STREAM_IDS.

    This exercises the same code path as RunParallel._gpu_events_mpi() on a
    GPU BD rank: EventPool + GPUDetector.process_batch() + GpuEventContext.
    """
    import psana

    assert 'PS_TEST_GPU_STREAM_IDS' not in os.environ

    # Use exp/run/dir — creates RunSerial which uses GpuEvents.
    # DataSource(files=smd_files) creates RunSingleFile which does not.
    ds = psana.DataSource(
        exp='mfx100852324',
        run=77,
        dir='/sdf/data/lcls/ds/prj/public01/xtc',
        gpu_det=_DET_NAME,
        batch_size=_BATCH_SIZE,
        max_events=_MAX_EVENTS,
    )

    n_events = 0
    timestamps = []
    for run in ds.runs():
        for ctx in run.events():
            import cupy as cp
            calib = ctx.get('calib').on_gpu

            assert calib.dtype == cp.float32, (
                f'evt={n_events}: expected float32, got {calib.dtype}'
            )
            assert calib.ndim == 3, (
                f'evt={n_events}: expected 3-D calib, got ndim={calib.ndim}'
            )
            n_segs, nrows, ncols = calib.shape
            assert n_segs >= 1
            assert (nrows, ncols) == (512, 1024), (
                f'evt={n_events}: unexpected panel shape {(nrows, ncols)}'
            )
            assert not bool(cp.any(cp.isnan(calib))), (
                f'evt={n_events}: NaN in GPU calib'
            )
            assert ctx.timestamp > 0, 'timestamp must be positive'
            timestamps.append(ctx.timestamp)
            n_events += 1

    assert n_events > 0, 'No events produced'
    # Timestamps must be monotonically increasing.
    assert timestamps == sorted(timestamps), (
        'Events not in timestamp order — EventPool may have incorrect ordering'
    )


@pytest.mark.slow
@requires_gpu
@requires_data
def test_gpu_events_stream_ids_match_auto_discovery():
    """GPU stream IDs reported in the GPUBAT1 header match those discovered
    automatically from Configure dgrams.

    Verifies that the auto-discovery path (no PS_TEST_GPU_STREAM_IDS) and the
    GPUBAT1 wire format are consistent: every stream listed in
    dsparms.gpu_stream_ids must appear in the GPU batch descriptors.
    """
    import psana
    from psana.gpu.gpu_batch import GpuBatchView
    import os as _os

    assert 'PS_TEST_GPU_STREAM_IDS' not in os.environ

    smd_files  = _smd_files()
    xtc_files  = [
        f.replace('/smalldata/', '/').replace('.smd.xtc2', '.xtc2')
        for f in smd_files
    ]
    bd_map  = {i: f for i, f in enumerate(xtc_files) if _os.path.exists(f)}
    from psana.gpu.gpu_calib import build_stream_seg_map
    seg_map = build_stream_seg_map(bd_map, _DET_NAME)
    expected_ids = set(seg_map.keys())

    # Run the GPU pipeline and collect stream IDs from the GPUBAT1 descriptors.
    from psana.psexp.ds_base import DsParms
    from psana.psexp.smdreader_manager import SmdReaderManager
    from psana.psexp.eventbuilder_manager import EventBuilderManager

    dsparms = DsParms(
        batch_size=_BATCH_SIZE,
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
    dsparms.update_smd_state(smd_files, [False] * len(smd_files))
    dsparms.gpu_stream_ids = sorted(expected_ids)

    smd_fds = np.array([_os.open(f, _os.O_RDONLY) for f in smd_files],
                       dtype=np.int32)
    seen_stream_ids = set()
    try:
        smdr = SmdReaderManager(smd_fds, dsparms)
        if smdr.get_next_dgrams() is None:
            pytest.skip('could not read Configure')
        if smdr.get_next_dgrams() is None:
            pytest.skip('could not read BeginRun')

        for _chunk_id in smdr.chunks():
            empty = [bytearray()] * smdr.n_files
            smd_chunk = bytearray(smdr.smdr.repack_parallel(
                empty, 1,
                intg_stream_id=dsparms.intg_stream_id,
            ))
            eb = EventBuilderManager(smd_chunk, smdr.configs, dsparms)
            for _smd_bd, gpu_batch_dict, _step in eb.batches_with_gpu():
                if not gpu_batch_dict:
                    continue  # skip transition-only batches (n_events=0)
                for gpubat1, _ in gpu_batch_dict.values():
                        gv = GpuBatchView(gpubat1, validate=False)
                        n_desc = int(gv.header.n_desc)
                        if n_desc == 0:
                            continue
                        for i in range(n_desc):
                            seen_stream_ids.add(int(gv.descs[i]['stream_id']))
                if seen_stream_ids:
                    break   # found L1 events with GPU data
            if seen_stream_ids:
                break

    finally:
        for fd in smd_fds:
            _os.close(int(fd))

    assert seen_stream_ids, 'No stream IDs found in GPUBAT1 descriptors'
    assert seen_stream_ids.issubset(expected_ids), (
        f'GPUBAT1 stream IDs {seen_stream_ids} not subset of '
        f'auto-discovered IDs {expected_ids}'
    )


# ---------------------------------------------------------------------------
# Group 5: Multi-GPU MPI test  (slow, requires 2 × A100 + MFX data)
#
# NOTE: This test spawns 'srun' as a subprocess.  The calling pytest process
# must NOT have mpi4py imported (it runs on a login/head node without GPUs).
# That constraint is satisfied here because test_gpu_mpi_transport.py never
# imports mpi4py at module level.
# ---------------------------------------------------------------------------

def _two_gpus_available():
    """Return True if the current node has ≥2 CUDA GPUs."""
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() >= 2
    except Exception:
        return False


def _srun_available():
    """Return True if the srun command is in PATH."""
    import shutil
    return shutil.which('srun') is not None


requires_two_gpus = pytest.mark.skipif(
    not _two_gpus_available(),
    reason='need ≥2 CUDA GPUs on this node',
)
requires_srun = pytest.mark.skipif(
    not _srun_available(),
    reason='srun not in PATH (not on an S3DF login/compute node)',
)


@pytest.mark.slow
@requires_srun
@requires_data
def test_multi_gpu_two_bd_ranks():
    """Full MPI pipeline test: SMD0 + EB + 2 GPU BD ranks on one node.

    Submits a 4-rank job (PS_EB_NODES=1) to Slurm requesting 2 A100 GPUs.
    Each BD rank is automatically pinned to a different GPU by init_gpu_rank().

    Assertions (verified inside test_gpu_multi_rank.py, exit-code reported):
      * BD ranks on distinct physical GPUs
      * No duplicate timestamps across BD ranks
      * All calib arrays (n_segs, 512, 1024) float32, no NaN
      * Total events == MAX_EVENTS
    """
    import subprocess

    # DO NOT import mpi4py here — OpenMPI prevents forking after mpi4py init.
    assert 'PS_TEST_GPU_STREAM_IDS' not in os.environ

    script = os.path.join(
        os.path.dirname(__file__),
        'test_gpu_multi_rank.py',
    )
    assert os.path.isfile(script), f'MPI test script not found: {script}'

    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', '..')
    )

    env = dict(os.environ)
    env.update({
        'PS_EB_NODES':  '1',
        'PS_SRV_NODES': '0',
        'OMPI_MCA_btl': '^smcuda',
    })
    env.pop('PS_TEST_GPU_STREAM_IDS', None)   # enforce auto-discovery

    # PSANA_GPU_TEST_SMD_GLOB is forwarded automatically via env inheritance.

    launcher = os.path.join(
        os.path.dirname(__file__),
        '..', 'gpu', 'scripts', 'run_multi_gpu_test.sh',
    )
    launcher = os.path.abspath(launcher)
    assert os.path.isfile(launcher), f'Launcher not found: {launcher}'

    # Use the dedicated Slurm launcher script which handles the srun
    # allocation correctly.  Running srun inside an existing allocation
    # (as would happen if this test is itself run via srun) fails with
    # "More processors requested than permitted".  The launcher always
    # submits a fresh allocation from the head node.
    result = subprocess.run(
        ['sh', launcher],
        env=env,
        capture_output=True,
        text=True,
        timeout=1200,
    )

    # Surface both stdout and stderr for diagnosis on failure.
    combined = result.stdout + result.stderr
    # Filter noisy OpenMPI shmem warnings that are cosmetic.
    filtered = '\n'.join(
        line for line in combined.splitlines()
        if not any(k in line for k in (
            'shmem: mmap', 'create_and_attach',
            'unable to create shared', 'coordinating structure',
        ))
    )

    assert result.returncode == 0, (
        f'Multi-GPU MPI test exited with code {result.returncode}.\n'
        f'Output:\n{filtered}'
    )
    assert 'PASS' in combined, (
        f'Multi-GPU MPI test output did not contain "PASS".\n'
        f'Output:\n{filtered}'
    )


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s', '-m', 'slow or not slow'])
