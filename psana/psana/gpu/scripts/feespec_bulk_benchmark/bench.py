"""MPI feespec consumer, optionally alongside the Jungfrau GPU pipeline."""
import argparse
import gc
import gzip
import hashlib
import json
import os
import pickle
from pathlib import Path
import struct
import time

if os.environ.get('BENCH_CPU_AFFINITY'):
    os.sched_setaffinity(0, {int(x) for x in os.environ['BENCH_CPU_AFFINITY'].split(',')})
rank_hint = int(os.environ.get('OMPI_COMM_WORLD_RANK', '0'))
if rank_hint < 2:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
from mpi4py import MPI
import numpy as np
from psana import DataSource
from psana.psexp.run import Run
from common import cache_inputs, digest, network

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
is_bd = rank == 2
active = False
counts = dict(requests=0, bytes=0, cpu_bd_reads=0, input_windows=0,
              read_submit_s=0., read_wait_s=0., request_to_ready_s=0., request_sizes={}, streams={})


def exclusive_feespec(include_jf=False):
    """Explicit benchmark exception: route the entire shared s000 to the GPU."""
    from psana.psexp.ds_base import DsParms
    original = DsParms.resolve_gpu_stream_ids

    def resolve(self):
        expected = ['jungfrau', 'feespec'] if include_jf else ['feespec']
        assert self._detector_names(self.gpu_det) == expected
        assert not self.hybrid_det
        streams = self.det_stream_ids_table['feespec']
        assert len(streams) == 1
        original_owners = self.stream_id_to_detnames
        owners = list(original_owners[streams[0]])
        assert 'feespec' in owners and 'jungfrau' not in owners
        self.stream_id_to_detnames = dict(original_owners)
        self.stream_id_to_detnames[streams[0]] = ['feespec']
        try:
            original(self)
        finally:
            self.stream_id_to_detnames = original_owners
        print('ROUTING_OVERRIDE ' + json.dumps(dict(rank=rank, streams=streams, original_owners=owners)), flush=True)
    DsParms.resolve_gpu_stream_ids = resolve

def no_calibration_services(constants_path=None):
    # Use the validated frozen JF constants; feespec needs no calibration adapter.
    def no_calibration(self):
        self._clear_calibconst()
        self._calib_const = {name: {} for name in self.dsparms.configinfo_dict}
        if constants_path:
            with gzip.open(constants_path, 'rb') as source:
                constants = pickle.load(source)
            assert 'jungfrau' in constants
            self._calib_const['jungfrau'] = constants['jungfrau']
        self.dsparms.calibconst = self._calib_const
    Run._setup_run_calibconst = no_calibration


def diagnostic_hooks(variant, include_jf=False):
    from psana.gpu.gpu_kvikio_read import KvikioGpuReader
    from psana.psexp.event_manager import EventManager
    issue_name = '_submit_read' if hasattr(KvikioGpuReader, '_submit_read') else 'issue_batch'
    issue, wait, read = getattr(KvikioGpuReader, issue_name), KvikioGpuReader.wait_batch, EventManager._read

    def issued(self, *args, **kwargs):
        start = time.perf_counter()
        pending = issue(self, *args, **kwargs)
        if active:
            counts['read_submit_s'] += time.perf_counter() - start
            counts['input_windows'] += 1
            dm = args[1]
            for desc, size, _ in pending.futures:
                counts['requests'] += 1
                counts['bytes'] += size
                sizes = counts['request_sizes']
                sizes[str(size)] = sizes.get(str(size), 0) + 1
                name = Path(desc.file.path if hasattr(desc, 'file') else dm.xtc_files[desc.stream_id]).name
                entry = counts['streams'].setdefault(name, dict(requests=0, bytes=0))
                entry['requests'] += 1
                entry['bytes'] += size
        return pending

    def waited(self, *args, **kwargs):
        start = time.perf_counter()
        result = wait(self, *args, **kwargs)
        if active:
            counts['read_wait_s'] += time.perf_counter() - start
        return result

    def cpu_read(self, *args, **kwargs):
        if active:
            counts['cpu_bd_reads'] += 1
        return read(self, *args, **kwargs)
    if variant != 'A' or include_jf:
        setattr(KvikioGpuReader, issue_name, issued)
        KvikioGpuReader.wait_batch = waited
    EventManager._read = cpu_read


def native_read_stats():
    """Accumulate existing reader counters across EB resets; no per-read tracing."""
    from psana.gpu.gpu_kvikio_read import KvikioGpuReader
    original = KvikioGpuReader.wait_batch
    def waited(self, pending):
        collect = active and not pending.completed
        before = (self._total_bytes_read, self._total_io_ns, self._total_issue_to_complete_ns)
        result = original(self, pending)
        if collect:
            counts['requests'] += len(pending.futures)
            counts['input_windows'] += 1
            counts['bytes'] += self._total_bytes_read - before[0]
            counts['read_wait_s'] += (self._total_io_ns - before[1]) / 1e9
            counts['request_to_ready_s'] += (self._total_issue_to_complete_ns - before[2]) / 1e9
        return result
    KvikioGpuReader.wait_batch = waited


def run(a, n, timed):
    global active
    kwargs = dict(exp='mfx101210926', run=387, dir=a.directory,
                    detectors=['jungfrau', 'feespec'] if a.include_jf else ['feespec'], max_events=n,
                    batch_size=100, n_gpu_streams=a.pool_depth, gpu_memory_budget_gb=8,
                    gpu_d2h_chunk_size=0, skip_calib_load='all', log_level='ERROR')
    if a.bulk_target_bytes is not None:
        kwargs['gpu_bulk_target_bytes'] = a.bulk_target_bytes
    if a.variant != 'A':
        kwargs.update(gpu_det=['jungfrau', 'feespec'] if a.include_jf else 'feespec',
                      gpu_bulk_read=a.variant == 'E-on')
    elif a.include_jf:
        kwargs['gpu_det'] = 'jungfrau'
    ds = DataSource(**kwargs)
    run = next(ds.runs())
    cpu_detector = run.Detector('feespec') if a.variant == 'A' else None
    assert ds.dsparms.batch_size == 100
    assert ds.dsparms.n_gpu_streams == a.pool_depth
    if a.bulk_target_bytes is not None:
        assert ds.dsparms.gpu_bulk_target_bytes == a.bulk_target_bytes
    if is_bd:
        import cupy as cp
        result = cp.empty(n, dtype=cp.int64)
        cp.cuda.Device().synchronize()
    before = net0 = None
    if timed and rank == 0:
        before = cache_inputs(a.directory, a.cache, include_jf=a.include_jf, ranges=a.cache_ranges)
        print('CACHE_BEFORE ' + json.dumps(before), flush=True)
        net0 = network()
    comm.Barrier()
    active = timed and (a.diagnostic or a.read_stats)
    stamps = []
    array_digest = hashlib.sha256()
    jf_checks = []
    trace_active = timed and a.fallback_trace is not None
    if trace_active:
        a.fallback_trace.begin()
    if timed and a.pipeline_stats is not None:
        a.pipeline_stats.begin()
    profiler = None
    if timed and is_bd and a.python_profile:
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()
    start = time.perf_counter()
    for i, evt in enumerate(run.events()):
        stamps.append(int(evt.timestamp))
        if a.include_jf and a.diagnostic and int(evt.timestamp) in a.jf_reference:
            jf_checks.append(dict(timestamp=int(evt.timestamp),
                raw=digest(evt.gpu.get('jungfrau.raw').on_cpu),
                calib=digest(evt.gpu.get('jungfrau.calib').on_cpu)))
        if a.variant == 'A':
            host_values = cpu_detector.raw.hproj(evt)
            values = cp.asarray(host_values)
            assert values.shape == (2048,) and values.dtype == np.int32
            result[i] = cp.sum(values, dtype=cp.int64)
            if a.diagnostic:
                array_digest.update(values.get().tobytes())
        else:
            field = evt.gpu.detector('feespec').field('raw', 'hproj')
            with field.on_gpu_view(cp.cuda.Stream.null) as segments:
                values = segments[0]
                assert values.shape == (2048,) and values.dtype == np.int32
                # Identical GPU consumer; E also includes its field locator access.
                result[i] = cp.sum(values, dtype=cp.int64)
                if a.diagnostic:
                    array_digest.update(values.get().tobytes())
            # A with-block does not clear its target or extracted array aliases.
            # Release these zero-copy aliases before requesting the next event;
            # otherwise 8 KiB of feespec can retain the full mixed input backing.
            field = segments = values = None
    if is_bd:
        cp.cuda.Device().synchronize()
    elapsed = time.perf_counter() - start
    if profiler is not None:
        profiler.disable()
        profiler.dump_stats(a.python_profile)
    pipeline_result = a.pipeline_stats.end() if timed and a.pipeline_stats is not None else None
    active = False
    fallback_result = a.fallback_trace.end(a.fallback_output) if trace_active else None
    comm.Barrier()
    if timed and rank == 0:
        net1 = network()
        after = cache_inputs(a.directory, a.cache, prepare=False, include_jf=a.include_jf, ranges=a.cache_ranges)
        delta = {k:net1[k]-v for k,v in net0.items()}
    sums_digest = None
    if is_bd:
        assert len(stamps) == n
        sums_digest = hashlib.sha256(result.get().tobytes()).hexdigest()
    gathered = comm.gather(dict(rank=rank, elapsed=elapsed, timestamps=stamps,
        sums_sha256=sums_digest, arrays_sha256=array_digest.hexdigest(),
        counts=counts, jf_checks=jf_checks, fallback_trace=fallback_result, pipeline_stats=pipeline_result,
        python_profile=str(a.python_profile) if profiler is not None else None,
        affinity=sorted(os.sched_getaffinity(0))), root=0)
    if not timed and a.warmup_reference and rank == 0:
        reference = json.loads(Path(a.warmup_reference).read_text())
        bd = gathered[2]
        assert reference['events'] == n
        assert hashlib.sha256(struct.pack(f'<{n}Q', *bd['timestamps'])).hexdigest() == reference['timestamp_sha256']
        assert bd['sums_sha256'] == reference['sums_sha256']
        assert bd['arrays_sha256'] == reference['arrays_sha256']
        assert bd['jf_checks'] == list(a.jf_reference.values())
        print('WARMUP_CHECK '+json.dumps(dict(events=n,feespec_arrays_pass=True,jf_samples=len(bd['jf_checks']))),flush=True)
    if timed and rank == 0:
        reference = json.loads(Path(a.reference).read_text())
        bd = gathered[2]
        ts = bd['timestamps']
        assert len(ts) == len(set(ts)) == n
        timestamp_hash = hashlib.sha256(struct.pack(f'<{n}Q', *ts)).hexdigest()
        assert timestamp_hash == reference['timestamp_sha256']
        assert bd['sums_sha256'] == reference['sums_sha256']
        assert reference['events'] == n
        if a.read_stats:
            assert a.variant != 'A'
            assert bd['counts']['bytes'] == reference['payload_bytes']
            if a.variant == 'E-off':
                assert bd['counts']['requests'] == n * (6 if a.include_jf else 1)
        if a.diagnostic:
            assert bd['arrays_sha256'] == reference['arrays_sha256']
            if a.variant != 'A':
                assert bd['counts']['cpu_bd_reads'] == 0
                assert bd['counts']['bytes'] == reference['payload_bytes']
            else:
                assert bd['counts']['cpu_bd_reads'] > 0
                if a.include_jf:
                    assert bd['counts']['bytes'] == reference['payload_bytes'] - reference['feespec_dgram_bytes']
            if a.variant == 'E-off':
                assert bd['counts']['requests'] == n * (6 if a.include_jf else 1)
            if a.include_jf:
                assert bd['jf_checks'] == list(a.jf_reference.values()), 'JF CPU/GPU mismatch'
        rx = max(v for k,v in delta.items() if k.endswith('/rx_bytes_phy'))
        if a.cache == 'cold' and rx < .98 * reference['payload_bytes']:
            raise RuntimeError(f'Insufficient physical NIC traffic for cold: {rx}')
        seconds = max(r['elapsed'] for r in gathered)
        output = dict(variant=a.variant, cache=a.cache, diagnostic=a.diagnostic, include_jf=a.include_jf,read_stats=a.read_stats,
            events=n, pool_depth=a.pool_depth, bulk_target_bytes=a.bulk_target_bytes,
            task_size=a.task_size, loop_s=seconds, events_per_s=n/seconds,
            input_gbps=reference['payload_bytes']/seconds/1e9,
            array_gbps=n*8192/seconds/1e9, payload_bytes=reference['payload_bytes'],
            timestamp_sha256=timestamp_hash, sums_sha256=bd['sums_sha256'],
            arrays_sha256=bd['arrays_sha256'] if a.diagnostic else None,
            cache_before=before, cache_after=after, network_delta=delta,
            physical_rx_bytes=rx, ranks=[{k:v for k,v in r.items() if k!='timestamps'} for r in gathered])
        print('RESULT ' + json.dumps(output), flush=True)
    evt = field = segments = values = None
    run.close_shared_memory()
    del run, ds
    if is_bd:
        del result
    gc.collect()
    if is_bd:
        cp.get_default_memory_pool().free_all_blocks()
    comm.Barrier()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory', required=True)
    p.add_argument('--reference', required=True)
    p.add_argument('--variant', choices=['A','E-off','E-on'], required=True)
    p.add_argument('--cache', choices=['cold','warm'], required=True)
    p.add_argument('--diagnostic', action='store_true')
    p.add_argument('--include-jf', action='store_true')
    p.add_argument('--constants')
    p.add_argument('--jf-reference', dest='jf_reference_path')
    p.add_argument('--events', type=int, default=10000)
    p.add_argument('--warmup-events', type=int, default=100)
    p.add_argument('--warmup-reference')
    p.add_argument('--range-manifest')
    p.add_argument('--read-stats', action='store_true')
    p.add_argument('--bulk-target-bytes', type=int, default=None,
                   help='Override the runtime bulk target; omit for historical builds')
    p.add_argument('--task-size', type=int, default=1 << 20)
    p.add_argument('--pool-depth', type=int, choices=(1, 2), default=1)
    p.add_argument('--pipeline-stats', action='store_true')
    p.add_argument('--python-profile', type=Path,
                   help='BD-only cProfile of the measured loop; excludes warmup and cache preparation')
    p.add_argument('--fallback-library')
    p.add_argument('--fallback-output')
    a = p.parse_args()
    assert a.events > 0 and a.warmup_events > 0
    assert not (a.read_stats and a.diagnostic)
    assert not (a.python_profile and (a.pipeline_stats or a.fallback_library or a.diagnostic))
    a.cache_ranges = None
    if a.range_manifest:
        manifest = json.loads(Path(a.range_manifest).read_text())
        assert manifest['events'] == a.events
        a.cache_ranges = {r['name']:r['last_end'] for r in manifest['streams']}
        assert len(a.cache_ranges) == (6 if a.include_jf else 1)
    a.jf_reference = {}
    if a.include_jf:
        assert a.constants and a.jf_reference_path
        # The JSON contains the trusted CPU raw/calibrated checks from the JF baseline.
        a.jf_reference = {r['timestamp']:r for r in json.loads(Path(a.jf_reference_path).read_text())}
    if a.variant != 'A':
        exclusive_feespec(a.include_jf)
    no_calibration_services(a.constants if a.include_jf else None)
    if is_bd:
        import cupy as cp
        import kvikio, kvikio.defaults as defaults
        import psana
        assert (bool(defaults.compat_mode()), defaults.get_num_threads(), defaults.task_size()) == (True,8,a.task_size)
        bus = cp.cuda.runtime.deviceGetPCIBusId(0)
        print('RUNTIME '+json.dumps(dict(psana=psana.__file__,cupy=cp.__version__,
            kvikio=kvikio.__version__,gpu=bus.decode() if isinstance(bus,bytes) else bus,
            compat=bool(defaults.compat_mode()),threads=defaults.get_num_threads(),task_size=defaults.task_size())),flush=True)
    if a.diagnostic:
        diagnostic_hooks(a.variant, a.include_jf)
    elif a.read_stats:
        native_read_stats()
    a.fallback_trace = None
    if a.fallback_library:
        assert a.fallback_output and a.variant != 'A'
        if is_bd:
            from kvikio_fallback_trace import FallbackTrace
            a.fallback_trace = FallbackTrace(a.fallback_library)
    pipeline_stats = a.pipeline_stats
    a.pipeline_stats = None
    if pipeline_stats and is_bd:
        from pipeline_stats import PipelineStats
        a.pipeline_stats = PipelineStats()
    diagnostic = a.diagnostic
    a.diagnostic = diagnostic or bool(a.warmup_reference)
    run(a,a.warmup_events,False)
    a.diagnostic = diagnostic
    run(a,a.events,True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        comm.Abort(2)
