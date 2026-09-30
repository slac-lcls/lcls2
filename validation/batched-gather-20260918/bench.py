"""Cross-revision MPI timing; correctness and detailed diagnostics are separate modes."""
import argparse
import cProfile
from collections import Counter, defaultdict
import gc
import gzip
import hashlib
import importlib.util
import json
import os
import pickle
from pathlib import Path
import struct
import sys
import time

# Keep the controller's allocation CPU mask identical for all variants/ranks.
if os.environ.get('BENCH_CPU_AFFINITY'):
    os.sched_setaffinity(0, {int(cpu) for cpu in os.environ['BENCH_CPU_AFFINITY'].split(',')})

# Respect the scheduler-assigned GPU. Non-BDs must not create CUDA contexts.
rank_hint = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
if rank_hint < 2:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
from mpi4py import MPI
import numpy as np
from psana import DataSource
import psana

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
is_bd = rank >= 2
active = False
stats = dict(reads=0, bytes=0, submissions=0, ledger_peak=0, ledger_held_peak=0)
streams = defaultdict(lambda: dict(reads=0, bytes=0, min_size=None, max_size=0))
subbatches = Counter()
residency = Counter()


def placement():
    status = Path('/proc/self/status').read_text().splitlines()
    result = dict(rank=rank, pid=os.getpid(), affinity=sorted(os.sched_getaffinity(0)),
                  numa_status=[line for line in status if line.startswith(('Cpus_allowed_list:', 'Mems_allowed_list:'))],
                  cupy_cache_dir=os.environ.get('CUPY_CACHE_DIR'))
    # Aggregate actual process NUMA pages; sampling occurs outside timing.
    pages = Counter()
    for line in Path('/proc/self/numa_maps').read_text().splitlines():
        for word in line.split():
            key, sep, value = word.partition('=')
            if sep and key.startswith('N') and key[1:].isdigit():
                pages[key] += int(value)
    result['numa_pages'] = dict(pages)
    return result


def sample_pool():
    import cupy as cp
    pool = cp.get_default_memory_pool()
    for name, value in (('pool_total', pool.total_bytes()), ('pool_used', pool.used_bytes())):
        stats[name + '_peak'] = max(stats.get(name + '_peak', 0), value)


def hooks():
    from psana.gpu.gpu_kvikio_read import KvikioGpuReader
    from psana.gpu.gpu_stream import EventPool
    issue_orig = KvikioGpuReader.issue_batch
    submit_orig = EventPool.submit

    def issue(self, view, dm, *args, **kwargs):
        result = issue_orig(self, view, dm, *args, **kwargs)
        if active:
            stats['io_path'] = self.io_path
            for desc, nbytes, _ in result.futures:
                if hasattr(desc, 'file'):
                    name = Path(desc.file.path).name
                else:
                    name = Path(dm.xtc_files[desc.stream_id]).name
                entry = streams[name]
                entry['reads'] += 1
                entry['bytes'] += int(nbytes)
                entry['min_size'] = min(entry['min_size'] or nbytes, nbytes)
                entry['max_size'] = max(entry['max_size'], nbytes)
                stats['reads'] += 1
                stats['bytes'] += int(nbytes)
        return result

    def submit(self, view, *args, **kwargs):
        if active:
            stats['submissions'] += 1
            count = view.n_events if hasattr(view, 'n_events') else view.header.n_events
            subbatches[int(count)] += 1
        return submit_orig(self, view, *args, **kwargs)

    KvikioGpuReader.issue_batch = issue
    EventPool.submit = submit
    try:
        from psana.gpu.gpu_budget import _GpuBudget
    except ImportError:
        from psana.gpu.gpu_events import _GpuBudget
    for method in ('reserve', 'hold'):
        original = getattr(_GpuBudget, method, None)
        if original is None:
            continue
        def account(self, *args, _original=original, **kwargs):
            result = _original(self, *args, **kwargs)
            if active:
                committed = self.committed()
                stats['ledger_peak'] = max(stats['ledger_peak'], committed)
                stats['ledger_held_peak'] = max(stats['ledger_held_peak'], committed + getattr(self, '_held', 0))
                sample_pool()
            return result
        setattr(_GpuBudget, method, account)
    from psana.gpu.gpu_events import GpuEventManager
    start_orig = getattr(GpuEventManager, '_start_resident_input', None)
    if start_orig:
        def resident(self, view, plan):
            if active:
                residency[str(tuple(plan.resident_streams))] += 1
            return start_orig(self, view, plan)
        GpuEventManager._start_resident_input = resident


def cache_state(a, prepare=True):
    if prepare:
        print(f'CACHE_PREPARE mode={a.cache} time={time.strftime("%FT%T")}', flush=True)
    helper = Path(a.repo) / 'notes/jungfrau_gpu_scale_20260828/page_cache_residency.py'
    spec = importlib.util.spec_from_file_location('cache_residency', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paths = sorted(Path(a.dir).glob('*.xtc2'))
    if prepare and a.cache == 'cold':
        for path in paths:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.fsync(fd)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
    attempts = a.warm_cache_passes if prepare and a.cache == 'warm' else 1
    for attempt in range(1, attempts + 1):
        if prepare and a.cache == 'warm':
            for path in paths:
                with open(path, 'rb', buffering=0) as stream:
                    while stream.read(16 * 1024**2):
                        pass
        records = [module.file_residency(str(path)) for path in paths]
        fraction = sum(r['resident_pages'] for r in records) / sum(r['pages'] for r in records)
        if not (prepare and a.cache == 'warm') or fraction >= 0.99:
            break
        print('CACHE_RETRY ' + json.dumps(dict(attempt=attempt, max_attempts=attempts,
                                             fraction=fraction)), flush=True)
    print(('CACHE_BEFORE ' if prepare else 'CACHE_AFTER ') + json.dumps(dict(mode=a.cache, fraction=fraction, files=records)), flush=True)
    if prepare and a.cache == 'cold' and fraction > 0.01:
        raise RuntimeError(f'cold residency too high: {fraction}')
    if a.cache == 'warm' and fraction < 0.99:
        raise RuntimeError(f'warm residency too low: {fraction}')
    return fraction


def digest(array):
    value = np.array(array, copy=True, order='C')
    if np.issubdtype(value.dtype, np.floating):
        # Match array_equal(equal_nan=True): signed zero and NaN payload bits
        # are not calibration-value mismatches. Finite nonzero bits stay exact.
        value[value == 0] = 0
        value[np.isnan(value)] = np.nan
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.tobytes()).hexdigest(), nonzero=int(np.count_nonzero(value)))


def run_once(a, count, *, check=False, timed=False):
    global active
    check = check or (timed and a.diagnostics)
    detectors = ['jungfrau', 'epix100_0'] if a.exp == 'mfx100848724' else ['jungfrau']
    kwargs = dict(exp=a.exp, run=a.run, dir=a.dir, max_events=count,
                  detectors=detectors, batch_size=a.batch_size,
                  n_gpu_streams=a.depth, gpu_memory_budget_gb=a.budget,
                  gpu_d2h_chunk_size=0, log_level='WARNING')
    kwargs['gpu_det'] = detectors if len(detectors) > 1 else detectors[0]
    if a.variant in ('C', 'D'):
        kwargs['gpu_bulk_read'] = a.variant == 'D'
    ds = DataSource(**kwargs)
    run = next(ds.runs())
    if ds.dsparms.batch_size != a.batch_size:
        raise RuntimeError(f'actual batch_size={ds.dsparms.batch_size}, requested={a.batch_size}')
    if timed and rank == 0:
        cache_state(a)
    comm.Barrier()
    active = timed and a.diagnostics
    profiler = cProfile.Profile() if active and is_bd else None
    if profiler is not None:
        profiler.enable()
    if a.phase_recorder is not None:
        a.phase_recorder.active = timed
        a.phase_recorder.steady = False
    if timed and is_bd and a.nsight_capture:
        import cupy as cp
        cp.cuda.profiler.start()
    if timed:
        print('PLACEMENT_BEFORE ' + json.dumps(placement()), flush=True)
        # Placement logging stays outside the event-loop timer.
        comm.Barrier()
    start = time.perf_counter()
    timestamps, checks = [], []
    for evt in run.events():
        timestamps.append(int(evt.timestamp))
        if a.phase_recorder is not None:
            a.phase_recorder.steady = True
        if check and (not timed or int(evt.timestamp) in a.reference_by_timestamp):
            gpu = evt.gpu if hasattr(evt, 'gpu') else evt
            checks.append(dict(timestamp=int(evt.timestamp),
                               raw=digest(gpu.get('jungfrau.raw').on_cpu),
                               calib=digest(gpu.get('jungfrau.calib').on_cpu)))
            if len(detectors) > 1:
                field = gpu.detector('epix100_0').field('raw', 'raw').on_cpu
                checks[-1]['epix_raw'] = {str(segment): digest(field[segment]) for segment in field.segment_ids}
    if is_bd:
        import cupy as cp
        cp.cuda.Device().synchronize()
    elapsed = time.perf_counter() - start
    if timed:
        print('PLACEMENT_AFTER ' + json.dumps(placement()), flush=True)
    if timed and is_bd and a.nsight_capture:
        cp.cuda.profiler.stop()
    phase_result = None
    if a.phase_recorder is not None:
        a.phase_recorder.active = False
        phase_result = a.phase_recorder.snapshot()
    if profiler is not None:
        profiler.disable()
        sample_pool()
        stats['pool_total_at_end'] = cp.get_default_memory_pool().total_bytes()
        stats['pool_used_at_end'] = cp.get_default_memory_pool().used_bytes()
        profile_path = Path(a.constants).parent / f'{a.case}-rank{rank}.pstats'
        profiler.dump_stats(str(profile_path))
        print(f'CPU_PROFILE rank={rank} path={profile_path}', flush=True)
    active = False
    record = dict(rank=rank, pid=os.getpid(), events=len(timestamps), timestamps=timestamps, elapsed=elapsed,
                  checks=checks, stats=stats, streams=dict(streams), subbatches=dict(subbatches),
                  residency=dict(residency), psana_path=psana.__file__, phase_timing=phase_result)
    records = comm.gather(record, root=0)
    if rank == 0:
        if timed:
            cache_state(a, prepare=False)
        bd = records[2:]
        ts = sorted(t for r in bd for t in r.pop('timestamps'))
        assert len(ts) == count and len(set(ts)) == count
        sha = hashlib.sha256(struct.pack(f'<{len(ts)}Q', *ts)).hexdigest()
        if timed:
            manifest = json.loads((Path(a.dir) / 'manifest.json').read_text())
            assert manifest['events'] == count and manifest['timestamp_sha256'] == sha
            io_bytes = sum(r['stats']['bytes'] for r in bd)
            if a.diagnostics:
                assert io_bytes == manifest['payload_bytes'], (io_bytes, manifest['payload_bytes'])
                observed = sorted((entry for r in bd for entry in r['checks']), key=lambda r: r['timestamp'])
                expected = sorted(a.reference_by_timestamp.values(), key=lambda r: r['timestamp'])
                assert observed == expected, 'CPU/GPU pixel mismatch under measured batch/budget settings'
                print(f'DIAGNOSTIC_PIXEL_PASS events={len(observed)} batch_size={a.batch_size} budget={a.budget} depth={a.depth}', flush=True)
            seconds = max(r['elapsed'] for r in records)
            result = dict(case=a.case, variant=a.variant, cache=a.cache, events=count,
                          node=os.uname().nodename,
                          timestamp_sha256=sha, loop_s=seconds, hz=count / seconds,
                          useful_gbps=manifest['payload_bytes'] / seconds / 1e9,
                          payload_bytes=manifest['payload_bytes'], batch_size=a.batch_size,
                          depth=a.depth, budget_gib=a.budget, n_bds=len(bd), diagnostics=a.diagnostics,
                          ranks=bd, timing_mode=a.timing_mode, nsight_capture=a.nsight_capture)
            print('PERF_RESULT ' + json.dumps(result, sort_keys=True), flush=True)
        elif check:
            print('CHECK_RESULT ' + json.dumps(dict(variant=a.variant, ranks=bd), sort_keys=True), flush=True)
    evt = None
    comm.Barrier()
    if is_bd and timed:
        print(f'CLEANUP_BEGIN rank={rank} time={time.strftime("%FT%T")}', flush=True)
    run.close_shared_memory()
    del run, ds
    gc.collect()
    if is_bd:
        cp.get_default_memory_pool().free_all_blocks()
    comm.Barrier()
    if is_bd and timed:
        print(f'CLEANUP_END rank={rank} time={time.strftime("%FT%T")}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', required=True)
    p.add_argument('--dir', required=True)
    p.add_argument('--exp', default='mfx101210926')
    p.add_argument('--run', type=int, default=387)
    p.add_argument('--variant', choices=list('OG'), required=True)
    p.add_argument('--case', required=True)
    p.add_argument('--events', type=int, default=10000)
    p.add_argument('--batch-size', type=int, default=20)
    p.add_argument('--depth', type=int, default=1)
    p.add_argument('--budget', type=float, default=8)
    p.add_argument('--cache', choices=['warm', 'cold'], default='warm')
    p.add_argument('--check', action='store_true')
    p.add_argument('--diagnostics', action='store_true')
    p.add_argument('--constants', required=True)
    p.add_argument('--timing-mode', choices=['off', 'cpu', 'cpu-nvtx'], default='off')
    p.add_argument('--nsight-capture', action='store_true')
    p.add_argument('--warm-cache-passes', type=int, default=1)
    a = p.parse_args()
    if a.warm_cache_passes < 1:
        p.error('warm-cache-passes must be positive')
    if a.diagnostics and a.timing_mode != 'off':
        p.error('phase timing must not be combined with cProfile diagnostics')
    if a.nsight_capture and a.timing_mode != 'cpu-nvtx':
        p.error('Nsight capture requires CPU/NVTX instrumentation')
    a.phase_recorder = None
    a.reference_by_timestamp = {}
    if a.diagnostics:
        cpu_log = Path(a.constants).parent / 'cpu-check.log'
        reference = json.loads(next(line[len('CPU_CHECK '):] for line in cpu_log.read_text().splitlines()
                                    if line.startswith('CPU_CHECK ')))
        a.reference_by_timestamp = {entry['timestamp']: entry for entry in reference}
    # Revisions differ in calibration URL handling. Freeze the fetched inputs
    # so database-client differences and availability cannot alter the workload.
    # All variants receive the same trusted, locally generated CPU-reference
    # constants. Leave RunParallel's normal MPI distribution untouched.
    from psana.psexp.run import Run
    def frozen_calibration(self):
        with gzip.open(a.constants, 'rb') as source:
            constants = pickle.load(source)
        missing = set(self.dsparms.configinfo_dict) - set(constants)
        if missing:
            raise RuntimeError(f'calibration snapshot lacks detectors: {missing}')
        self._clear_calibconst()
        self._calib_const = constants
        self.dsparms.calibconst = constants
    Run._setup_run_calibconst = frozen_calibration
    if is_bd:
        import cupy as cp
        import kvikio
        import kvikio.defaults as defaults
        import psana.dgram
        import psana.eventbuilder
        import psana.gpu.gpu_events
        import psana.gpu.gpu_kvikio_read
        modules = (psana.dgram, psana.eventbuilder, psana.gpu.gpu_events, psana.gpu.gpu_kvikio_read)
        print('RUNTIME ' + json.dumps(dict(rank=rank, psana=psana.__file__, cupy=cp.__version__,
              cuda=cp.cuda.runtime.runtimeGetVersion(), kvikio=kvikio.__version__,
              compat=bool(defaults.compat_mode()), gds_available=bool(kvikio.DriverProperties().is_gds_available),
              visibility=os.environ.get('CUDA_VISIBLE_DEVICES'), task_size=defaults.task_size(),
              nthreads=defaults.get_num_threads(),
              modules={m.__name__: dict(path=m.__file__, sha256=hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()) for m in modules})), flush=True)
        if a.diagnostics:
            hooks()
        if a.timing_mode != 'off':
            from phase_timing import install
            a.phase_recorder = install(a.variant, cp.cuda.nvtx if a.timing_mode == 'cpu-nvtx' else None)
    run_once(a, 3 if a.check else 100, check=a.check)
    if not a.check:
        if rank == 0:
            print(f'WARMUP_COMPLETE time={time.strftime("%FT%T")}', flush=True)
        run_once(a, a.events, timed=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        comm.Abort(2)
