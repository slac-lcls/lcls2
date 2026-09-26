"""One fresh MPI Jungfrau-only sample; pin CUDA before importing MPI/psana."""
import argparse
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import struct
import time

rank_hint = int(os.environ['OMPI_COMM_WORLD_RANK'])
size_hint = int(os.environ['OMPI_COMM_WORLD_SIZE'])
ngpus = int(os.environ['SLURM_GPUS_ON_NODE'])
is_bd = rank_hint >= 2
physical_gpu = (rank_hint - 2) % ngpus if is_bd else None
os.environ['CUDA_VISIBLE_DEVICES'] = str(physical_gpu) if is_bd else ''
if os.environ.get('BENCH_CPU_AFFINITY'):
    os.sched_setaffinity(0, {int(x) for x in os.environ['BENCH_CPU_AFFINITY'].split(',')})

from mpi4py import MPI
import psana
from psana import DataSource
from psana.psexp.run import Run
from common import digest

assert Path(psana.__file__).resolve().parent.parent == Path(__file__).resolve().parents[2]/'python'
comm = MPI.COMM_WORLD
assert (comm.rank, comm.size) == (rank_hint, size_hint)
assert os.environ['PS_EB_NODES'] == '1' and os.environ['PS_SRV_NODES'] == '0'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', required=True)
    p.add_argument('--reference', required=True)
    p.add_argument('--constants', required=True)
    p.add_argument('--pixels', required=True)
    p.add_argument('--bulk', choices=('on', 'off'), required=True)
    p.add_argument('--events', type=int, default=10000)
    p.add_argument('--check-pixels', action='store_true')
    p.add_argument('--include-feespec', action='store_true')
    a = p.parse_args()
    reference = json.loads(Path(a.reference).read_text())
    pixels = {r['timestamp']: r for r in json.loads(Path(a.pixels).read_text())}
    if a.include_feespec:
        from feespec import exclusive_feespec
        exclusive_feespec(comm.rank)

    def calibration(self):
        self._clear_calibconst()
        self._calib_const = {name: {} for name in self.dsparms.configinfo_dict}
        with gzip.open(a.constants, 'rb') as source:
            self._calib_const['jungfrau'] = pickle.load(source)['jungfrau']
        self.dsparms.calibconst = self._calib_const
    Run._setup_run_calibconst = calibration

    counters = dict(bytes=0, requests=0, read_wait_s=0., peak_owned_and_held=0,
                    budget_limit=0, cpu_bd_reads=0)
    managers = []
    if is_bd:
        import cupy as cp
        import kvikio
        import kvikio.defaults
        from psana.gpu.gpu_budget import _GpuBudget
        from psana.gpu.gpu_events import GpuEventManager
        from psana.gpu.gpu_kvikio_read import KvikioGpuReader
        from psana.psexp.event_manager import EventManager
        wait = KvikioGpuReader.wait_batch
        def waited(self, pending):
            fresh = not pending.completed
            before = self._total_bytes_read, self._total_io_ns
            result = wait(self, pending)
            if fresh:
                counters['requests'] += len(pending.futures)
                counters['bytes'] += self._total_bytes_read - before[0]
                counters['read_wait_s'] += (self._total_io_ns - before[1]) / 1e9
            return result
        KvikioGpuReader.wait_batch = waited
        for method in ('reserve', 'hold'):
            original = getattr(_GpuBudget, method)
            def charged(self, *args, original=original, **kwargs):
                result = original(self, *args, **kwargs)
                used = self.committed() + self._held
                assert used <= self.limit()
                counters['peak_owned_and_held'] = max(counters['peak_owned_and_held'], used)
                counters['budget_limit'] = self.limit()
                return result
            setattr(_GpuBudget, method, charged)
        setup = GpuEventManager._setup_gpu_pipeline
        def configured(self, *args, **kwargs):
            setup(self, *args, **kwargs)
            managers.append(self)
        GpuEventManager._setup_gpu_pipeline = configured
        cpu_read = EventManager._read
        def counted_cpu_read(self, *args, **kwargs):
            counters['cpu_bd_reads'] += 1
            return cpu_read(self, *args, **kwargs)
        EventManager._read = counted_cpu_read
        assert cp.cuda.runtime.getDeviceCount() == 1
        bus = cp.cuda.runtime.deviceGetPCIBusId(0)
        if isinstance(bus, bytes):
            bus = bus.decode()
        device = dict(bus=bus, total_bytes=cp.cuda.Device().mem_info[1],
                      cupy=cp.__version__, kvikio=kvikio.__version__,
                      gds_available=bool(kvikio.DriverProperties().is_gds_available),
                      workers=int(kvikio.defaults.get_num_threads()),
                      task_bytes=int(kvikio.defaults.task_size()))
    else:
        device = None

    selected = ['jungfrau', 'feespec'] if a.include_feespec else ['jungfrau']
    ds = DataSource(exp='mfx101210926', run=387, dir=a.directory,
        detectors=selected, gpu_det=selected, gpu_bulk_read=a.bulk == 'on',
        gpu_bulk_target_bytes=1 << 20, max_events=a.events, batch_size=20,
        n_gpu_streams=1, gpu_memory_budget_gb=0, gpu_d2h_chunk_size=0,
        skip_calib_load='all', log_level='ERROR')
    run = next(ds.runs())
    feespec_arrays = []
    if is_bd and a.include_feespec:
        # Compact validation result only; copied to CPU after the timed loop.
        feespec_sums = cp.empty(a.events, dtype=cp.int64)
        cp.cuda.Device().synchronize()
    comm.Barrier()
    stamps, checks = [], []
    start = time.perf_counter()
    for evt in run.events():
        timestamp = int(evt.timestamp)
        stamps.append(timestamp)
        if a.include_feespec:
            field = evt.gpu.detector('feespec').field('raw', 'hproj')
            with field.on_gpu_view(cp.cuda.Stream.null) as segments:
                values = segments[0]
                assert values.shape == (2048,) and values.dtype == cp.int32
                feespec_sums[len(stamps)-1] = cp.sum(values, dtype=cp.int64)
                if a.check_pixels:
                    feespec_arrays.append(dict(timestamp=timestamp, digest=digest(values.get())))
            field = segments = values = None
        if a.check_pixels and timestamp in pixels:
            checks.append(dict(timestamp=timestamp,
                raw=digest(evt.gpu.get('jungfrau.raw').on_cpu),
                calib=digest(evt.gpu.get('jungfrau.calib').on_cpu)))
    if is_bd:
        cp.cuda.Device().synchronize()
    elapsed = time.perf_counter() - start
    if is_bd:
        assert managers and all(m._closed for m in managers)
        assert all(m._gpu_budget._held == 0 and not m.gpu_reader._pending for m in managers)
        assert all(not m.event_pool.active_count for m in managers)
    record = dict(rank=comm.rank, is_bd=is_bd, timestamps=stamps, checks=checks,
        loop_s=elapsed, device=device, physical_gpu=physical_gpu, counts=counters,
        affinity=sorted(os.sched_getaffinity(0)), pid=os.getpid(),
        peers=[m._n_bd_per_gpu for m in managers])
    if a.include_feespec:
        record.update(feespec_sums=feespec_sums[:len(stamps)].get().tolist() if is_bd else [],
                      feespec_arrays=feespec_arrays)
    records = comm.gather(record, root=0)
    if comm.rank == 0:
        from contract import validate_result
        result = validate_result(records, reference[str(a.events)], pixels,
            events=a.events, ngpus=ngpus, check_pixels=a.check_pixels,
            expected_requests=reference[str(a.events)]['requests'][a.bulk] if a.include_feespec else None)
        if a.include_feespec:
            from feespec import validate_feespec
            result.update(validate_feespec(records, reference[str(a.events)]['feespec'], a.check_pixels))
        result.update(bulk=a.bulk, diagnostic=a.check_pixels, ranks=records,
                      include_feespec=a.include_feespec)
        print('JF_SCALE_RESULT ' + json.dumps(result, sort_keys=True), flush=True)
    # All GPU work and peer checks finish before shared windows are freed.
    comm.Barrier()
    evt = None
    run.close_shared_memory()
    managers.clear()
    del run, ds
    gc.collect()
    comm.Barrier()


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        comm.Abort(1)
        raise
