"""Fresh MPI calibration sample; identical observation wrappers for A/B/C."""
import argparse
from collections import Counter
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
os.environ['CUDA_VISIBLE_DEVICES'] = '0' if rank_hint >= 2 else ''
os.sched_setaffinity(0, {int(x) for x in os.environ['BENCH_CPU_AFFINITY'].split(',')})
from mpi4py import MPI
import numpy as np
from psana import DataSource
from psana.psexp.run import Run
from psana.gpu import GpuTask
from psana.gpu.examples.jungfrau_calibration import JungfrauCalibration
from check_runtime_provenance import verify_install
comm = MPI.COMM_WORLD
print('RUNTIME_PROVENANCE '+json.dumps(verify_install(os.environ['BENCH_INSTALL'])), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--directory', required=True)
    ap.add_argument('--variant', choices=list('ABC'), required=True)
    ap.add_argument('--events', type=int, required=True)
    ap.add_argument('--diagnostic', action='store_true')
    args = ap.parse_args()
    worker = comm.rank >= 2
    with gzip.open(args.root/'constants.pkl.gz', 'rb') as f:
        constants = pickle.load(f)['jungfrau']
    refs = {r['timestamp']: r for r in json.loads((args.root/'pixels.json').read_text())}
    user = JungfrauCalibration(use_offset=True, status_bits=(1 << 64)-1)
    requested_bytes = sum(constants[k][0].nbytes for _, k in user.calibconst)

    def calibration(self):
        self._clear_calibconst()
        self._calib_const = {name: {} for name in self.dsparms.configinfo_dict}
        self._calib_const['jungfrau'] = constants
        self.dsparms.calibconst = self._calib_const
    Run._setup_run_calibconst = calibration
    counters = dict(bytes=0, requests=0, read_wait_s=0., peak_owned_and_held=0,
                    budget_limit=0, cpu_bd_reads=0)
    checks, snapshots, managers = [], [], []
    callback_sizes = Counter()
    if worker:
        import cupy as cp
        import kvikio
        import kvikio.defaults
        from psana.gpu.gpu_budget import _GpuBudget
        from psana.gpu.gpu_events import GpuEventManager
        from psana.gpu.gpu_kvikio_read import KvikioGpuReader
        from psana.psexp.event_manager import EventManager
        assert cp.cuda.runtime.getDeviceCount() == 1
        assert kvikio.__version__ == '26.10.00' and cp.__version__ == '14.0.1'
        assert int(kvikio.defaults.get('num_threads')) == 8
        assert int(kvikio.defaults.get('task_size')) == 1 << 20
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
    ds = None

    def callback(batch, stream):
        out, _ = user.calibrate(batch, stream, publish=False)
        callback_sizes[len(batch.timestamps)] += 1
        if not snapshots:
            assert len(managers) == 1
            m = managers[0]
            budget = m._gpu_budget
            tc = m._task_constants
            placement = getattr(m, '_placement', None)
            if args.variant in ('B', 'C'):
                assert placement is not None and placement is run._gpu_placement
                if m._n_bd_per_gpu > 1:
                    assert tc._placement is placement
            shared = int(getattr(tc, 'shared_bytes', 0))
            imported = int(getattr(tc, 'imported_bytes', 0))
            owned = sum(int(a['capacity']) for a in budget.allocation_snapshot()
                        if a.get('category') == 'task-constants') + max(0, shared-imported)
            bus = cp.cuda.runtime.deviceGetPCIBusId(0)
            if isinstance(bus, bytes):bus = bus.decode()
            snap = dict(cupy=cp.__version__, kvikio=kvikio.__version__, pci=bus,
                cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                free_bytes=int(cp.cuda.runtime.memGetInfo()[0]),
                pool_bytes=int(cp.get_default_memory_pool().total_bytes()),
                sharing=getattr(placement, 'sharing', 'private'),
                peers=m._n_bd_per_gpu, is_owner=getattr(placement, 'is_owner', None),
                shared_bytes=shared, imported_bytes=imported, owned_constants_bytes=owned,
                usable_bytes=getattr(placement, 'usable_bytes', None),
                budget_limit=budget.limit(), budget_committed=budget.committed(), budget_held=budget._held,
                subbatch_budget_bytes=m._subbatch_budget_bytes)
            assert owned+imported == requested_bytes, (snap, requested_bytes)
            if args.diagnostic:
                before = (budget.limit(), budget.committed(), budget._held, m._subbatch_budget_bytes)
                report = m._snapshot_memory('live-calibration-diagnostic')
                m.log_memory('live-calibration-diagnostic')
                after = (budget.limit(), budget.committed(), budget._held, m._subbatch_budget_bytes)
                assert before == after, (before, after)
                snap['report'] = {k:getattr(report,k) for k in
                    ('committed','held','retained','borrowed','raw_input','xtc_config','xtc_slots','input_bytes')}
                # PR208 introduces this field; null records its absence in A/B.
                snap['report']['constants'] = report.constants if args.variant == 'C' else None
                snap['reporting_preserves_budget'] = True
                if args.variant == 'C':
                    assert report.constants == owned and report.borrowed == imported
                    residual = report.committed-report.raw_input-report.xtc_config-report.xtc_slots-sum(report.input_bytes.values())-owned
                    assert residual >= 0 and report.retained == residual
            snapshots.append(snap)
        if args.diagnostic:
            for i, ts in enumerate(batch.timestamps):
                if ts in refs:
                    stream.synchronize()
                    order = np.argsort(batch.segment_ids('jungfrau'))
                    actual = np.ascontiguousarray(out[i].get()[order])
                    actual[actual == 0] = 0
                    actual[np.isnan(actual)] = np.nan
                    digest = hashlib.sha256(actual.tobytes()).hexdigest()
                    assert digest == refs[ts]['calib']['sha256'], (int(ts), digest, refs[ts]['calib'])
                    checks.append(int(ts))

    started = time.perf_counter()
    ds = DataSource(exp='mfx101210926', run=387, dir=args.directory,
        detectors=['jungfrau'], gpu_det=['jungfrau'],
        gpu_fn=GpuTask(callback, inputs=user.inputs, calibconst=user.calibconst),
        max_events=args.events, batch_size=20, n_gpu_streams=1,
        gpu_memory_budget_gb=0, gpu_bulk_read=True, gpu_bulk_target_bytes=1 << 20,
        gpu_d2h_chunk_size=0, skip_calib_load='all', log_level='INFO')
    run = next(ds.runs())
    setup_s = time.perf_counter()-started
    role = run.comms.node_type()
    comm.Barrier()
    stamps = []
    first_s = None
    began = time.perf_counter()
    for evt in run.events():
        if first_s is None:first_s = time.perf_counter()-began
        stamps.append(int(evt.timestamp))
    if worker:cp.cuda.Device().synchronize()
    elapsed = time.perf_counter()-began
    if worker:
        assert managers and all(m._closed for m in managers)
        assert all(m._gpu_budget._held == 0 and not m.gpu_reader._pending for m in managers)
        assert all(not m.event_pool.active_count for m in managers)
        assert counters['cpu_bd_reads'] == 0
    row = dict(rank=comm.rank, role=role, timestamps=stamps, events=len(stamps),
        setup_s=setup_s, loop_s=elapsed, first_s=first_s,
        callback_events=user.events, callback_calls=user.calls, checks=checks,
        callback_sizes=dict(callback_sizes), counts=counters,
        snapshots=snapshots, affinity=sorted(os.sched_getaffinity(0)))
    rows = comm.gather(row, root=0)
    if comm.rank == 0:
        bds = [r for r in rows if r['role']=='bd']
        assert len(bds)==comm.size-2
        assert all(r['events'] > 0 and r['events']==r['callback_events'] for r in bds)
        assert all(sum(int(k)*v for k,v in r['callback_sizes'].items()) == r['events'] for r in bds)
        assert all(not r['events'] for r in rows if r['role'] != 'bd')
        all_stamps = sorted(t for r in bds for t in r['timestamps'])
        assert len(all_stamps)==len(set(all_stamps))==args.events
        reference = json.loads((args.root/'reference.json').read_text())[str(args.events)]
        digest = hashlib.sha256(struct.pack(f'<{len(all_stamps)}Q', *all_stamps)).hexdigest()
        assert digest==reference['timestamp_sha256']
        assert sum(r['counts']['bytes'] for r in bds)==reference['payload_bytes']
        assert sum(r['counts']['requests'] for r in bds)==args.events*5
        checked = sorted(t for r in bds for t in r['checks'])
        if args.diagnostic:
            assert checked==sorted(t for t in refs if t in all_stamps) and len(checked)>0
        live = [r['snapshots'][0] for r in bds]
        if args.diagnostic:
            print('CALIBRATION_LIVE_CHECK '+json.dumps(dict(variant=args.variant, bds=len(bds), live=live)), flush=True)
        assert all(s['peers']==len(bds) for s in live)
        assert len({s['pci'] for s in live})==1
        if args.variant in ('B','C') and len(bds)>1:
            assert all(s['sharing']=='on' for s in live)
            assert sum(s['imported_bytes']>0 for s in live)==len(bds)-1
            assert sum(s['owned_constants_bytes'] for s in live)==requested_bytes
        else:
            assert all(s['imported_bytes']==0 and s['owned_constants_bytes']==requested_bytes for s in live)
        wall = max(r['loop_s'] for r in rows)
        result = dict(variant=args.variant, bds=len(bds), events=args.events,
            mpi=MPI.Get_library_version(), python=os.sys.version,
            diagnostic=args.diagnostic, loop_s=wall, events_per_s=args.events/wall,
            bytes=reference['payload_bytes'], requests=args.events*5,
            timestamp_sha256=digest, setup_s=max(r['setup_s'] for r in rows),
            first_event_s=max(r['first_s'] for r in bds),
            constants_bytes=requested_bytes, rows=rows)
        print('PERF_RESULT '+json.dumps(result), flush=True)
    comm.Barrier()
    evt = None
    run.close_shared_memory()
    managers.clear()
    del run, ds
    gc.collect()
    comm.Barrier()


if __name__ == '__main__':
    try:main()
    except BaseException:
        import traceback
        traceback.print_exc()
        comm.Abort(1)
        raise
