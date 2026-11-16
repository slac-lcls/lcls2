"""One fresh-process, one-BD sample10 throughput/profile case. No MPI campaign."""
import argparse
from contextlib import closing, nullcontext
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from unittest.mock import patch

from perf_cube import Cube, dense_cube
from timing import Recorder, admission_counts
from fixture_check import ROOT, check_10k
import measure
import gpu_accept as stage2
from fixture import file_sha, require
from psana import DataSource
from psana.gpu import GpuTask
from psana.psexp.run import RunSerial

GIB = 1 << 30
PILOT = Path('/sdf/scratch/users/m/monarin/gpu-validation/mfx-expand10/20261105-stage2-r1/pilot')


def run(args):
    import cupy as cp
    import kvikio
    require(os.environ.get('PS_PARALLEL') == 'none' and os.environ.get('PS_FAKESTEP_FLAG') == '0', 'Use the reviewed runtime')
    require(args.streams in (1,2) and 1 <= args.batch_size <= 16 and 2 <= args.bins <= 200, 'Unsupported bounded case')
    require(1 <= args.framework_budget_gib <= 8 and 1 <= args.bulk_target_bytes <= 8<<20, 'Unsupported input budget/target')
    require(args.reserve_gib >= 8 and args.state_cap_gib > 0, 'Keep at least 8 GiB non-state reserve')
    args.output.mkdir(mode=0o770, exist_ok=False)
    report = dict(status='running', mode=args.mode, host=platform.node(), job=os.environ.get('SLURM_JOB_ID'),
                  fixture=str(args.root), bins=args.bins, active_bins=args.bins-1, streams=args.streams,
                  batch_size=args.batch_size, requested_warmup=args.warmup, delivered_events=0)
    task = None
    memory = measure.Memory()
    start = time.perf_counter()
    def save():
        temp = args.output/'report.json.partial'
        temp.write_text(json.dumps(report, indent=2)+'\n')
        temp.replace(args.output/'report.json')
    save()
    try:
        memory.sample('before_setup')
        report['runtime'] = dict(python=sys.version, executable=sys.executable, cupy=cp.__version__, kvikio=kvikio.__version__,
                                gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
                                cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                                environment={k:os.environ.get(k) for k in ('PS_PARALLEL','PS_FAKESTEP_FLAG','PS_SMD_N_EVENTS','PS_SMD_CHUNKSIZE',
                                             'KVIKIO_COMPAT_MODE','KVIKIO_NTHREADS','CUDA_VISIBLE_DEVICES')})
        modules = ('psana','psana.gpu.gpu_events','psana.gpu.gpu_task_batch','psana.gpu.gpu_kvikio_read',
                   'psana.gpu.gpu_stream','psana.gpu.gpudgram.batch','psana.psexp.event_manager')
        report['runtime_files'] = {name:dict(path=importlib.import_module(name).__file__, sha256=file_sha(importlib.import_module(name).__file__)) for name in modules}
        paths = list(Path(__file__).parent.glob('*.py')) + [dense_cube.REFERENCE_FILE, Path(dense_cube.__file__),
                 dense_cube.STAGE2/'adapter.py', dense_cube.adapter.CALIBRATION_FILE,
                 dense_cube.STAGE2/'gpu_accept.py', dense_cube.STAGE2/'runtime.sh', Path(measure.__file__),
                 Path(stage2.fixture.__file__), Path(stage2.verify.__file__),
                 Path(importlib.import_module('xtc_fixture').__file__)]
        report['code_sha256'] = {str(p):file_sha(p) for p in paths}
        t = time.perf_counter()
        if args.root.resolve() == PILOT.resolve():
            manifest = stage2.preflight(args.root)  # Original full-hash/schema check.
            report['fixture_check'] = dict(validation='original_1k_full_hash_schema_preflight',
                                           generation_sha256=file_sha(args.root/'generation.json'), cache='verification warms cache')
        else:
            manifest, report['fixture_check'] = check_10k(args.root, file_sha, full_hash=args.verify_files)
        stamps = [r['timestamp'] for r in manifest['schedule'] if r['service'] == 12]
        constants, report['calibration'] = stage2.load_constants()
        report['fixture_calibration_preflight_seconds'] = time.perf_counter()-t
        recorder = Recorder(args.mode == 'profile')
        task = Cube(stamps, warmup=args.warmup, recorder=recorder, nbins=args.bins,
                    active_bins=args.bins-1, max_streams=args.streams, max_state_bytes=args.state_cap_gib*GIB)
        free,total = cp.cuda.runtime.memGetInfo()
        require(task.required_state_bytes + args.reserve_gib*GIB <= free, 'Persistent banks plus reserve exceed free GPU memory')
        report['admission'] = dict(state_bytes=task.required_state_bytes, nonstate_reserve_bytes=args.reserve_gib*GIB,
                                   free_device_bytes=free, total_device_bytes=total, framework_budget_gib=args.framework_budget_gib,
                                   bulk_target_bytes=args.bulk_target_bytes)
        t = time.perf_counter()
        task.prepare()
        report['kernel_prepare_seconds'] = time.perf_counter()-t
        dense_cube.adapter.register_reader()
        t = time.perf_counter()
        ds = DataSource(exp='mfx101555026', run=10, dir=str(args.root), live=False, skip_calib_load='all', gpu_det='jungfrau',
                        batch_size=args.batch_size, n_gpu_streams=args.streams,
                        gpu_memory_budget_gb=args.framework_budget_gib, gpu_bulk_target_bytes=args.bulk_target_bytes,
                        gpu_fn=GpuTask(task, inputs=task.inputs, calibconst=task.calibconst))
        def frozen_setup(run):
            require(run.dsparms is ds.dsparms, 'Unexpected run construction')
            run._calib_const = {'jungfrau':constants}
            run.dsparms.calibconst = run._calib_const
        with patch.object(RunSerial, '_setup_run_calibconst', frozen_setup):
            run = next(ds.runs())
        report['datasource_setup_seconds'] = time.perf_counter()-t
        memory.sample('before_iteration')
        t = time.perf_counter()
        with admission_counts(task.window) as pressure, recorder.framework(), (measure.read_stats() if recorder.enabled else nullcontext(None)) as io:
            with closing(run.events()) as events:
                for event in events:
                    i = report['delivered_events']
                    require(i < len(stamps) and int(event.timestamp) == stamps[i], 'Event ordering mismatch')
                    report['delivered_events'] = i+1
            eof = time.perf_counter()
            task.join()
            report['explicit_final_join_seconds'] = time.perf_counter()-eof
            report['window'] = task.window.finish()
        report['iteration_completed_seconds'] = time.perf_counter()-t
        report['admission_pressure_trims'] = pressure
        require(report['delivered_events'] == len(stamps), 'Incomplete delivered events')
        memory.sample('after_completion')
        report['checks'] = task.check_counts()
        report['callbacks'] = task.callbacks
        report['bank_events'] = [len(b.events) for b in task.banks]
        report['read_stats'] = io
        if recorder.enabled:
            (args.output/'profile.json').write_text(json.dumps(recorder.summary(), indent=2)+'\n')
        report['timing_contract'] = ('Warmup rounded up to a whole callback; prior bank streams joined once before next callback. '
                                    'Steady wall ends after EOF and bank joins, includes tail delivery/drain. Input for boundary batch may already be prefetched/parsed. '
                                    'Whole iteration includes startup warmup. No per-event D2H, CUDA timing events or framework timer patches in throughput mode. '
                                    'Both modes count pressure-path trims without timers; nonzero counts indicate forced pool drains before retries. '
                                    'Profile mode is diagnostic, not the throughput baseline.')
        if args.export is not None:
            import shutil
            require(shutil.disk_usage(args.export.parent).free >= args.bins*(32*512*1024*8+128)+2*GIB, 'Insufficient export space')
            report['export'] = task.finish_to(args.export)
        report['status'] = 'pass'
    except BaseException as exc:
        report.update(status='fail', error=repr(exc))
        raise
    finally:
        try:
            if task is not None:
                task.close()
        except BaseException as exc:
            report.update(status='fail', cleanup_error=repr(exc))
            raise
        finally:
            memory.sample('after_cleanup')
            report['memory'] = memory.summary()
            report['memory']['interpretation'] = ('Device/CuPy memory sampled only outside iteration, not continuous peaks. '
                                                  'VmHWM is process-lifetime host RSS including setup. Device usage may include other processes.')
            report['elapsed_seconds'] = time.perf_counter()-start
            save()
    print(json.dumps({k:report[k] for k in ('status','mode','delivered_events','window')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('throughput','profile'), default='throughput')
    parser.add_argument('--bins', type=int, default=100)
    parser.add_argument('--streams', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=7)
    parser.add_argument('--warmup', type=int, default=1000)
    parser.add_argument('--state-cap-gib', type=int, default=27)
    parser.add_argument('--reserve-gib', type=int, default=8)
    parser.add_argument('--framework-budget-gib', type=int, default=4)
    parser.add_argument('--bulk-target-bytes', type=int, default=512<<10)
    parser.add_argument('--verify-files', action='store_true')
    parser.add_argument('--export', type=Path, help='Optional new full-cube directory; outside steady timing')
    run(parser.parse_args())
