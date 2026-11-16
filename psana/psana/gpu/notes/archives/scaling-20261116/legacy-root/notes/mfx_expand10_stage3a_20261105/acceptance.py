"""Native sample10 cube correctness; explicit application finalization, no benchmark."""
import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import time
from unittest.mock import patch
import numpy as np

import cube
import test_cube
import gpu_accept as stage2
from fixture import file_sha, require
from psana import DataSource
from psana.gpu import GpuTask


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=7)
    p.add_argument('--save-cube',action='store_true')
    args=p.parse_args()
    require(1<=args.batch_size<=16,'batch size 1..16 required')
    require(os.environ.get('PS_PARALLEL')=='none' and os.environ.get('PS_FAKESTEP_FLAG')=='0','Use Stage 2 runtime')
    args.output.mkdir(mode=0o700,exist_ok=False)
    report={'status':'running','fixture':str(args.root),'events_completed':0,'nbins':30,'active_bins':29,
            'batch_size':args.batch_size,'n_gpu_streams':2,'KVIKIO_COMPAT_MODE':os.environ.get('KVIKIO_COMPAT_MODE'),
            'scope':'Serial application-owned cube correctness; no managed lifecycle/performance claim',
            'code_sha256':{str(f):file_sha(f) for f in (Path(__file__),Path(cube.__file__),Path(test_cube.__file__),
                Path(stage2.__file__),cube.STAGE2/'adapter.py',cube.STAGE2/'runtime.sh',
                cube.STAGE2/'validation-summary.json',Path(__file__).with_name('run_gpu.sbatch'),stage2.adapter.CALIBRATION_FILE)}}
    start=time.monotonic(); task=None
    def save(): (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        import cupy as cp
        import kvikio
        import psana, psana.gpu.gpu_events, psana.gpu.gpu_task_batch
        report.update(host=os.uname().nodename,gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
                      cupy=cp.__version__,kvikio=kvikio.__version__,cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                      runtime_files={m.__name__:{'path':m.__file__,'sha256':file_sha(m.__file__)}
                                     for m in (psana,psana.gpu.gpu_events,psana.gpu.gpu_task_batch)})
        report['controlled_gpu_tests']=test_cube.device_tests()
        manifest=stage2.preflight(args.root)
        report['generation_sha256']=file_sha(args.root/'generation.json')
        baseline=json.loads((cube.STAGE2/'validation-summary.json').read_text())
        witnesses=[v for v in baseline['fixtures'].values() if Path(v['generation']['path'])==args.root/'generation.json']
        require(len(witnesses)==1 and report['generation_sha256']==witnesses[0]['generation']['sha256'],
                'Fixture differs from accepted Stage 2 generation')
        report['stage2_cpu_verification']=witnesses[0]['verification']
        stamps=[row['timestamp'] for row in manifest['schedule'] if row['service']==12]
        require(len(stamps)==manifest['event_count'],'Unexpected schedule')
        report['source_start_event']=manifest['source_start_event']
        constants,report['calibration']=stage2.load_constants()
        cube.adapter.register_reader()
        task=cube.Cube(stamps)
        ds=DataSource(exp='mfx101555026',run=10,dir=str(args.root),live=False,skip_calib_load='all',gpu_det='jungfrau',
                      gpu_fn=GpuTask(task,inputs=task.inputs,calibconst=task.calibconst),batch_size=args.batch_size,
                      n_gpu_streams=2,gpu_memory_budget_gb=4,gpu_bulk_target_bytes=8<<20)
        from psana.psexp.run import RunSerial
        def frozen_setup(run):
            require(run.dsparms is ds.dsparms,'Unexpected calibration constructor')
            run._calib_const={'jungfrau':constants}; run.dsparms.calibconst=run._calib_const
        with patch.object(RunSerial,'_setup_run_calibconst',frozen_setup): run=next(ds.runs())
        with closing(run.events()) as events:
            for i,evt in enumerate(events):
                require(i<len(stamps) and int(evt.timestamp)==stamps[i],'Event order mismatch')
                require(not np.any(evt.gpu.get('sim10.errors').on_cpu),'Invalid seed panel')
                report['events_completed']+=1
                if i%100==0: save(); print('GPU cube event',i,flush=True)
        require(report['events_completed']==len(stamps),'Lost native events')
        # EOF is proven by exhausting the iterator. finish also joins each stream.
        actual,counts=task.finish()
        report['banks']=[{'event_indices':b.events,'step_labels':sorted(b.step_labels)} for b in task.banks]
        report.update(callback_batch_sizes=task.batch_sizes,persistent_bytes_per_stream=task.state_bytes_per_stream,
                      persistent_bytes_total=len(task.banks)*task.state_bytes_per_stream,
                      cupy_pool_peak_bytes=task.pool_peak_bytes,export_count=task.export_count)
        require(task.finish()[0] is actual and task.export_count==1,'Duplicate export')
        task.close()
        # Independent CPU source class calibrates expanded seed input using CPU v3.
        source=DataSource(exp='mfx101555026',run=10,dir='/sdf/data/lcls/ds/prj/public01/xtc',
                          max_events=1,live=False,skip_calib_load='all')
        src_run=next(source.runs()); det=src_run.Detector('jungfrau',status=False); det.raw._calibconst=constants
        cpu_ds=DataSource(exp='mfx101555026',run=10,dir=str(args.root),live=False,skip_calib_load='all')
        cpu_run=next(cpu_ds.runs()); reader=cpu_run.Detector('jungfrau').sim10
        expected=np.zeros_like(actual); absolute=np.zeros_like(actual); cpu_counts=np.zeros(30,np.uint64)
        raw=np.empty((32,512,1024),np.uint16)
        with closing(src_run.events()) as originals:
            orig=next(originals)
            for i,evt in enumerate(cpu_run.events()):
                require(i<len(stamps) and int(evt.timestamp)==stamps[i],'CPU event identity mismatch')
                seeds=reader.seed(evt); require(set(seeds)==set(range(32)),'Missing CPU seed')
                for panel,seed in seeds.items(): raw[panel]=stage2.expand(seed)
                image=det.raw.calib(orig,cversion=3,cmpars=None,nda_raw=raw).reshape(raw.shape)
                require(np.isfinite(image).all(),'Nonfinite input to cube')
                b=i%29; expected[b]+=image; absolute[b]+=np.abs(image); cpu_counts[b]+=1
                if i%100==0: print('CPU cube event',i,flush=True)
        require(int(cpu_counts.sum())==len(stamps) and np.array_equal(counts,cpu_counts),'Bin count mismatch')
        report['counts']=counts.tolist(); report['bins']=[]
        eps=np.finfo(np.float32).eps/2
        # Each per-stream float32 sum has <=n additions; host merge uses float64.
        # gamma_n sum|x| bounds arbitrary scheduling/cancellation. Absolute floor
        # covers negligible float64 reference/merge rounding near zero.
        for b in range(30):
            n=int(counts[b]); gamma=n*eps/(1-n*eps)
            error=np.abs(actual[b]-expected[b]); bound=gamma*absolute[b]+1e-5
            require(np.isfinite(actual[b]).all() and np.all(error<=bound),'Cube sum exceeds rounding bound')
            if n==0: require(not np.any(actual[b]),'Nonzero empty bin')
            report['bins'].append({'bin':b,'count':n,'max_abs_error':float(error.max()),
                                   'max_bound_ratio':float(np.max(error/bound)),
                                   'actual_sha256':stage2.digest(actual[b]),'reference_sha256':stage2.digest(expected[b])})
        require(len(report['banks'])==2,'Both CUDA streams were not exercised')
        require(report['persistent_bytes_total']<=5<<30,'Persistent cap exceeded')
        report['tolerance']='abs(error) <= gamma_n * sum(abs(float32 calibrated input)) + 1e-5; gamma_n=n*u/(1-n*u), u=2^-24'
        report['compared_cube_pixels']=int(actual.size)
        report['all_counts_exact']=True; report['all_cube_pixels_within_bound']=True
        report['published_per_event']='sim10.errors only; no full raw/calibration D2H'
        report['final_merge']='CPU float64, one bank/bin D2H at a time after stream synchronization'
        if args.save_cube:
            np.save(args.output/'cube.npy',actual); np.save(args.output/'counts.npy',counts)
            report['cube_file']={'path':str(args.output/'cube.npy'),'sha256':file_sha(args.output/'cube.npy')}
        report['status']='pass'
    except BaseException as exc:
        report.update(status='fail',error_type=type(exc).__name__,error=str(exc)); raise
    finally:
        try:
            if task is not None: task.close()
        except BaseException as cleanup:
            report.update(status='fail',cleanup_error=repr(cleanup))
            raise
        finally:
            report['elapsed_seconds']=time.monotonic()-start; save()


if __name__=='__main__': main()
