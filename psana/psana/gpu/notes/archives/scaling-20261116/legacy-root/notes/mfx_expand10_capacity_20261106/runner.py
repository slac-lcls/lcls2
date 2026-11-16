"""One fresh-process capacity case: native GPU run, streamed export, CPU check."""
import argparse
from contextlib import closing
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from unittest.mock import patch

import numpy as np
import dense_cube
from dense_cube import Cube, adapter, STAGE2
import measure
import validate
import gpu_accept as stage2
from fixture import file_sha, require
from psana import DataSource
from psana.gpu import GpuTask
from psana.psexp.run import RunSerial

ROOT=Path('/sdf/scratch/users/m/monarin/gpu-validation/mfx-expand10/20261105-stage2-r1/pilot')
GIB=1<<30


def run(args):
    import cupy as cp
    import kvikio
    require(os.environ.get('PS_PARALLEL')=='none' and os.environ.get('PS_FAKESTEP_FLAG')=='0','Use Stage2 runtime')
    require(args.bins in (30,100,200) and 1<=args.batch_size<=16,'This campaign supports 30/100/200 bins and batches 1..16')
    args.output.mkdir(mode=0o700,exist_ok=False)
    args.local.mkdir(mode=0o700,exist_ok=False)
    report=dict(status='running',nbins=args.bins,active_bins=args.bins-1,n_gpu_streams=2,batch_size=args.batch_size,
                fixture=str(args.root),events=0,job=os.environ.get('SLURM_JOB_ID'),
                artifact_root=str(args.output),local_workspace=str(args.local),
                scope='Small native dense-cube correctness and phase profiling, not sustained throughput')
    memory=measure.Memory();task=None;start=time.monotonic()
    def save(): (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        memory.sample('baseline')
        report['runtime']=dict(host=platform.node(),python=sys.version,executable=sys.executable,cupy=cp.__version__,
                               kvikio=kvikio.__version__,gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
                               cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),KVIKIO_COMPAT_MODE=os.environ.get('KVIKIO_COMPAT_MODE'),
                               PS_FAKESTEP_FLAG=os.environ.get('PS_FAKESTEP_FLAG'))
        modules=('psana','psana.gpu.gpu_events','psana.gpu.gpu_task_batch','psana.gpu.gpu_kvikio_read')
        report['runtime_files']={name:dict(path=importlib.import_module(name).__file__,sha256=file_sha(importlib.import_module(name).__file__)) for name in modules}
        own=Path(__file__).parent
        paths=[own/n for n in ('runner.py','dense_cube.py','measure.py','validate.py','test_dense_cube.py','run.sbatch')]
        paths += [dense_cube.REFERENCE_FILE,STAGE2/'adapter.py',adapter.CALIBRATION_FILE,STAGE2/'gpu_accept.py',STAGE2/'runtime.sh',STAGE2/'validation-summary.json']
        report['code_sha256']={str(p):file_sha(p) for p in paths}
        manifest=stage2.preflight(args.root)
        seed_meta=json.loads((args.seed_cache/'manifest.json').read_text())
        stamps=[r['timestamp'] for r in manifest['schedule'] if r['service']==12]
        require(seed_meta['timestamps']==stamps and seed_meta['generation_sha256']==file_sha(args.root/'generation.json'),'Seed fixture binding mismatch')
        report['generation_sha256']=seed_meta['generation_sha256'];report['seed_cache_manifest']=seed_meta
        constants,report['calibration']=stage2.load_constants()
        task=Cube(stamps,nbins=args.bins,active_bins=args.bins-1,max_state_bytes=args.state_cap_gib*GIB,
                  memory_sample=memory.sample)
        free,total=cp.cuda.runtime.memGetInfo()
        require(args.reserve_gib>=0 and task.required_state_bytes+args.reserve_gib*GIB<=free,'Persistent banks plus reserve exceed currently free device memory')
        output_bytes=args.bins*(32*512*1024*8+128)
        require(shutil.disk_usage(args.local).free>output_bytes+2*GIB,'Insufficient node-local output space')
        report['admission']=dict(required_state_bytes=task.required_state_bytes,state_cap_bytes=args.state_cap_gib*GIB,
                                 nonstate_reserve_bytes=args.reserve_gib*GIB,device_free_bytes=free,device_total_bytes=total,
                                 output_file_bytes_estimate=output_bytes,
                                 qualification='Single assigned GPU; static application preflight, not framework-wide allocator accounting')
        t=time.monotonic();task.prepare_profile();report['kernel_compile_seconds']=time.monotonic()-t
        adapter.register_reader()
        t=time.monotonic()
        with measure.read_stats() as io:
            ds=DataSource(exp='mfx101555026',run=10,dir=str(args.root),live=False,skip_calib_load='all',gpu_det='jungfrau',
                          batch_size=args.batch_size,n_gpu_streams=2,gpu_memory_budget_gb=4,gpu_bulk_target_bytes=8<<20,
                          gpu_fn=GpuTask(task,inputs=task.inputs,calibconst=task.calibconst))
            def frozen_setup(run):
                require(run.dsparms is ds.dsparms,'Unexpected Run construction')
                run._calib_const={'jungfrau':constants};run.dsparms.calibconst=run._calib_const
            with patch.object(RunSerial,'_setup_run_calibconst',frozen_setup): run=next(ds.runs())
            report['datasource_setup_seconds']=time.monotonic()-t
            t=time.monotonic()
            with closing(run.events()) as events:
                for i,event in enumerate(events):
                    require(i<len(stamps) and int(event.timestamp)==stamps[i],'GPU event ordering mismatch')
                    require(not np.any(event.gpu.get('sim10.errors').on_cpu),'Malformed seed input')
                    report['events']+=1
                    memory.sample('event_delivered')
            report['iteration_wall_seconds']=time.monotonic()-t
        report['read_stats']=io
        require(report['events']==len(stamps) and len(task.banks)==2,'Incomplete run or unexercised stream')
        report['bank_event_counts']=[len(b.events) for b in task.banks]
        report['callback_sizes']=task.batch_sizes
        report['cuda_timing']=task.timings.summary()
        report['persistent_bytes_total']=len(task.banks)*task.state_bytes_per_stream
        memory.sample('before_export')
        report['export']=task.finish_to(args.local/'cube')
        require(task.finish_to(args.local/'cube') is report['export'] and task.export_count==1,'Non-idempotent export')
        report['export_count']=task.export_count
        report['gpu_phase_memory']=memory.summary()['peaks']
        task.close();memory.sample('after_task_close')
        # Full reference is bounded to one bin; output file reads are independent.
        report['verification']=validate.verify(args.local/'cube',args.seed_cache,constants,stamps,args.bins,args.bins-1,
                                                args.output/'retained-bins',memory.sample)
        require(report['verification']['all_pixels_within_bound'],'Numerical validation failed')
        shutil.copyfile(args.local/'cube/manifest.json',args.output/'export-manifest.json')
        report['status']='pass'
        report['temporary_output_policy']='All bins written and read back; only bins 0, active_bins-1 and last retained on shared scratch, others removed after pass.'
        save()  # Preserve validation before removing this case's generated files.
        shutil.rmtree(args.local/'cube')
        report['temporary_cube_removed']=True
    except BaseException as exc:
        report.update(status='fail',error=repr(exc));raise
    finally:
        try:
            if task is not None:task.close()
        except BaseException as exc:
            report.update(status='fail',cleanup_error=repr(exc));raise
        finally:
            memory.sample('final')
            report['memory']=memory.summary();report['elapsed_seconds']=time.monotonic()-start;save()
    print(json.dumps({k:report[k] for k in ('status','nbins','events','iteration_wall_seconds','persistent_bytes_total')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--local',type=Path,required=True);p.add_argument('--seed-cache',type=Path,required=True)
    p.add_argument('--bins',type=int,required=True);p.add_argument('--batch-size',type=int,default=7)
    p.add_argument('--state-cap-gib',type=int,default=27);p.add_argument('--reserve-gib',type=int,default=8)
    run(p.parse_args())
