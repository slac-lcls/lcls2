"""Native GPU pipeline acceptance; every raw/calibrated pixel checked on CPU.

This intentionally copies full results to host and is not a throughput test.
"""
import argparse
import ast
from contextlib import closing
from unittest.mock import patch
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from psana import DataSource
from psana.gpu import GpuTask

import adapter
import fixture
import verify
from fixture import META, PANELS, SNAPSHOT, file_sha, read_record, require, schemas
from verify import expand


def digest(a):
    return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest()


def load_constants():
    manifest = json.loads((SNAPSHOT/'manifest.json').read_text())
    result, provenance = {}, {}
    for item in manifest['calibration']:
        doc = item['source_document']
        if item['detector'] != 'jungfrau' or doc['ctype'] not in ('pedestals', 'pixel_gain', 'pixel_offset'):
            continue
        a = np.memmap(SNAPSHOT/item['file'], dtype=doc['data_dtype'], mode='r',
                      shape=ast.literal_eval(doc['data_shape']))
        require(a.shape == (3,32,512,1024) and digest(a) == item['sha256'], 'Frozen calibration mismatch')
        result[doc['ctype']] = (a, doc)
        provenance[doc['ctype']] = {'id':doc['_id'], 'shape':list(a.shape), 'dtype':str(a.dtype), 'sha256':item['sha256']}
    require(len(result) == 3, 'Missing constants')
    return result, provenance


def preflight(root):
    from psana import dgram
    manifest = json.loads((root/'generation.json').read_text())
    require(1 <= manifest['event_count'] <= 1000 and len(manifest['streams']) == 10, 'Invalid fixture')
    for item in manifest['streams']:
        for record in item['files'].values():
            p = Path(record['path'])
            require(p.parent in (root, root/'smalldata') and file_sha(p) == record['sha256'], 'Fixture digest mismatch')
        stream = item['stream']
        with Path(item['files']['bd']['path']).open('rb') as f:
            raw = read_record(f,0)
        cfg = dgram.Dgram(view=raw)
        names = [v for v in schemas(raw).values() if v['detname']=='jungfrau' and v['alg']=='sim10']
        require([n['segment'] for n in names] == PANELS.get(stream,[]), 'Wrong physical panel mapping')
        require(all(n['version']==65536 and n['fields'][0]==('seed',1,1) for n in names), 'Bad sim10 schema')
        if names:
            require(all(getattr(cfg.sim10meta[stream].config,k)==v for k,v in META.items()), 'Bad sample10 metadata')
    return manifest


def device_guards(cp):
    """Valid unaligned seed and malformed descriptor cases, on actual hardware."""
    a = adapter
    seed = np.arange(a.WORDS, dtype=np.uint16)
    storage = cp.asarray(np.concatenate([np.array([255],np.uint8), seed.view(np.uint8)]))
    kernel = cp.RawKernel(a.kernel_source(), 'expand10', options=('--std=c++17',))
    cases = ['valid_odd_offset','missing','null_base','wrong_field_type','wrong_field_rank',
             'missing_locator','wrong_shape','wrong_size','out_of_bounds','offset_overflow']
    results = []
    for name in cases:
        loc = np.zeros((1,a.LOC_NCOLS), np.uint64)
        loc[0,[a.LOC_TYPE,a.LOC_RANK,a.LOC_DIM0,a.LOC_OFFSET,a.LOC_NBYTES,a.LOC_STATUS]] = [1,1,a.WORDS,1,a.WORDS*2,a.STATUS_FOUND]
        f = np.array([[storage.data.ptr,storage.nbytes,0,0,1,1,2,1]],np.uint64)
        if name=='missing': f[0,a.FIELD_SOURCE_PRESENT]=0
        elif name=='null_base': f[0,a.FIELD_RAW_PTR]=0
        elif name=='wrong_field_type': f[0,a.FIELD_TYPE]=3
        elif name=='wrong_field_rank': f[0,a.FIELD_RANK]=2
        elif name=='missing_locator': loc[0,a.LOC_STATUS]=0
        elif name=='wrong_shape': loc[0,a.LOC_DIM0]=a.WORDS-1
        elif name=='wrong_size': loc[0,a.LOC_NBYTES]=a.WORDS*2-1
        elif name=='out_of_bounds': f[0,a.FIELD_RAW_NBYTES]=storage.nbytes-1
        elif name=='offset_overflow': loc[0,a.LOC_OFFSET]=np.iinfo(np.uint64).max
        loc_gpu=cp.asarray(loc); f[0,a.FIELD_LOCATOR_PTR]=loc_gpu.data.ptr
        f_gpu=cp.asarray(f)
        raw=cp.empty((1,a.PIXELS),cp.uint16); present=cp.empty(1,cp.uint8); errors=cp.empty(1,cp.uint32)
        kernel(((a.PIXELS+255)//256,), (256,), (f_gpu,raw,present,errors,np.uint64(1)))
        got=raw.get(); code=int(errors.get()[0]); found=int(present.get()[0])
        if name=='valid_odd_offset':
            require(code==0 and found==1 and np.array_equal(got.reshape(512,1024),expand(seed)), 'Unaligned expansion failed')
        else:
            require(code!=0 and found==0 and not np.any(got), 'Invalid descriptor was not safely rejected')
        results.append({'case':name,'error_code':code,'pass':True})
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=3)
    args=parser.parse_args()
    require(1<=args.batch_size<=16,'batch size must be1..16')
    require(os.environ.get('PS_PARALLEL')=='none' and os.environ.get('PS_FAKESTEP_FLAG')=='0','Use runtime.sh')
    args.output.mkdir(mode=0o700,exist_ok=False)
    report={'status':'running','events_completed':0,'records':[],'fixture':str(args.root),
            'batch_size':args.batch_size,'n_gpu_streams':2,'gpu_bulk_target_bytes':8<<20,'scope':'Native GPU input/expansion/calibration correctness, no throughput claim',
            'KVIKIO_COMPAT_MODE':os.environ.get('KVIKIO_COMPAT_MODE'),'calibration_settings':{'use_offset':True,'status':False,'cmpars':None},
            'python':sys.version,'code_sha256':{str(p):file_sha(p) for p in [Path(__file__),Path(adapter.__file__),Path(fixture.__file__),Path(verify.__file__),
                Path(__file__).with_name('runtime.sh'),adapter.CALIBRATION_FILE]}}
    report_path=args.output/'report.json'
    def save(): report_path.write_text(json.dumps(report,indent=2)+'\n')
    save(); start=time.monotonic()
    try:
        import cupy as cp
        import kvikio
        import psana, psana.gpu.gpu_task_batch, psana.gpu.gpu_events
        report.update(cupy=cp.__version__,kvikio=kvikio.__version__,cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                      gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),host=os.uname().nodename,
                      runtime_files={m.__name__:{'path':m.__file__,'sha256':file_sha(m.__file__)}
                                     for m in (psana,psana.gpu.gpu_task_batch,psana.gpu.gpu_events)})
        report['device_guards']=device_guards(cp)
        adapter.register_reader()
        manifest=preflight(args.root)
        constants,report['calibration']=load_constants()
        # CPU reference uses the source detector class with frozen constants;
        # nda_raw supplies the expanded synthetic array, never source pixels.
        source=DataSource(exp='mfx101555026',run=10,dir='/sdf/data/lcls/ds/prj/public01/xtc',
                          max_events=1,live=False,skip_calib_load='all')
        source_run=next(source.runs()); det=source_run.Detector('jungfrau',status=False)
        det.raw._calibconst=constants
        cpu_raw=np.empty((32,512,1024),np.uint16)
        cpu_ds=DataSource(exp='mfx101555026',run=10,dir=str(args.root),live=False,skip_calib_load='all')
        cpu_run=next(cpu_ds.runs())
        cpu_seed_reader=cpu_run.Detector('jungfrau').sim10
        task=adapter.Sample10()
        gpu_ds=DataSource(exp='mfx101555026',run=10,dir=str(args.root),live=False,
                          skip_calib_load='all',gpu_det='jungfrau',
                          gpu_fn=GpuTask(task,inputs=task.inputs,calibconst=task.calibconst),
                          batch_size=args.batch_size,n_gpu_streams=2,gpu_memory_budget_gb=4,gpu_bulk_target_bytes=8<<20)
        # RunSerial creates its GPU manager inside __init__, before runs() yields.
        # Scope the offline calibration loader to this single test constructor;
        # restore it immediately. No pipeline methods or installed files change.
        from psana.psexp.run import RunSerial
        def frozen_setup(run):
            require(run.dsparms is gpu_ds.dsparms and run.expt=='mfx101555026' and run.runnum==10,
                    'Offline calibration hook reached an unexpected run')
            run._calib_const={'jungfrau':constants}
            run.dsparms.calibconst=run._calib_const
        with patch.object(RunSerial,'_setup_run_calibconst',frozen_setup):
            gpu_run=next(gpu_ds.runs())
        report['calibration_loading']='frozen offline snapshot, constructor-scoped calibration-loader override'

        with closing(source_run.events()) as originals, closing(cpu_run.events()) as cpu_events, closing(gpu_run.events()) as gpu_events:
            original_evt=next(originals)
            for index,gpu_evt in enumerate(gpu_events):
                cpu_evt=next(cpu_events,None)
                require(cpu_evt is not None and int(cpu_evt.timestamp)==int(gpu_evt.timestamp),'CPU/GPU event identity mismatch')
                seeds=cpu_seed_reader.seed(cpu_evt)
                require(seeds is not None and set(seeds)==set(range(32)), 'Missing/duplicate physical panels')
                for panel,seed in seeds.items():
                    cpu_raw[panel]=expand(seed)
                expected=det.raw.calib(original_evt,cversion=3,cmpars=None,nda_raw=cpu_raw).reshape(cpu_raw.shape)
                raw=gpu_evt.gpu.get('sim10.expanded').on_cpu
                errors=gpu_evt.gpu.get('sim10.errors').on_cpu
                calibrated=gpu_evt.gpu.get('jungfrau.calib').on_cpu
                require(not np.any(errors),'GPU expansion metadata error')
                require(raw.shape==cpu_raw.shape and raw.dtype==np.uint16 and np.array_equal(raw,cpu_raw),'Expanded GPU raw mismatch')
                require(calibrated.shape==expected.shape and calibrated.dtype==np.float32 and
                        np.array_equal(calibrated.view(np.uint32),expected.view(np.uint32)),'GPU calibration differs from CPU v3 bits')
                require(np.isfinite(calibrated).all(),'Nonfinite calibration result')
                report['records'].append({'event':index,'timestamp':int(gpu_evt.timestamp),'raw_sha256':digest(raw),
                                          'calibrated_sha256':digest(calibrated),'raw_equal':True,'calibrated_bitwise_equal':True,
                                          'calibrated_min':float(calibrated.min()),'calibrated_max':float(calibrated.max())})
                report['events_completed']+=1
                if index%25==0:
                    save();print('verified GPU event',index,flush=True)
            require(next(cpu_events,None) is None,'GPU skipped final events')
        require(report['events_completed']==manifest['event_count'],'Wrong event count')
        report.update(status='pass',callback_calls=task.calls,callback_batch_sizes=task.batch_sizes,
                      cuda_streams=sorted(task.streams),cupy_pool_peak_bytes=task.pool_peak_bytes,
                      source_manifest_sha256=file_sha(args.root/'generation.json'),
                      compared_pixels=manifest['event_count']*32*512*1024)
    except BaseException as exc:
        report.update(status='fail',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        report['elapsed_seconds']=time.monotonic()-start
        save()


if __name__=='__main__': main()
