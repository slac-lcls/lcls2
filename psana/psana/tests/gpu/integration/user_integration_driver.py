"""Real geometry + default CPU calib reference versus external batched integration."""
import argparse
from contextlib import closing
import hashlib
import importlib
import json
from pathlib import Path
import sys
import numpy as np


def cpu_reference(directory, user_module):
    from psana import DataSource
    from psana.pscalib.geometry.GeometryAccess import GeometryAccess
    from psana.tests.gpu.user_integration_reference import integrate
    directory.mkdir(exist_ok=False)
    manifest=dict(events={}, constants={}, geometry=dict(units='mm',center_mm=[0,0],
        center_meaning='detector geometry origin; not a measured beam center',q_validation=False))
    ds=DataSource(exp='mfx100848724',run=51,dir='/sdf/data/lcls/ds/prj/public01/xtc',detectors=['jungfrau'],max_events=13)
    for run in ds.runs():
        det=run.Detector('jungfrau')
        constants={k:det.calibconst[k][0] for k in ('pedestals','pixel_gain','pixel_offset','pixel_status','status_extra')
                   if det.calibconst.get(k) is not None and det.calibconst[k][0] is not None}
        geometry_text=det.calibconst['geometry'][0]
        assert isinstance(geometry_text,str)
        geometry=GeometryAccess()
        geometry.load_pars_from_str(geometry_text)
        x,y,z=geometry.get_pixel_coords()
        assert x.size==32*512*1024 and y.size==x.size
        x=x.reshape(32,512,1024)/1000; y=y.reshape(x.shape)/1000
        edges=np.linspace(0,0.9*np.nanmax(np.hypot(x,y)),65)
        bins=user_module.radial_bin_ids(x,y,edges,center_mm=(0,0))
        assert np.any(bins<0) and np.any(bins>=0)
        np.savez(directory/'bins.npz',bin_ids=bins,edges=edges)
        manifest['geometry'].update(text_sha256=hashlib.sha256(geometry_text.encode()).hexdigest(),
                                    edges=edges.tolist(),excluded_pixels=int(np.sum(bins<0)))
        for key,value in constants.items():
            manifest['constants'][key]=dict(shape=list(value.shape),dtype=str(value.dtype),
                sha256=hashlib.sha256(value.tobytes()).hexdigest())
        status_bits=(1<<64)-1 if 'pixel_status' in constants else 0
        stextra_bits=(1<<64)-1 if 'status_extra' in constants else 0
        with closing(run.events()) as events:
            for event in events:
                raw=det.raw.raw(event)
                assert raw.shape==(32,512,1024)
                image=det.raw.calib(event).reshape(raw.shape)
                result=integrate(image[None],raw[None],np.ones((1,32),np.uint8),tuple(range(32)),
                    constants,bins,64,status_bits=status_bits,stextra_bits=stextra_bits)[0]
                stamp=str(int(event.timestamp))
                manifest['events'][stamp]=dict(histogram=result.tolist(),
                    calibration_sha256=hashlib.sha256(image.tobytes()).hexdigest())
    assert len(manifest['events'])==13 and 'cupy' not in sys.modules
    manifest['bins_sha256']=hashlib.sha256((directory/'bins.npz').read_bytes()).hexdigest()
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('CPU_INTEGRATION_REFERENCE_OK '+json.dumps(dict(events=13,geometry=manifest['geometry'])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--user-dir',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--mode',choices=('cpu','serial','exclusive','hybrid'),required=True)
    a=p.parse_args()
    sys.path.insert(0,str(a.user_dir.resolve()))
    module=importlib.import_module('jungfrau_azimuthal_integration')
    assert Path(module.__file__).resolve().parent==a.user_dir.resolve()
    if a.mode=='cpu':
        cpu_reference(a.reference,module)
        return
    from psana import DataSource
    from psana.gpu import GpuTask
    from mpi4py import MPI
    comm=MPI.COMM_WORLD
    assert comm.size==(1 if a.mode=='serial' else 4)
    reference=json.loads((a.reference/'manifest.json').read_text())
    with np.load(a.reference/'bins.npz') as data:
        analysis=module.JungfrauAzimuthalIntegration(data['bin_ids'],len(data['edges'])-1,
            use_offset='pixel_offset' in reference['constants'],
            status_bits=(1<<64)-1 if 'pixel_status' in reference['constants'] else 0,
            stextra_bits=(1<<64)-1 if 'status_extra' in reference['constants'] else 0)
    assert 'cupy' not in sys.modules
    launches=[];batches=[];seen=[];retained=[];errors=[]
    def callback(batch,stream):
        import cupy as cp
        original=cp.RawKernel
        def counted(source,name,**kwargs):
            kernel=original(source,name,**kwargs)
            def launch(grid,block,args,**kw):
                launches.append((name,kw['stream'].ptr))
                return kernel(grid,block,args,**kw)
            return launch
        cp.RawKernel=counted
        try: analysis(batch,stream)
        finally: cp.RawKernel=original
        assert launches[-2:]==[('calibrate',stream.ptr),('integrate',stream.ptr)]
        batches.append(batch.size)
    task=GpuTask(callback,inputs=analysis.inputs,calibconst=analysis.calibconst)
    options=dict(exp='mfx100848724',run=51,dir='/sdf/data/lcls/ds/prj/public01/xtc',
        detectors=['jungfrau'],gpu_fn=task,max_events=13,batch_size=5,n_gpu_streams=2,gpu_bulk_read=True)
    options['hybrid_det' if a.mode=='hybrid' else 'gpu_det']='jungfrau'
    for run in DataSource(**options).runs():
        with closing(run.events()) as events:
            for event in events:
                result=event.gpu.get(analysis.output)
                actual=result.on_cpu
                expected=np.array(reference['events'][str(int(event.timestamp))]['histogram'])
                assert actual.shape==(3,64) and actual.dtype==np.float64 and actual.nbytes==1536
                np.testing.assert_array_equal(actual[2],expected[2])
                np.testing.assert_allclose(actual[:2],expected[:2],rtol=1e-12,atol=1e-9)
                errors.append(float(np.max(np.abs(actual[:2]-expected[:2]))))
                seen.append((int(event.timestamp),hashlib.sha256(actual.tobytes()).hexdigest()))
                if not retained: retained.append((result,seen[-1][1]))
    for result,digest in retained:
        assert hashlib.sha256(result.on_cpu.tobytes()).hexdigest()==digest
    assert analysis.calls==len(batches) and len(launches)==2*len(batches)
    if not batches: assert 'cupy' not in sys.modules
    rows=comm.gather(dict(rank=comm.rank,seen=seen,batches=batches,launches=len(launches),
                         max_abs_error=max(errors,default=0)),root=0)
    if comm.rank==0:
        stamps=[ts for r in rows for ts,h in r['seen']]
        assert len(stamps)==len(set(stamps))==13
        assert sorted(n for r in rows for n in r['batches'])==[3,5,5]
        assert sum(r['launches'] for r in rows)==6
        print('USER_INTEGRATION_OK '+json.dumps(dict(mode=a.mode,ranks=rows,output_bytes_per_event=1536)),flush=True)


if __name__=='__main__':
    try: main()
    except BaseException:
        import traceback
        traceback.print_exc()
        from mpi4py import MPI
        if MPI.COMM_WORLD.size>1: MPI.COMM_WORLD.Abort(1)
        raise
