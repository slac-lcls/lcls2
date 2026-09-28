"""Four-rank public GpuTask delivery: exclusive/hybrid, batching, host results."""
import json
import sys

import numpy as np


def main():
    from mpi4py import MPI
    from psana import DataSource
    from psana.gpu import GpuTask
    mode=sys.argv[1]
    comm=MPI.COMM_WORLD
    assert comm.size==4 and mode in ('exclusive','hybrid')
    calls=[]
    def callback(batch,stream):
        import cupy as cp
        calls.append((batch.size,batch.step_generation))
        assert cp.cuda.get_current_stream().ptr==stream.ptr
        raw=batch.input('jungfrau.raw')
        batch.publish('stamp',batch.timestamps_gpu)
        batch.publish('pixel',raw.reshape(batch.size,-1)[:,0].copy())
        batch.publish('count',cp.full(batch.size,9,cp.uint32))
        batch.publish('empty',cp.empty((batch.size,0,2),cp.uint8))
        assert batch.calibconst('jungfrau','pixel_gain').size>0
    task=GpuTask(callback,['jungfrau.raw'],[('jungfrau','pixel_gain')])
    options=dict(exp='mfx100848724',run=51,dir='/sdf/data/lcls/ds/prj/public01/xtc',
        detectors=['jungfrau'],max_events=13,batch_size=5,n_gpu_streams=2,
        gpu_fn=task,gpu_d2h_pinned_bytes=8192,gpu_bulk_read=True)
    options['hybrid_det' if mode=='hybrid' else 'gpu_det']='jungfrau'
    seen=[];held=[];role=None
    for run in DataSource(**options).runs():
        role=run.comms.node_type()
        for event in run.events():
            assert event.gpu.get('stamp').on_cpu==event.timestamp
            assert event.gpu.get('count').on_cpu==9
            assert event.gpu.get('empty').on_cpu.shape==(0,2)
            fields=event.gpu.detector('jungfrau').field('raw','raw').on_cpu
            assert event.gpu.get('pixel').on_cpu==fields[fields.segment_ids[0]].flat[0]
            seen.append(int(event.timestamp));held.append(event)
    for event in held:assert event.gpu.get('stamp').on_cpu==event.timestamp
    if role!='bd':assert 'cupy' not in sys.modules
    assert sum(n for n,_ in calls)==len(seen)
    summary=comm.gather(dict(role=role,seen=seen,calls=calls),root=0)
    if comm.rank==0:
        stamps=[ts for r in summary for ts in r['seen']]
        batches=[n for r in summary for n,_ in r['calls']]
        assert len(stamps)==len(set(stamps))==13
        assert max(batches)==5 and sum(batches)==13 and min(batches)<5
        assert sum(bool(r['seen']) for r in summary)==2
        print('MPI_PUBLICATION_OK '+json.dumps(dict(mode=mode,ranks=summary)),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        from mpi4py import MPI
        traceback.print_exc();MPI.COMM_WORLD.Abort(1)
