"""Cheap bounded reader accounting; optional diagnostic hooks are separate."""
from contextlib import contextmanager
from unittest.mock import patch
from support import require

@contextmanager
def accounting():
    from psana.gpu.gpu_kvikio_read import KvikioGpuReader
    from psana.gpu.gpu_events import GpuEventManager
    submit=KvikioGpuReader._submit_read;wait=KvikioGpuReader.wait_batch;trim=GpuEventManager._trim_gpu_caches
    stats=dict(requests=0,requested_bytes=0,completed_bytes=0,completed_requests=0,pressure_drains=0,peak_outstanding_requests=0,peak_outstanding_bytes=0,read_size_histogram={})
    pending={}
    outstanding=[0,0]
    def issue(reader,desc,ranges,total,slot,plan):
        p=submit(reader,desc,ranges,total,slot,plan)
        n=sum(r.size for r in ranges)
        for r in ranges:
            key=str(r.size);stats['read_size_histogram'][key]=stats['read_size_histogram'].get(key,0)+1
        pending[id(p)]=(len(ranges),n)
        stats['requests']+=len(ranges);stats['requested_bytes']+=n
        outstanding[0]+=len(ranges);outstanding[1]+=n
        stats['peak_outstanding_requests']=max(stats['peak_outstanding_requests'],outstanding[0])
        stats['peak_outstanding_bytes']=max(stats['peak_outstanding_bytes'],outstanding[1])
        return p
    def complete(reader,p):
        result=wait(reader,p) # frozen reader validates every future's byte count
        if id(p) in pending:
            count,n=pending.pop(id(p));outstanding[0]-=count;outstanding[1]-=n;stats['completed_bytes']+=n;stats['completed_requests']+=count
        return result
    def pressure(manager):stats['pressure_drains']+=1;return trim(manager)
    with patch.object(KvikioGpuReader,'_submit_read',issue),patch.object(KvikioGpuReader,'wait_batch',complete),patch.object(GpuEventManager,'_trim_gpu_caches',pressure):
        yield stats
    require(not pending,'Uncompleted input reads')
    require(stats['completed_bytes']==stats['requested_bytes'] and stats['completed_requests']==stats['requests'],'Unbalanced I/O')
