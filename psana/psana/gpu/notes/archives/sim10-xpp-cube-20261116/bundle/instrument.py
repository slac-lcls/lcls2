"""Read-only handle witnesses and bounded request accounting; no scheduler changes."""
import hashlib,os,struct
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
from support import require,fd_record

@contextmanager
def budget_peaks():
    """Observe every successful reserve/hold under the budget's existing lock.

    These are exact tracked charge/hold maxima, not total VRAM use. Rollbacks
    transfer committed to held without increasing their sum.
    """
    from psana.gpu.gpu_budget import _GpuBudget
    records={}
    def wrapped(original):
        def call(budget,*args,**kwargs):
            with budget._lock:
                result=original(budget,*args,**kwargs)
                row=records.setdefault(id(budget),dict(limit=budget.limit(),peak_committed=0,peak_committed_plus_held=0,observations=0))
                row['peak_committed']=max(row['peak_committed'],budget._committed)
                row['peak_committed_plus_held']=max(row['peak_committed_plus_held'],budget._committed+budget._held)
                row['observations']+=1
                require(budget._committed+budget._held<=budget.limit(),'Tracked budget exceeded')
                return result
        return call
    with patch.object(_GpuBudget,'reserve',wrapped(_GpuBudget.reserve)),patch.object(_GpuBudget,'hold',wrapped(_GpuBudget.hold)):
        yield records

def fd_snapshot():
    result={}
    for p in Path('/proc/self/fd').iterdir():
        try:result[int(p.name)]=os.path.realpath(os.readlink(p))
        except OSError:pass
    return result

@contextmanager
def bindings(files):
    from psana.dgrammanager import DgramManager
    from psana.gpu.gpu_kvikio_read import KvikioGpuReader
    expected={r['path']:r for r in files};records={'dgram':[],'kvikio':[]};seen={}
    dm_init=DgramManager.__init__;get=KvikioGpuReader._file_for_identity
    def init(dm,*a,**kw):
        dm_init(dm,*a,**kw)
        for fd in getattr(dm,'fds',[]):
            real=os.path.realpath('/proc/self/fd/'+str(fd))
            if Path(real).name.endswith('.xtc2') and not real.endswith('.smd.xtc2'):
                require(real in expected,'DgramManager opened wrong payload '+real);records['dgram'].append(fd_record(fd,expected[real]))
    def opened(reader,ident):
        key=(id(reader),ident)
        if key in seen:
            fh=get(reader,ident);old,fd=seen[key]
            require(fh is old and fh.fileno()==fd,'CuFile reopened; binding must be recaptured')
            return fh
        require(ident.path in expected,'KvikIO requested wrong layout')
        before=fd_snapshot();fh=get(reader,ident);after=fd_snapshot()
        primary=fd_record(fh.fileno(),expected[ident.path])
        new=[fd_record(fd,expected[ident.path]) for fd,path in after.items() if path==ident.path and before.get(fd)!=path]
        require(new,'No newly opened CuFile descriptor')
        require(primary['fd'] in {r['fd'] for r in new},'CuFile primary not captured')
        records['kvikio'].append({'primary':primary,'all_new_handles':new});seen[key]=(fh,primary['fd'])
        return fh
    with patch.object(DgramManager,'__init__',init),patch.object(KvikioGpuReader,'_file_for_identity',opened):yield records

@contextmanager
def accounting():
    from psana.gpu.gpu_kvikio_read import KvikioGpuReader
    from psana.gpu.gpu_events import GpuEventManager
    submit=KvikioGpuReader._submit_read;wait=KvikioGpuReader.wait_batch;trim=GpuEventManager._trim_gpu_caches
    stats={'requests':0,'requested_bytes':0,'completed_bytes':0,'completed_requests':0,'pressure_drains':0,'peak_outstanding_requests':0,'peak_outstanding_bytes':0,'read_size_histogram':{}}
    pending={};out_bytes=0;out_count=0;digests={}
    def issue(reader,desc,ranges,total,slot,plan):
        nonlocal out_bytes,out_count
        require(len(ranges)==1 and desc.shape==(1,6),'Fixed512KiB must yield one dgram per read')
        r=ranges[0];ts,offset,size=map(int,(desc[0,2],desc[0,3],desc[0,4]))
        require(r.file_offset==offset and r.size==size,'Physical and logical ranges disagree')
        digests.setdefault(r.file.path,hashlib.sha256()).update(struct.pack('<QQQ',ts,offset,size))
        p=submit(reader,desc,ranges,total,slot,plan);n=r.size
        pending[id(p)]=(1,n);out_count+=1;out_bytes+=n
        stats['requests']+=1;stats['requested_bytes']+=n
        k=str(n);stats['read_size_histogram'][k]=stats['read_size_histogram'].get(k,0)+1
        stats['peak_outstanding_requests']=max(stats['peak_outstanding_requests'],out_count);stats['peak_outstanding_bytes']=max(stats['peak_outstanding_bytes'],out_bytes)
        return p
    def complete(reader,p):
        nonlocal out_bytes,out_count
        result=wait(reader,p)
        if id(p) in pending:
            count,n=pending.pop(id(p));stats['completed_bytes']+=n;stats['completed_requests']+=count;out_count-=count;out_bytes-=n
        return result
    def pressure(m):stats['pressure_drains']+=1;return trim(m)
    with patch.object(KvikioGpuReader,'_submit_read',issue),patch.object(KvikioGpuReader,'wait_batch',complete),patch.object(GpuEventManager,'_trim_gpu_caches',pressure):yield stats
    require(not pending and out_count==out_bytes==0,'Uncompleted requests')
    stats['range_sha256']={k:v.hexdigest() for k,v in digests.items()}
