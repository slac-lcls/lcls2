"""Strict owner-scoped cache gates and fixed run387 provenance."""
import ctypes
import hashlib
import json
import mmap
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import numpy as np
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
SOURCE=Path('/sdf/data/lcls/drpsrcf/ffb/users/monarin/jf-feespec-bulk-38995226/xtc')
REF=json.loads((HERE/'reference.json').read_text())
NAMES=tuple(REF['10000']['stage_bytes'])
sys.path.insert(0,str(HERE.parent/'mfx_kvikio_read_20261108'))
from network_watch import physical_counters,counter_delta
from plan_reads import run_queue

class CacheGateError(ValueError):pass

def require(ok,message):
    if not ok:raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def digest(array):
    a=np.array(array,copy=True,order='C')
    return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest(),nonzero=int(np.count_nonzero(a)))

def ts_hash(stamps):return hashlib.sha256(struct.pack('<%dQ'%len(stamps),*stamps)).hexdigest()

def paths(directory):
    directory=Path(directory).resolve()
    local=os.environ.get('JF_LOCAL_INPUT')
    require(directory==SOURCE.resolve() or (local and directory==Path(local).resolve()),'Foreign input root')
    result=[directory/n for n in NAMES]
    for p in result:
        st=p.stat();require(p.resolve().parent==directory and st.st_uid==os.geteuid(),'Foreign cache file')
        require(st.st_size==REF['10000']['stage_bytes'][p.name],'Input size changed')
    return result

def residency(directory,evict=False):
    libc=ctypes.CDLL(None,use_errno=True)
    libc.mincore.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p];libc.mincore.restype=ctypes.c_int
    rows=[]
    for p in paths(directory):
        fd=os.open(p,os.O_RDONLY)
        try:
            n=os.fstat(fd).st_size;pages=(n+mmap.PAGESIZE-1)//mmap.PAGESIZE
            def count():
                with mmap.mmap(fd,n,access=mmap.ACCESS_COPY) as m:
                    address=ctypes.addressof(ctypes.c_char.from_buffer(m));v=np.empty(pages,np.uint8)
                    if libc.mincore(address,n,v.ctypes.data):raise OSError(ctypes.get_errno(),'mincore')
                    return int(np.count_nonzero(v&1))
            before=after=count()
            if evict:
                os.fsync(fd)
                for attempt in range(20):
                    os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED);after=count()
                    if after==0:break
                    time.sleep(.1)
                require(after==0,'Failed cold gate '+str(p))
            rows.append(dict(path=str(p),bytes=n,pages=pages,before=before,resident_pages=after))
        finally:os.close(fd)
    return dict(files=rows,pages=sum(x['pages'] for x in rows),resident_pages=sum(x['resident_pages'] for x in rows))

def condition(directory,kind):
    from memory_state import snapshot
    start=time.perf_counter();history=[]
    record=dict(kind=kind,history=history,memory_before=snapshot())
    def save():
        record['seconds']=time.perf_counter()-start
        target=os.environ.get('JF_CACHE_REPORT')
        if target:Path(target).write_text(json.dumps(record,indent=2)+'\n')
    try:
        # Reset before warming so existing cold-run/staging pages do not retain
        # their previous allocation placement. This is outside all timed work.
        require(kind in ('cold','warm'),'Invalid cache kind')
        history.append(dict(stage='reset',cache=residency(directory,True)));save()
        if kind=='warm':
            ranges=[(str(p),p.stat().st_size) for p in paths(directory)]
            subprocess.run(['numactl','--interleave=all',sys.executable,str(HERE/'warm_cache.py'),'--prefixes',json.dumps(ranges)],check=True)
        cache=residency(directory);history.append(dict(stage='prepared',cache=cache));save()
        if kind=='warm':
            for attempt in range(1,9):
                missing=[(x['path'],x['bytes']) for x in cache['files'] if x['resident_pages']/x['pages']<.999]
                if not missing:break
                subprocess.run(['numactl','--interleave=all',sys.executable,str(HERE/'warm_cache.py'),'--missing-prefixes',json.dumps(missing)],check=True)
                cache=residency(directory);history.append(dict(stage='repair-%d'%attempt,cache=cache));save()
        record['cache']=cache;record['memory_after']=snapshot();save()
        if not all(x['resident_pages']==0 if kind=='cold' else x['resident_pages']/x['pages']>=.999 for x in cache['files']):
            raise CacheGateError('Cache gate failed: '+json.dumps(cache))
        record['status']='pass';save();return record
    except BaseException as exc:
        record['status']='fail';record['error']=repr(exc);record['memory_after']=snapshot();save();raise

def begin(directory,kind):
    a=physical_counters();time.sleep(1);b=physical_counters()
    return dict(idle=dict(before=a,after=b,delta=counter_delta(a,b)),conditioning=condition(directory,kind),nic_before=physical_counters())

def end(record,directory,seconds):
    record['nic_after']=physical_counters();record['nic_delta']=counter_delta(record['nic_before'],record['nic_after'])
    record['cache_after']=residency(directory);record['loop_seconds']=seconds
    require(all(x['seconds']>=seconds for x in record['nic_delta'].values()),'NIC bracket too short')

def requests(directory,events=10000):
    from psana import dgram
    rows=[];reference=None
    for i,p in enumerate(paths(directory)):
        smd=p.parent/'smalldata'/p.name.replace('.xtc2','.smd.xtc2')
        require(sha(smd)==REF['10000']['smd_hashes'][smd.name],'SMD hash differs')
        fd=os.open(smd,os.O_RDONLY);part=[];stamps=[]
        try:
            cfg=dgram.Dgram(file_descriptor=fd)
            while len(part)<events:
                d=dgram.Dgram(config=cfg)
                if d.service()!=12:continue
                info=d.smdinfo[0].offsetAlg;offset=int(info.intOffset);size=int(info.intDgramSize)
                require(offset>=0 and size>1<<20 and offset+size<=p.stat().st_size,'Bad SMD range')
                require(not part or offset>=part[-1][1]+part[-1][2],'Overlapping ranges')
                part.append((i,offset,size));stamps.append(int(d.timestamp()))
        finally:os.close(fd)
        require(ts_hash(stamps)==REF[str(events)]['timestamp_sha256'],'Wrong timestamps')
        if reference is None:reference=stamps
        require(stamps==reference,'Stream identity mismatch');rows.append(part)
    plan=[r for group in zip(*rows) for r in group]
    require(sum(r[2] for r in plan)==REF[str(events)]['payload_bytes'],'Wrong payload')
    return plan,reference

def sources():
    files=list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+list(HERE.glob('*.sbatch'))
    files += [HERE.parent/'mfx_kvikio_read_20261108/network_watch.py',HERE.parent/'mfx_kvikio_read_20261108/plan_reads.py']
    return {str(p):sha(p) for p in files}


def prefix_residency(path, length, *, missing_ranges=False):
    """Measure only the input prefix exercised by a short run, without reading it."""
    import ctypes
    import mmap
    import numpy as np
    with open(path, 'rb', buffering=0) as source:
        if not 0 < length <= os.fstat(source.fileno()).st_size:
            raise ValueError('invalid measured input extent')
        mapping = mmap.mmap(source.fileno(), length, flags=mmap.MAP_PRIVATE,
                            prot=mmap.PROT_READ | mmap.PROT_WRITE)
    pages = (length + mmap.PAGESIZE - 1) // mmap.PAGESIZE
    vector = (ctypes.c_ubyte * pages)()
    anchor = ctypes.c_char.from_buffer(mapping)
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        rc = libc.mincore(ctypes.c_void_p(ctypes.addressof(anchor)),
                         ctypes.c_size_t(length), ctypes.byref(vector))
        if rc:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(path))
        present = np.ctypeslib.as_array(vector) & 1
        resident = int(present.sum(dtype=np.uint64))
        if missing_ranges:
            edges = np.flatnonzero(np.diff(np.r_[False, present == 0, False]))
            missing = [(int(first)*mmap.PAGESIZE,
                        min(int(stop)*mmap.PAGESIZE, length)-int(first)*mmap.PAGESIZE)
                       for first, stop in zip(edges[::2], edges[1::2])]
    finally:
        del anchor
        mapping.close()
    result = dict(path=str(path), bytes=length, pages=pages, resident_pages=resident,
                  resident_fraction=resident/pages)
    if missing_ranges:
        result['missing_ranges'] = missing
    return result


def cache_unavailable(output,settings,exc):
    import platform
    target=Path(os.environ['JF_CACHE_REPORT'])
    report=dict(status='cache-unavailable',settings=settings,host=platform.node(),job=os.getenv('SLURM_JOB_ID'),error=str(exc),cache_report=str(target),cache_sha256=sha(target),source_sha256=sources())
    require(settings['cache']=='warm','Only unavailable warm gates may be skipped')
    with open(output,'x') as f:json.dump(report,f,indent=2);f.write('\n')
