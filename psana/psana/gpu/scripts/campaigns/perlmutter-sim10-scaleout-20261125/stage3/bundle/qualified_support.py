"""Pinned data/build contracts and bounded cache checks for the PR298 campaign."""
import ctypes,hashlib,json,mmap,os,subprocess,time
from pathlib import Path
ROOT=Path('/pscratch/sd/m/monarin/psana2-gpu/pr298-striping-20261111')
SOURCE=ROOT/'source'
PREFIX=SOURCE/'install_psana'
SHA='67c327a148a08b2aaf972e67e1e02e367c2b298f'
HERE=Path(__file__).resolve().parent
COUNTS=(1,2,4,8,16,32)
EVENT_BYTES=3358200

def require(ok,msg):
    if not ok:raise ValueError(msg)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def write(path,obj):
    path=Path(path);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)
def identity(path):
    s=Path(path).stat();return {k:getattr(s,k) for k in ['st_dev','st_ino','st_size','st_mtime_ns','st_uid']}
def load_inputs():return json.loads((ROOT/'inputs.json').read_text())
def verify_payloads(info):
    for g in info['sets']:
        for r in g['files']:
            require(identity(r['path'])==r['identity'],'Payload changed: '+r['path'])
            raw=subprocess.check_output(['lfs','getstripe',r['path']],text=True)
            require(raw==r['layout_raw'],'Stripe layout changed: '+r['path'])
def view(info,count):return next(g for g in info['sets'] if g['count']==count)
def histogram(n):return {'524724':n,'629664':n,'734604':3*n}
from runtime_check import selected
_runtime=selected()
SOURCE=Path(_runtime['source']);PREFIX=Path(_runtime['prefix']);SHA=_runtime['commit']

def runtime(allow_test_source=False):
    import sys
    expected=selected()
    rows={}
    for name,m in tuple(sys.modules.items()):
        f=getattr(m,'__file__',None)
        if name=='psana' or name.startswith('psana.') or name in ('dgram','smdreader','parallelreader','eventbuilder'):
            if f:
                p=Path(f).resolve()
                if allow_test_source and p.is_relative_to(SOURCE/'psana/psana'):
                    source_rel=str(p.relative_to(SOURCE))
                    if source_rel in expected['tests']:
                        require(sha(p)==expected['tests'][source_rel],'Test source changed');rows[name]={'path':str(p),'sha256':sha(p),'test_source':True};continue
                    rel='lib/python3.11/site-packages/psana/'+str(p.relative_to(SOURCE/'psana/psana'))
                    require(rel in expected['installed'] and sha(p)==expected['installed'][rel],'Test-loaded source differs from install')
                    rows[name]={'path':str(p),'sha256':sha(p),'test_source_equal_install':True};continue
                require(p.is_relative_to(PREFIX),'Unexpected runtime '+str(p))
                rel=str(p.relative_to(PREFIX));require(rel in expected['installed'],'Unpinned module '+rel)
                actual=sha(p);require(actual==expected['installed'][rel],'Runtime changed '+rel);rows[name]={'path':str(p),'sha256':actual}
    require('psana' in rows,'psana not imported')
    return rows

def select_affinity(probes):
    require(len(probes)==6 and sorted(r['rank'] for r in probes)==list(range(6)), 'Six rank probes')
    require(len({r['host'] for r in probes})==1,'One probe host')
    common=set.intersection(*(set(r['cpus']) for r in probes));chosen=[];seen=set()
    for cpu in sorted(common):
        key=tuple(probes[0]['topology'][str(cpu)])
        require(all(tuple(r['topology'][str(cpu)])==key for r in probes),'Topology differs')
        if key not in seen:chosen.append(cpu);seen.add(key)
    require(len(chosen)>=34,'Need34 physical cores common to every rank')
    sizes=[3,3,7,7,7,7];mapping={};start=0
    for rank,size in enumerate(sizes):
        mapping[str(rank)]=chosen[start:start+size];start+=size
    return mapping

def affinity():
    rank=int(os.environ['SLURM_PROCID']);mapping=json.loads(os.environ['JF_CPU_AFFINITY'])
    chosen=mapping[str(rank)];size=3 if rank<2 else 7
    require(len(chosen)==len(set(chosen))==size and set(chosen)<=set(os.sched_getaffinity(0)),'Rank cores not in Slurm step mask')
    union=[c for values in mapping.values() for c in values]
    require(len(union)==len(set(union))==34,'Rank masks overlap')
    seen=set()
    for cpu in union:
        p=Path('/sys/devices/system/cpu/cpu%d/topology'%cpu)
        seen.add(((p/'physical_package_id').read_text(),(p/'core_id').read_text()))
    require(len(seen)==34,'CPU list contains SMT siblings')
    os.sched_setaffinity(0,chosen);require(set(os.sched_getaffinity(0))==set(chosen),'Affinity readback differs')
    return chosen

def cache(rows,evict):
    libc=ctypes.CDLL(None,use_errno=True);libc.mincore.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p];libc.mincore.restype=ctypes.c_int
    result=[]
    for r in rows:
        p=Path(r['path']);require(not p.is_symlink() and identity(p)==r['identity'],'Cache allowlist identity')
        fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
        try:
            s=os.fstat(fd);require(s.st_uid==os.geteuid() and s.st_ino==r['identity']['st_ino'],'Cachefd identity')
            if evict:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
            resident=0;pages=0
            for offset in range(0,s.st_size,256<<20):
                size=min(256<<20,s.st_size-offset);n=(size+mmap.PAGESIZE-1)//mmap.PAGESIZE
                with mmap.mmap(fd,size,access=mmap.ACCESS_COPY,offset=offset) as m:
                    address=ctypes.addressof(ctypes.c_char.from_buffer(m));v=(ctypes.c_ubyte*n)()
                    if libc.mincore(address,size,v):raise OSError(ctypes.get_errno(),'mincore')
                    import numpy as np
                    resident+=int(np.count_nonzero(np.ctypeslib.as_array(v)&1));pages+=n
            result.append({'path':str(p),'pages':pages,'resident':resident})
        finally:os.close(fd)
    return result

def cold_gate(info,count):
    rows=[r for g in info['sets'] for r in g['files']]
    scans=[]
    for attempt in range(3):
        a=cache(rows,True);time.sleep(.05);b=cache(rows,False)
        if all(r['resident']==0 for r in a+b):
            selected=cache(view(info,count)['files'],False)
            if all(r['resident']==0 for r in selected):return {'all30_scans':[a,b],'selected':selected,'attempts':attempt+1}
        scans.append([a,b])
    raise ValueError('Local cold gate failed')

def fd_record(fd,expected):
    s=os.fstat(fd);r={'fd':int(fd),'path':os.path.realpath('/proc/self/fd/'+str(fd)),'dev':s.st_dev,'ino':s.st_ino,'bytes':s.st_size}
    require(r['path']==expected['path'] and r['dev']==expected['identity']['st_dev'] and r['ino']==expected['identity']['st_ino'] and r['bytes']==expected['bytes'],'Wrong actual input handle')
    return r

def memory():
    return {'meminfo':Path('/proc/meminfo').read_text(),'time':time.time()}
def diagnostics():
    result={'memory':memory()}
    for pattern in ['llite.*.max_read_ahead*','llite.*.max_cached_mb','osc.*.stats']:
        cmd=['/usr/sbin/lctl','get_param',pattern]
        try:
            r=subprocess.run(cmd,capture_output=True,text=True,timeout=10);result[pattern]={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
        except (OSError,subprocess.TimeoutExpired) as e:result[pattern]={'error':str(e)}
    result['bdi']={str(p):p.read_text() for p in Path('/sys/class/bdi').glob('*/read_ahead_kb') if os.access(p,os.R_OK)}
    return result
