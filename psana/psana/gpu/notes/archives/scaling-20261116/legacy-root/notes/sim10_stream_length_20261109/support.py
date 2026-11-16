"""Relocated fixture identity and owner-scoped Lustre page-cache evidence."""
import ctypes, hashlib, json, mmap, os, struct, subprocess, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
NOTES=HERE.parent
FIXTURE=NOTES/'mfx_expand10_100k_20261109'
TRANSFER=NOTES/'sim10_globus_transfer_20261109/transfer-manifest.json'
ROOT=Path('/pscratch/sd/m/monarin/psana2-gpu/datasets/mfx101555026-r0010-sim10-100k-jf')
EVENT_BYTES=3358200
BULK=524288
BATCH=100
BUDGET=6
WORKERS=8
MODE='local-page-cache-only; remote traffic unverified'
def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def ts_hash(stamps):return hashlib.sha256(struct.pack('<%dQ'%len(stamps),*stamps)).hexdigest()
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2)+'\n')
def affinity():
    cores=[];seen=set()
    for cpu in sorted(os.sched_getaffinity(0)):
        t=Path('/sys/devices/system/cpu/cpu%d/topology'%cpu)
        key=((t/'physical_package_id').read_text(),(t/'core_id').read_text())
        if key not in seen:cores.append(cpu);seen.add(key)
    require(len(cores)>=12,'Need >=12 physical cores for 8 workers plus main/helpers')
    os.sched_setaffinity(0,cores);return cores

def fixture():
    a=json.loads((FIXTURE/'acceptance.json').read_text());v=json.loads((FIXTURE/'verification.json').read_text())
    transfer=json.loads(TRANSFER.read_text())
    require(a['accepted'] and a['event_count']==100000,'Unaccepted fixture')
    require(sha(FIXTURE/'verification.json')==a['verification_sha256'],'Validation receipt changed')
    require(sha(ROOT/'generation.json')==a['generation_sha256']==v['generation_sha256'],'Manifest identity')
    manifest=json.loads((ROOT/'generation.json').read_text())
    stamps=[p['timestamp'] for p in manifest['plan'] if p['service']==12]
    require(len(stamps)==100000 and ts_hash(stamps)==v['timestamp_sha256'],'Timestamp receipt')
    require(manifest['event_count']==100000 and manifest['format']=='sample10-v1','Fixture format')
    require(transfer['destination_root']==str(ROOT) and transfer['selected_streams']==[5,6,7,8,9],'Transfer relocation')
    streams=[s for s in manifest['streams'] if s['panels']]
    require([s['stream'] for s in streams]==[5,6,7,8,9],'JF stream set')
    require(sorted(p for s in streams for p in s['panels'])==list(range(32)),'JF panel coverage')
    refs={r['source_path']:r for r in transfer['files']}
    for s in streams:
        for kind,r in s['files'].items():
            t=refs[r['path']];p=Path(t['destination_path'])
            require(p==ROOT/t['relative_path'] and p.resolve().parent==(ROOT if kind=='bd' else ROOT/'smalldata').resolve(),'Source escaped root')
            require(t['stream']==s['stream'] and t['kind']==kind and p.stat().st_uid==os.geteuid(),'Source ownership/selection')
            require(p.stat().st_size==r['bytes']==t['bytes'] and r['sha256']==t['sha256'],'Source file identity')
            r['source_path']=r['path'];r['path']=str(p)
    require(len(transfer['files'])==11,'Transfer set')
    return manifest,streams,stamps

def mount():
    result=json.loads(subprocess.check_output(['findmnt','-J','-T',str(ROOT)],text=True))
    require(result['filesystems'][0]['fstype']=='lustre','Require Lustre input')
    return result

def cache(files,evict=False):
    mount();refs={str(ROOT/r['relative_path']):r for r in json.loads(TRANSFER.read_text())['files'] if r['kind']=='bd'}
    require(list(map(str,files))==list(refs),'Cache file set')
    libc=ctypes.CDLL(None,use_errno=True)
    libc.mincore.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p]
    libc.mincore.restype=ctypes.c_int
    rows=[]
    for p in files:
        require(p.resolve().parent==ROOT.resolve() and not p.is_symlink(),'Foreign cache path')
        fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
        try:
            st=os.fstat(fd);n=st.st_size
            require(st.st_uid==os.geteuid() and n==refs[str(p)]['bytes'],'Foreign cache file')
            pages=(n+mmap.PAGESIZE-1)//mmap.PAGESIZE
            def count():
                with mmap.mmap(fd,n,access=mmap.ACCESS_COPY) as m:
                    address=ctypes.addressof(ctypes.c_char.from_buffer(m));v=(ctypes.c_ubyte*pages)()
                    if libc.mincore(address,n,v):raise OSError(ctypes.get_errno(),'mincore')
                    import numpy as np
                    return int(np.count_nonzero(np.ctypeslib.as_array(v)&1))
            if evict:
                os.fsync(fd)
                for _ in range(20):
                    os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
                    if count()==0:break
                    time.sleep(.1)
            rows.append(dict(path=str(p),bytes=n,pages=pages,page_size=mmap.PAGESIZE,resident_pages=count()))
        finally:os.close(fd)
    if evict:require(all(r['resident_pages']==0 for r in rows),'Cold gate failed')
    return rows

def diagnostics():
    """Raw diagnostic evidence only: no counter is asserted to measure remote bytes."""
    rows={}
    for pattern in ('/proc/fs/lustre/llite/*/stats','/sys/fs/lustre/llite/*/stats','/sys/class/cxi/cxi*/device/counters/*'):
        import glob
        for name in glob.glob(pattern):
            try:rows[name]=Path(name).read_text()
            except OSError as e:rows[name]=repr(e)
    for cmd in (['lctl','get_param','llite.*.stats'],['ip','-s','link']):
        try:
            r=subprocess.run(cmd,text=True,capture_output=True,timeout=10)
            rows[' '.join(cmd)]=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
        except (OSError,subprocess.TimeoutExpired) as e:rows[' '.join(cmd)]=repr(e)
    return dict(mode=MODE,wall=time.time(),monotonic=time.perf_counter(),raw=rows)

def histogram(events):return {'524724':events,'629664':events,'734604':3*events}
def source_hashes():
    files=[HERE/x for x in ('campaign.py','runner.py','support.py','bulk_accounting.py','run.sbatch','runtime.sh','clean_exec.py','validate_input.py','audit.py')]
    files += [NOTES/'mfx_expand10_stage2_20261105/adapter.py',NOTES/'synthetic_mfx_stage3_20261104/xtc_fixture.py',NOTES.parent/'psana/psana/gpu/examples/jungfrau_calibration.py',FIXTURE/'acceptance.json',FIXTURE/'verification.json',TRANSFER,HERE/'perlmutter-runtime.json']
    return {str(p.relative_to(NOTES.parent)):sha(p) for p in files}
MATRIX=[('pilot-s%d'%s,s,1000) for s in (1,2)]
PAIRS=((1,10000),(2,10000),(1,100000),(2,100000))
MATRIX += [('%02d-s%d-n%d'%(i,s,n),s,n) for i,(s,n) in enumerate(PAIRS+PAIRS[::-1])]
