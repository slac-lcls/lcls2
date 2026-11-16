"""Bounded Weka/local comparison with independent fresh-process samples."""
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,os,shutil,subprocess,sys,signal
from pathlib import Path
from common import HERE,SOURCE,NAMES,REF,require,sha,paths,requests,residency,condition,CacheGateError
output=Path(sys.argv[1]);output.mkdir(mode=0o770,exist_ok=False)
local=Path(os.environ['JF_LOCAL_INPUT']);local.mkdir(mode=0o700,exist_ok=False)
cases=[]
def terminate(signum,frame):raise SystemExit(128+signum)
signal.signal(signal.SIGTERM,terminate)
try:
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,pci.bus_id,name','--format=csv,noheader'],text=True)
    (output/'gpu-inventory.csv').write_text(inventory);os.environ['JF_GPU_UUID']=inventory.splitlines()[0].split(',')[0].strip()
    cores=[];seen=set()
    for cpu in sorted(os.sched_getaffinity(0)):
        root=Path('/sys/devices/system/cpu/cpu%d/topology'%cpu);key=((root/'physical_package_id').read_text(),(root/'core_id').read_text())
        if key not in seen:cores.append(cpu);seen.add(key)
    require(len(cores)>=34,'Need >=34 physical cores for common historical affinity')
    os.environ['BENCH_CPU_AFFINITY']=','.join(map(str,cores))
    (output/'placement.json').write_text(json.dumps(dict(physical_cores=cores,device_mask=os.environ['JF_GPU_UUID']),indent=2)+'\n')
    for label,command in [('weka-mount',['findmnt','-T',str(SOURCE)]),('local-mount',['findmnt','-T',str(local)]),('block-devices',['lsblk','-o','NAME,TYPE,SIZE,MODEL,TRAN,MOUNTPOINTS'])]:
        result=subprocess.run(command,capture_output=True,text=True,check=True);(output/(label+'.txt')).write_text(result.stdout)
    require(shutil.disk_usage(local).free>sum(REF['10000']['stage_bytes'].values())+20*(1<<30),'Insufficient private local stage space')
    os.environ['JF_CACHE_REPORT']=str(output/'preparation-probe.cache.json')
    try:
        condition(SOURCE,'warm');print('Strict interleaved Weka warm-cache preparation probe passed',flush=True)
    except CacheGateError:
        print('Initial Weka warm cache unavailable; preserve failed gate and continue independent cases',flush=True)
    expected=json.loads((HERE/'expected-sha256.json').read_text())
    def copy_one(src):
        target=local/src.name;h=hashlib.sha256();before=src.stat()
        with src.open('rb',buffering=0) as f,target.open('xb',buffering=0) as g:
            for b in iter(lambda:f.read(16<<20),b''):
                h.update(b);view=memoryview(b)
                while view:
                    n=g.write(view);require(n and n>0,'Short stage write');view=view[n:]
            os.fsync(g.fileno())
        require(h.hexdigest()==expected[src.name] and sha(target)==expected[src.name],'Stage/source SHA mismatch')
        after=src.stat();require((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),'Source changed')
        return dict(source=str(src),local=str(target),bytes=before.st_size,sha256=h.hexdigest(),mtime_ns=before.st_mtime_ns,inode=before.st_ino)
    with ThreadPoolExecutor(max_workers=5) as ex:staged=list(ex.map(copy_one,paths(SOURCE)))
    (local/'smalldata').mkdir()
    for name,h in REF['10000']['smd_hashes'].items():
        src=SOURCE/'smalldata'/name;require(sha(src)==h,'SMD source changed');shutil.copyfile(src,local/'smalldata'/name)
    (output/'staging.json').write_text(json.dumps(staged,indent=2)+'\n');requests(SOURCE);requests(local)
    print('Stage/source SHA and independent SMD references passed',flush=True)
    def execute(name,directory,cache,bds=None,preflight=False,threads=8,task=1):
        args=['--directory',str(directory),'--cache',cache,'--output',str(output/(name+'.json'))]
        if bds:
            cmd=['mpiexec','-n',str(bds+2),'--bind-to','none',sys.executable,str(HERE/'bench.py')]+args+['--bds',str(bds)]
            if preflight:cmd+=['--events','200','--preflight']
        else:cmd=[sys.executable,str(HERE/'read.py')]+args+['--threads',str(threads),'--task-mib',str(task)]
        print('Starting '+name,flush=True)
        with (output/(name+'.log')).open('x') as f:completed=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=False,timeout=1200,env=dict(os.environ,JF_CACHE_REPORT=str(output/(name+'.cache.json'))))
        r=json.loads((output/(name+'.json')).read_text())
        accepted=completed.returncode==0 and r['status']=='pass'
        unavailable=completed.returncode==42 and r['status']=='cache-unavailable' and cache=='warm' and not preflight
        require(accepted or unavailable,'Case failed: '+name)
        cases.append(dict(name=name,command=cmd,status=r['status'],gib_s=r.get('gib_s'),artifact=str(output/(name+'.json')),conditioning_seconds=r.get('measurement',{}).get('conditioning',{}).get('seconds')))
        (output/'progress.json').write_text(json.dumps(cases,indent=2)+'\n');print(json.dumps(cases[-1]),flush=True)
    for bds in (1,4):execute('preflight-b%d'%bds,SOURCE,'cold',bds,True)
    for rep in (1,2):
        configs=[('weka',SOURCE,1),('weka',SOURCE,4),('local',local,4)]
        if rep==2:configs.reverse()
        for storage,directory,bds in configs:
            # Evict other inode set to avoid its cache competing with this source.
            residency(local if directory==SOURCE else SOURCE,True)
            for cache in ('cold','warm'):
                execute('psana-%s-b%d-%s-r%d'%(storage,bds,cache,rep),directory,cache,bds)
    for storage,directory,threads,task in [('weka',SOURCE,4,4),('weka',SOURCE,8,1),('local',local,8,1)]:
        residency(local if directory==SOURCE else SOURCE,True)
        for cache in ('cold','warm'):
            execute('read-%s-t%d-%s'%(storage,threads,cache),directory,cache,threads=threads,task=task)
    for row in staged:
        st=Path(row['source']).stat();require((st.st_size,st.st_mtime_ns,st.st_ino)==(row['bytes'],row['mtime_ns'],row['inode']),'Source metadata changed')
    (output/'complete.json').write_text(json.dumps(dict(status='pass' if all(c['status']=='pass' for c in cases) else 'partial',cases=cases),indent=2)+'\n')
finally:
    # Only this newly created local directory is removed. Shared source untouched.
    shutil.rmtree(local)
    (output/'cleanup.json').write_text(json.dumps(dict(local=str(local),removed=not local.exists()))+'\n')
