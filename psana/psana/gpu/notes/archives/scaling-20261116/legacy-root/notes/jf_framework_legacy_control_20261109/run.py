"""Same-node cold control using the previously reviewed raw staging harness."""
import json,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'jf_raw_cache_20261109'
sys.path.insert(0,str(OLD))
import common
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
inv=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,pci.bus_id,name','--format=csv,noheader'],text=True)
os.environ['JF_GPU_UUID']=inv.splitlines()[0].split(',')[0].strip()
cores=[];seen=set()
for cpu in sorted(os.sched_getaffinity(0)):
    t=Path('/sys/devices/system/cpu/cpu%d/topology'%cpu);key=((t/'physical_package_id').read_text(),(t/'core_id').read_text())
    if key not in seen:cores.append(cpu);seen.add(key)
common.require(len(cores)>=8,'Need >=8 physical cores')
os.environ['BENCH_CPU_AFFINITY']=','.join(map(str,cores))
# Identity fully hashed in the immediately preceding same-node framework job;
# assert the same named, owned prefixes and saved SMD/timestamp reference here.
common.requests(common.SOURCE)
(out/'provenance.json').write_text(json.dumps(dict(gpu_inventory=inv,allowed_physical_cores=cores,source_sha256=common.sources(),
    launcher_sha256={str(p):common.sha(p) for p in [HERE/'run.py',HERE/'run.sbatch']},
    input_identity_note='Full SHA verified by predecessor job43996574; this control rechecks owner/sizes and saved SMD/timestamp hashes.',
    predecessor='/sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/jf_framework_cold_20261109_r1/provenance.json'),indent=2)+'\n')
cases=[]
for rep in (1,2):
    name='legacy-raw-cold-r%d'%rep
    cmd=['numactl','--interleave=all','mpiexec','-n','3','--bind-to','none',sys.executable,str(OLD/'bench.py'),
        '--directory',str(common.SOURCE),'--output',str(out/(name+'.json')),'--cache','cold','--bds','1','--events','10000']
    print('Starting '+name,flush=True)
    with (out/(name+'.log')).open('x') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600,
        env=dict(os.environ,JF_CACHE_REPORT=str(out/(name+'.cache.json'))))
    r=json.loads((out/(name+'.json')).read_text());common.require(r['status']=='pass','Failed historical control')
    cases.append(dict(name=name,command=cmd,gib_s=r['gib_s'],hz=r['events_per_second']))
    (out/'progress.json').write_text(json.dumps(cases,indent=2)+'\n');print(json.dumps(cases[-1]),flush=True)
(out/'complete.json').write_text(json.dumps(dict(status='pass',cases=cases),indent=2)+'\n')
