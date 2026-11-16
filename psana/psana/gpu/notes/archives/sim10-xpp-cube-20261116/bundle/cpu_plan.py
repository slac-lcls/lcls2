"""Physical-core provisioning for BD/KvikIO demand; fail rather than clamp."""
import json
import math
import os
from pathlib import Path

HELPERS_PER_RANK = 2
# (BDs, workers per BD, EB batch, bounce MiB); one stream throughout.
CONFIGURATIONS = ((1,8,50,16), (1,8,50,64), (4,4,200,16), (4,4,200,64))
BATCH_SIZES = (50,200)


def plan_key(bds,workers):return 'bd%d-workers%d'%(bds,workers)


def required_cores(bds,workers):
    return 2*(1+HELPERS_PER_RANK)+bds*(workers+1+HELPERS_PER_RANK)


def validate_case(bds,workers,batch,bounce_mib,streams,events):
    if (bds,workers,batch,bounce_mib) not in CONFIGURATIONS or streams!=1 or events not in (1000*bds,100000):
        raise ValueError('Case outside reviewed one-stream bounce-buffer screen')


def discover():
    rows=[]
    for cpu in sorted(os.sched_getaffinity(0)):
        root=Path('/sys/devices/system/cpu/cpu%d/topology'%cpu)
        rows.append(dict(cpu=cpu,socket=int((root/'physical_package_id').read_text()),core=int((root/'core_id').read_text())))
    return rows


def plan(topology,bds,workers=8):
    if bds not in (1,2,3,4) or workers<1:
        raise ValueError('Unsupported BD/worker count')
    unique={}
    for row in sorted(topology,key=lambda r:r['cpu']):
        unique.setdefault((row['socket'],row['core']),row)
    # Spread the growing core set across sockets using a stable ordering.
    cores=sorted(unique.values(),key=lambda r:(r['core'],r['socket']))
    sizes=[1+HELPERS_PER_RANK]*2+[workers+1+HELPERS_PER_RANK]*bds
    required=sum(sizes)
    if len(cores)<required:
        raise ValueError('Need %d physical cores for %d BDs/%d workers; only %d available'%(required,bds,workers,len(cores)))
    ranks=[];offset=0
    for size in sizes:
        ranks.append([r['cpu'] for r in cores[offset:offset+size]]);offset+=size
    return dict(bds=bds,workers_per_bd=workers,total_workers=bds*workers,
                helper_cores=(bds+2)*HELPERS_PER_RANK,required_physical_cores=required,
                available_physical_cores=len(cores),allowed_logical_cpus=len(topology),ranks=ranks)


def bind_rank(p,rank):
    chosen=p['ranks'][rank]
    if not set(chosen)<=os.sched_getaffinity(0):
        raise ValueError('Rank binding is outside inherited CPU affinity')
    if len(chosen)!=len(set(chosen)):
        raise ValueError('Duplicate CPUs in rank binding')
    os.sched_setaffinity(0,chosen)
    if set(os.sched_getaffinity(0))!=set(chosen):
        raise ValueError('Rank binding did not take effect')
    return chosen


def case_matrix():
    pairs=list(CONFIGURATIONS)
    pilots=[('preflight-bd%d-workers%d-batch%d-bounce%d'%(b,w,batch,t),b,w,1,batch,t,1000*b) for b,w,batch,t in pairs]
    full=[('%02d-bd%d-workers%d-batch%d-bounce%d'%(i,b,w,batch,t),b,w,1,batch,t,100000) for i,(b,w,batch,t) in enumerate(pairs+pairs[::-1])]
    return pilots+full


def slurm_request(configurations=CONFIGURATIONS):
    ntasks=max(b+2 for b,w,batch,*_ in configurations)
    required=max(required_cores(b,w) for b,w,batch,*_ in configurations)
    per_task=math.ceil(required/ntasks)
    return dict(ntasks=ntasks,cpus_per_task=per_task,hint='nomultithread',
                required_physical_cores=required,requested_physical_cores=ntasks*per_task,
                policy='Reserve largest case once; bind each case to its own demand-derived disjoint rank sets')


def thread_snapshot():
    rows=[]
    for p in Path('/proc/self/task').iterdir():
        try:
            text=(p/'stat').read_text();end=text.rindex(')');fields=text[end+2:].split()
            rows.append(dict(tid=int(p.name),name=text[text.index('(')+1:end],
                             state=fields[0],user_ticks=int(fields[11]),system_ticks=int(fields[12])))
        except FileNotFoundError:
            continue
    return dict(clock_ticks=os.sysconf('SC_CLK_TCK'),threads=rows)
