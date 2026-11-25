"""Host-qualified placement: helpers on node 0, exactly four BDs per node."""
import json,os,socket
from pathlib import Path


def require(value, message):
    if not value:raise ValueError(message)


def rank_hosts(hosts):
    require(len(hosts) in (1,2,4) and len(hosts)==len(set(hosts)), 'Unique supported hosts')
    return [hosts[0]]*2+[host for host in hosts for _ in range(4)]


def build(probes, hosts):
    expected=rank_hosts(hosts)
    require([p['rank'] for p in probes]==list(range(len(expected))), 'Complete rank probes')
    require([p['host'] for p in probes]==expected,'Slurm arbitrary rank placement')
    mapping=[None]*len(expected);uuids=['',''];pci={}
    for host in hosts:
        local=[p for p in probes if p['host']==host]
        common=set.intersection(*(set(p['cpus']) for p in local))
        cores=[];seen=set()
        for cpu in sorted(common):
            key=tuple(local[0]['topology'][str(cpu)])
            require(all(tuple(p['topology'][str(cpu)])==key for p in local),'CPU topology consistency')
            if key not in seen:cores.append(cpu);seen.add(key)
        demand=34 if host==hosts[0] else 28
        require(len(cores)>=demand,'Insufficient common physical cores')
        offset=0
        for p in local:
            size=3 if p['rank']<2 else 7
            mapping[p['rank']]=cores[offset:offset+size];offset+=size
        devices=local[0]['gpus']
        require(len(devices)==4 and len({d['uuid'] for d in devices})==4,'Four distinct GPUs per node')
        require(len({d['pci'] for d in devices})==4,'Four PCI functions per node')
        require(all(p['gpus']==devices for p in local),'Same granted GPUs per node')
        for d in devices:
            require(d['uuid'].startswith('GPU-') and d['uuid'] not in pci,'Globally unique GPU UUID')
            pci[d['uuid']]=d['pci'];uuids.append(d['uuid'])
    return dict(hosts=hosts,rank_hosts=expected,mapping=mapping,
                rank_uuids=uuids,pci_by_uuid=pci)


def affinity():
    p=json.loads(os.environ['SIM10_GPU_PLAN']);mapping=json.loads(os.environ['SIM10_CPU_PLAN'])
    rank=int(os.environ['SLURM_PROCID']);host=socket.gethostname()
    require(host==p['rank_hosts'][rank] and len(mapping)==len(p['rank_hosts']),'Rank host/map')
    local=[r for r,h in enumerate(p['rank_hosts']) if h==host]
    physical=[]
    for r in local:
        require(len(mapping[r])==(3 if r<2 else 7),'Per-rank core demand')
        for cpu in mapping[r]:
            t=Path('/sys/devices/system/cpu/cpu%d/topology'%cpu)
            physical.append(((t/'physical_package_id').read_text(),(t/'core_id').read_text()))
    require(len(physical)==len(set(physical)),'Same-host core or SMT overlap')
    require(set(mapping[rank])<=set(os.sched_getaffinity(0)),'Outside Slurm affinity')
    os.sched_setaffinity(0,mapping[rank]);require(set(os.sched_getaffinity(0))==set(mapping[rank]),'Affinity readback')
    return mapping[rank]


def audit_placement(rows, placement):
    require([r['host'] for r in rows]==placement['rank_hosts'],'Host placement')
    require([r['affinity'] for r in rows]==placement['mapping'],'CPU placement')
    require(len({(r['host'],r['device']['pci']) for r in rows[2:]})==len(rows)-2,'Physical GPUs per host')
    counts={h:sum(len(r['timestamps']) for r in rows[2:] if r['host']==h) for h in placement['hosts']}
    require(all(len(r['timestamps'])>0 for r in rows[2:]),'Every BD exercised')
    return counts
