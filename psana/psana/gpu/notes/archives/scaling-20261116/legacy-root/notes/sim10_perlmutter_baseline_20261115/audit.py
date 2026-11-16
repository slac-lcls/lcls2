"""Independent receipt checks, including physical ranges and GPU cube receipts."""
import hashlib
import json
import math
from pathlib import Path
import struct
from support import require, EVENT_BYTES, histogram, sha, normalize_pci

def audit_case(report, case, info, ranges, mapping, gpu_plan):
    require(report['status']=='pass' and report['stage']==case['stage'] and report['kind']==case['kind'], 'Case identity')
    require(report['events']==case['events'] and report['active_gpus']==case['gpus'], 'Case size/placement')
    require(math.isfinite(report['seconds']) and report['seconds']>0, 'Finite processing time')
    require(math.isclose(report['hz'],case['events']/report['seconds'],rel_tol=1e-12), 'Rate')
    require(math.isclose(report['payload_gib_s'],case['events']*EVENT_BYTES/2**30/report['seconds'],rel_tol=1e-12), 'Payload rate')
    rows=report['rows']
    require([r['rank'] for r in rows]==list(range(case['bds']+2)), 'Rank coverage')
    require(len({r['host'] for r in rows})==1, 'One node')
    require([r['affinity'] for r in rows]==mapping, 'Physical CPU map')
    require(all(not r['timestamps'] and r['device'] is None for r in rows[:2]), 'Helper roles')
    stamps=info['timestamps'][:case['events']]
    merged=[t for r in rows[2:] for t in r['timestamps']]
    require(len(set(merged))==len(merged)==case['events'] and sorted(merged)==stamps, 'Global event union')
    index={t:i for i,t in enumerate(info['timestamps'])}
    for row in rows[2:]:
        count=len(row['timestamps']);io=row['io']
        require(row['device']['mask']==gpu_plan['rank_uuids'][row['rank']], 'GPU UUID map')
        require(row['device']['mask'] in row['granted_uuids'], 'Granted GPU')
        require(normalize_pci(row['device']['pci'])==gpu_plan['pci_by_uuid'][row['device']['mask']], 'UUID/PCI hardware mapping')
        require(io['completed_requests']==io['requests']==5*count and io['completed_bytes']==io['requested_bytes']==EVENT_BYTES*count, 'Exact byte counts')
        require(io['read_size_histogram']==(histogram(count) if count else {}), 'Read sizes')
        require(row['errors']==row['cpu_reads']==io['pressure_drains']==0, 'Errors/fallback/drains')
        require(row['drain']==dict(closed=True,pending=0,active=0,held=0,peers=case['bds']//case['gpus']), 'Final resource drain')
        require([t for c in row['callbacks'] for t in c['timestamps']]==row['timestamps'], 'Callback coverage')
        require(all(c['size']==len(c['timestamps']) and 0<c['size']<=case['batch'] for c in row['callbacks']), 'Batch size')
        require(len({c['stream'] for c in row['callbacks']})<=1, 'Single stream')
        wanted={}
        for path in info['files']:
            record=ranges[Path(path).name];digest=hashlib.sha256()
            for timestamp in row['timestamps']:
                digest.update(struct.pack('<QQQ',timestamp,record['offsets'][index[timestamp]],record['size']))
            if count:wanted[path]=digest.hexdigest()
        require(io['range_sha256']==wanted, 'Independent physical range digest')
        require(all(m['free']>=4<<30 for m in row['sampled_memory']), 'Physical headroom')
        require(row['task_copy']['bytes']==(0 if case['stage']=='S' else count*128), 'Publication byte total')
        require(row['task_copy']['ordinary_host_fallbacks']==0 and row['task_copy']['pinned_capacity_high']<=64<<20, 'Pinned output ceiling')
        require(row['array_weakref_probe']['released'] and all(row['optional_shared_resources_absent'].values()), 'Ownership/sharing')
        require(all(p['peak_committed_plus_held']<=6<<30 for p in row['budget_peaks']), 'Framework budget')
    require(len({r['device']['pci'] for r in rows[2:]})==case['gpus'], 'Physical GPU count')
    require(all(c['resident_pages']==0 for c in report['observation']['initial_cache']), 'Local cold cache')
    require(report['cube_policy']=='gpu_shape_finite_nonzero_v1' and
            report['cube_exported'] is False and report['numerical_reference_checked'] is False,
            'GPU-only validation policy')
    require(all(row['cube_verification'] is None for row in rows[:2]), 'Helpers own no cube')
    for row in rows[2:]:
        if case['stage']=='ECC':
            audit_cube(row['cube_verification'], row['timestamps'], info['timestamps'][:case['events']])
        else:
            require(row['cube_verification'] is None, 'Non-ECC cube absent')
    if case['kind']=='acceptance':
        sizes=[c['size'] for row in rows[2:] for c in row['callbacks']]
        require(case['batch'] in sizes and any(size<case['batch'] for size in sizes), 'Full batch and tail')
    return dict(status='pass',events=case['events'],logical_payload_bytes=case['events']*EVENT_BYTES)


def audit_cube(receipt, delivered, timestamps):
    require(receipt is not None and receipt['status']=='pass', 'GPU cube receipt')
    require(receipt['shape']==[100,32,512,1024] and receipt['dtype']=='float32' and
            receipt['bytes']==100*32*512*1024*4, 'GPU cube dimensions')
    require(receipt['counts_shape']==[100] and receipt['counts_dtype']=='uint64', 'Cube count dimensions')
    ordinal={timestamp:index for index,timestamp in enumerate(timestamps)}
    wanted=[0]*100
    for timestamp in delivered:wanted[ordinal[timestamp]%99]+=1
    require(receipt['counts']==wanted and len(receipt['bins'])==100, 'Per-BD ordinal counts')
    for index,record in enumerate(receipt['bins']):
        require(record==dict(bin=index,finite=True,nonzero=wanted[index]>0), 'Finite/populated/empty bin')
    require(receipt['drained'] is True and receipt['cube_payload_host_bytes']==0 and
            receipt['exported'] is False and receipt['reference_checked'] is False, 'GPU-only cube checks')
    require(0<=receipt['seconds'] and math.isfinite(receipt['seconds']), 'Finite verification time')


if __name__=='__main__':
    import sys
    from support import write
    context=json.loads(Path(sys.argv[1]).read_text())
    report=json.loads(Path(sys.argv[2]).read_text())
    info=json.loads(Path(context['inputs']).read_text())
    ranges=json.loads(Path(context['ranges']).read_text())
    verdict=audit_case(report,context['case'],info,ranges,context['mapping'],context['gpu_plan'])
    write(Path(sys.argv[3]),verdict)
