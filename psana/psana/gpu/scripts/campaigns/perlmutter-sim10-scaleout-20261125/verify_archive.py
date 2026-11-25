"""Offline verification of compact evidence; no Slurm, CUDA, or raw scratch access."""
import argparse,csv,hashlib,json,math,statistics
from collections import Counter,defaultdict
from pathlib import Path

def verify(repo):
    gpu=repo/'psana/psana/gpu'
    doc=gpu/'docs/performance/campaigns/perlmutter/sim10-xpp-cube/2026-11-25-scaleout-closeout'
    arc=gpu/'notes/archives/sim10-scaleout-20261125'
    rows=list(csv.DictReader((doc/'runs.csv').open()))
    cases=[json.loads(line) for line in (arc/'cases.jsonl').read_text().splitlines()]
    byid={c['case_id']:c for c in cases}
    assert len(byid)==len(rows)==159
    assert Counter(int(r['campaign_stage']) for r in rows)=={1:39,2:72,3:48}
    assert sum(r['primary']=='True' for r in rows)==108
    for r in rows:
        c=byid[r['case_id']];ev=int(r['events_global']);seconds=float(r['loop_seconds'])
        assert seconds>0 and ev==sum(x['events'] for x in c['ranks'])
        assert c['audit']['status']=='pass' and c['audit']['events']==ev
        assert int(r['requested_bytes'])==ev*3358200
        assert math.isclose(float(r['events_per_second']),ev/seconds,rel_tol=1e-12)
        assert math.isclose(float(r['payload_gib_s']),ev*3358200/2**30/seconds,rel_tol=1e-12)
        ranks=c['ranks'];assert sorted(x['rank'] for x in ranks)==list(range(int(r['bds'])+2))
        assert all(x['events']>0 and x['device'] for x in ranks if x['rank']>=2)
        assert all(not x['errors'] for x in ranks)
        if int(r['campaign_stage'])==3:
            hosts=r['node'].split(';');n=int(r['nodes']);assert len(hosts)==n
            expected=[hosts[0]]*2+[h for h in hosts for _ in range(4)]
            physical=defaultdict(set);gpus=defaultdict(set)
            for x in ranks:
                rank=x['rank'];h=x['host'];assert h==expected[rank]
                cores={tuple(z) for z in x['physical_cores']};assert len(cores)==(3 if rank<2 else 7)
                assert not physical[h]&cores;physical[h].update(cores)
                if rank>=2:
                    dev=x['device'];assert dev['pci'] not in gpus[h];gpus[h].add(dev['pci'])
            assert all(len(v)==4 for v in gpus.values()) and len(gpus)==n
    summary=json.loads((doc/'summary.json').read_text())['results']
    for s in summary:
        matching=[r for r in rows if r['primary']=='True' and int(r['campaign_stage'])==s['campaign_stage'] and int(r['nodes'])==s['nodes'] and int(r['bds'])==s['bds'] and r['runtime']==s['runtime'] and r['stage']==s['stage']]
        rates=[float(r['events_per_second']) for r in matching];assert len(rates)==3
        assert math.isclose(statistics.median(rates),s['median_events_per_second'],rel_tol=1e-12)
        if s['campaign_stage']==3:
            one=next(x for x in summary if x['campaign_stage']==3 and x['nodes']==1 and x['stage']==s['stage'])
            assert math.isclose(s['weak_efficiency_percent'],100*s['median_events_per_second']/(s['nodes']*one['median_events_per_second']),rel_tol=1e-12)
    # Recalculate same-allocation P/C ratios separately from ratios of median rates.
    paired={}
    for b in (1,4):
        for stage in ('S','E','EC','ECC'):
            ratios=[]
            for block in (1,2,3):
                rr=[r for r in rows if r['campaign_stage']=='2' and r['primary']=='True' and int(r['bds'])==b and int(r['block'])==block and r['stage']==stage]
                assert len(rr)==2 and len({r['node'] for r in rr})==1
                times={r['runtime']:float(r['loop_seconds']) for r in rr};ratios.append(times['C']/times['P'])
            paired[str(b)+'BD-'+stage]=100*(statistics.median(ratios)-1)
    maps=0
    for g in json.loads((arc/'groups.json').read_text()):
        mapping=json.loads((repo/g['bundle_map']).read_text())
        for rel,item in mapping.items():
            p=repo/item['path'];data=p.read_bytes();assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
            if p.suffix=='.py':compile(data,str(p),'exec')
        frozen=json.loads((repo/mapping['bundle.json']['path']).read_text())
        for rel,digest in frozen['files'].items():assert mapping[rel]['sha256']==digest
        maps+=1
    assert maps==12
    accounting=(arc/'scheduler.txt').read_text().splitlines()
    for line in accounting[1:]:
        fields=line.split('|');assert fields[2]=='COMPLETED' and fields[3]=='0:0'
    devices=0
    for p in (arc/'groups').glob('*/qualification-receipts.json'):
        for name,item in json.loads(p.read_text()).items():
            d=item['receipt']
            if name.startswith('device-') and not name.endswith('.guards.json'):
                assert d['exitcode']==0
                calls=[x for x in d['reports'] if x['when']=='call'];assert len(calls)==len(d['collected'])
                assert all(x['outcome']=='passed' for x in calls);devices+=1
    result={'status':'pass','cases':len(rows),'primary_cases':108,'groups':maps,'device_receipts':devices,'paired_time_change_percent':paired,'scope':'compact offline evidence, script hashes/syntax, numeric replay and placement; not a fresh raw/pixel audit'}
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repository',type=Path,required=True);a=p.parse_args();verify(a.repository.resolve())
