import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from audit import audit_case
from campaign import schedule,launch,deadline
from support import select_affinity,EVENT_BYTES,SHA,histogram
from nic import snapshot,delta,COUNTERS

def probes(tasks,cpus=None):
    cpus=list(range(128)) if cpus is None else cpus
    return [dict(rank=r,host='node',cpus=cpus,topology={str(c):['0',str(c%64)] for c in cpus}) for r in range(tasks)]

def receipt(gpus=4,policy='compact',case=None):
    case=next(c for c in schedule('pilot') if c['gpus']==gpus) if case is None else case
    case=copy.deepcopy(case);case['cpu_policy']=policy
    n=case['events'];mapping=select_affinity(probes(4*gpus+2),4*gpus,policy)
    files=[dict(path='/c1/f%d'%i,bytes=100,identity=dict(st_dev=2,st_ino=10+i)) for i in range(5)]
    info=dict(timestamps=list(range(n)),sets=[dict(count=c,files=[dict(f,path='/c%d/f%d'%(c,i)) for i,f in enumerate(files)]) for c in (1,2,4,8,16,32)])
    fds=[dict(path=f['path'],bytes=100,dev=2,ino=f['identity']['st_ino']) for f in files]
    callbacks=[[] for _ in range(4*gpus)]
    for i,t in enumerate(range(0,n,200)):
        stamps=list(range(t,min(t+200,n)));callbacks[i%len(callbacks)].append(dict(timestamps=stamps,size=len(stamps),stream=1))
    rows=[]
    for rank in range(4*gpus+2):
        row=dict(rank=rank,cleanup_complete=True,host='node',elapsed=10.,affinity=mapping[str(rank)],runtime={'psana':dict(path='/source/install_psana/psana.py')},timestamps=[],device=None)
        if rank>=2:
            cb=callbacks[rank-2];stamps=[t for c in cb for t in c['timestamps']];count=len(stamps);gpu=(rank-2)//4
            row.update(timestamps=stamps,callbacks=cb,cpu_reads=0,io=dict(requests=5*count,completed_requests=5*count,requested_bytes=count*EVENT_BYTES,completed_bytes=count*EVENT_BYTES,pressure_drains=0,read_size_histogram=histogram(count),range_sha256={f['path']:'digest' for f in files}),device=dict(uuid='GPU-%d'%gpu,pci='PCI-%d'%gpu,compat_mode='ON',effective=dict(num_threads=4,task_size=1<<20,bounce_buffer_size=16<<20,auto_direct_io_read=False)),inherited_uuids=['GPU-%d'%g for g in range(4)],drain=[dict(closed=True,pending=0,active=0,held=0)],admission=[dict(peers=4,budget=6*2**30)],handles=dict(dgram=fds,kvikio=[dict(primary=f,all_new_handles=[f]) for f in fds]),witness_values=count*96 if case['kind']=='tail' else 0,budget_high_water=[dict(limit=6*2**30,observations=2,peak_committed=100,peak_committed_plus_held=200)])
        rows.append(row)
    scan=[dict(path=f['path'],resident=0) for g in info['sets'] for f in g['files']]
    def nic(t,v):return dict(started_unix=t,finished_unix=t+.01,devices={'cxi0':dict(pci='/pci/0',counters={c:dict(value=v,kernel_timestamp=t) for c in COUNTERS})})
    before=nic(0.,100);after=nic(12.,200)
    r=dict(state='pass',execution_sha=SHA,parameters=case.copy(),events=n,bytes=n*EVENT_BYTES,seconds=10.,gib_s=n*EVENT_BYTES/2**30/10.,rows=rows,host='node',measurement=dict(started_unix=1.,finished_unix=11.,cache=dict(all30_scans=[scan,copy.deepcopy(scan)],selected=[dict(path=f['path'],resident=0) for f in files]),nic_before=before,nic_after=after,nic_delta=delta(before,after)))
    return r,case,info

class ScalingTests(unittest.TestCase):
    def test_profiles_and_strong_scaling(self):
        cases=schedule('screen');self.assertEqual(len(cases),12);self.assertEqual(cases,schedule('screen'))
        for block in (1,2,3):self.assertEqual(sorted(c['gpus'] for c in cases if c['block']==block),[1,2,3,4])
        self.assertTrue(all(c['events']==100000 and c['bds']==4*c['gpus'] and c['count']==1 for c in cases))
        self.assertEqual(len(schedule('pilot')),6)
    def test_affinity_no_shared_physical_cores(self):
        for g in range(1,5):
            mapping=select_affinity(probes(4*g+2),4*g);physical={}
            for rank,cpus in mapping.items():
                unique={c%64 for c in cpus};self.assertFalse(unique.intersection(physical));physical.update({c:rank for c in unique})
                self.assertEqual(len(cpus),3 if int(rank)<2 else 6)
            self.assertEqual(len(physical),6+12*g)
    def test_affinity_stable_prefix(self):
        largest=select_affinity(probes(18),16)
        for g in (1,2,3):
            small=select_affinity(probes(4*g+2),4*g)
            self.assertTrue(all(largest[k]==v for k,v in small.items()))
        legacy=select_affinity(probes(6),4,'legacy');self.assertEqual(sum(map(len,legacy.values())),34)
    def test_real_perlmutter_masks_preserve_shared_ranks(self):
        from support import campaign_affinity,HERE
        raw=json.loads((HERE/'evidence/pilot01-affinity-probes.json').read_text())
        common=set.intersection(*(set(p['cpus']) for group in raw.values() for p in group))
        self.assertEqual(len(common),106) # Actual failed pilot had 53 common cores.
        with self.assertRaises(ValueError):select_affinity([dict(p,cpus=sorted(common)) for p in raw['18']],16)
        maps=campaign_affinity(raw)
        for g in (1,2,3):
            self.assertTrue(all(maps[str(g)][r]==maps['4'][r] for r in maps[str(g)]))
        for g in range(1,5):
            available=set.intersection(*(set(p['cpus']) for p in raw[str(4*g+2)]))
            assigned=[c for cpus in maps[str(g)].values() for c in cpus]
            self.assertTrue(set(assigned)<=available)
            self.assertEqual(len({tuple(raw[str(4*g+2)][0]['topology'][str(c)]) for c in assigned}),6+12*g)
        # No shared-core relaxation: a common set below 42 still fails.
        short=copy.deepcopy(raw)
        for p in short['6']:
            p['cpus']=[c for c in p['cpus'] if int(p['topology'][str(c)][1])<40]
        with self.assertRaises(ValueError):campaign_affinity(short)

    def test_affinity_missing_smt_or_cores(self):
        for mask in (list(range(64)),list(range(50))+list(range(64,114))):
            with self.assertRaises(ValueError):select_affinity(probes(18,mask),16)
        with self.assertRaises(ValueError):select_affinity(probes(18),16,'legacy')
    def test_all_gpu_counts_accept_valid_receipts(self):
        for g in range(1,5):
            r,c,i=receipt(g);self.assertEqual(audit_case(r,c,i)['gpus'],g)
        r,c,i=receipt(1,'legacy');audit_case(r,c,i)
    def test_bad_placement_and_io_and_cache_rejected(self):
        mutations=[lambda r:r['rows'][6]['device'].update(uuid='GPU-0'),lambda r:r['rows'][6]['device'].update(pci='PCI-0'),lambda r:r['rows'][2]['admission'][0].update(peers=16),lambda r:r['rows'][2]['timestamps'].__setitem__(0,1),lambda r:r['rows'][2]['io'].update(completed_bytes=0),lambda r:r['rows'][2].update(witness_values=0),lambda r:r['rows'][2].update(cpu_reads=1),lambda r:r['rows'][2]['handles']['dgram'][0].update(ino=999),lambda r:r['rows'][2]['budget_high_water'][0].update(limit=24*2**30),lambda r:r['measurement']['cache']['selected'][0].update(resident=1),lambda r:r['rows'][2].update(affinity=r['rows'][3]['affinity']),lambda r:r['rows'][2]['device']['effective'].update(num_threads=8),lambda r:r['rows'][2].update(cleanup_complete=False)]
        for index,mutate in enumerate(mutations):
            with self.subTest(index=index):
                r,c,i=receipt();mutate(r)
                with self.assertRaises(ValueError):audit_case(r,c,i)
    def test_split_tail_and_empty_bd_rejected(self):
        r,c,i=receipt();row=next(row for row in r['rows'][2:] if row['callbacks'][-1]['size']==3)
        last=row['callbacks'].pop();row['callbacks'] += [dict(last,size=1,timestamps=last['timestamps'][:1]),dict(last,size=2,timestamps=last['timestamps'][1:])]
        with self.assertRaisesRegex(ValueError,'Globalcallbackhist'):audit_case(r,c,i)
        r,c,i=receipt();r['rows'][2]['timestamps']=[]
        with self.assertRaises(ValueError):audit_case(r,c,i)
    def test_nic_missing_and_reset_rejected(self):
        r,_,_=receipt();b=r['measurement']['nic_before'];a=r['measurement']['nic_after'];self.assertEqual(delta(b,a)['rx_bytes'],100)
        a['devices']['cxi0']['counters'][COUNTERS[0]]['value']=1
        with self.assertRaises(ValueError):delta(b,a)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(ValueError):snapshot(root)
            p=root/'cxi0/device/telemetry';p.mkdir(parents=True)
            for c in COUNTERS:(p/c).write_text('1234@100.123\n')
            self.assertEqual(snapshot(root)['devices']['cxi0']['counters'][COUNTERS[0]]['value'],1234)
    def test_nic_changed_or_unbracketed_receipt_rejected(self):
        r,c,i=receipt();r['measurement']['nic_delta']['rx_bytes']=0
        with self.assertRaises(ValueError):audit_case(r,c,i)
        r,c,i=receipt();r['measurement']['nic_before']['finished_unix']=2
        with self.assertRaises(ValueError):audit_case(r,c,i)
    def test_launch_limits_and_gpu_visibility(self):
        for tasks in (6,10,14,18):
            command=launch(tasks,'runner.py');self.assertIn('--cpu-bind=none',command);self.assertIn('--gpu-bind=none',command);cpus=int(next(x.split('=')[1] for x in command if x.startswith('--cpus-per-task=')))
            self.assertEqual(cpus%2,0);self.assertLessEqual(tasks*((cpus+1)//2),64);self.assertGreaterEqual(tasks*cpus//2,54)
        p=dict(mode='pilot',requested_seconds=2700)
        text='Account=lcls_g QOS=gpu_premium NumNodes=1 StartTime=2026-11-16T00:00:00 EndTime=2026-11-16T00:45:00'
        deadline(p,text)
        with self.assertRaises(ValueError):deadline(p,text.replace('00:45:00','01:45:00'))
        with self.assertRaises(ValueError):deadline(p,text.replace('gpu_premium','gpu_regular'))
    def test_no_screen_without_accepted_pilot(self):
        from prepare import validate_plan,prepare
        plan=dict(mode='pilot',commit='a'*40,requested_seconds=2700,per_case_timeout=600,pilot=None);validate_plan(plan)
        with self.assertRaises(ValueError):prepare('a'*40,'scale-test','screen')
        plan['requested_seconds']=9000
        with self.assertRaises(ValueError):validate_plan(plan)
    def test_memory_guards(self):
        from memory_watch import validate_memory
        r=dict(samples=[dict(time=t,total=40*2**30,free=30*2**30,used=10*2**30) for t in (1.,1.1,1.2)],errors=[],max_used=10*2**30,min_free=30*2**30)
        validate_memory(r,1.02,1.18)
        r['samples'][1]['free']=1
        with self.assertRaises(ValueError):validate_memory(r,1.02,1.18)

if __name__=='__main__':unittest.main(verbosity=2)
