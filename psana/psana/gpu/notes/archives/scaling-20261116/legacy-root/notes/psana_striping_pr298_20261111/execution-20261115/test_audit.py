import copy,json,tempfile,unittest
from pathlib import Path
from audit import audit_case
from campaign import schedule
from instrument import fd_snapshot,budget_peaks
from support import SHA,EVENT_BYTES,histogram,identity,fd_record,select_affinity,SOURCE

def fixture():
    paths=[{'path':'/count2/f%d'%i,'bytes':100,'identity':{'st_dev':2,'st_ino':i+10}} for i in range(5)]
    info={'timestamps':list(range(1000)),'sets':[{'count':c,'files':paths if c==2 else [dict(r,path='/count%d/f%d'%(c,i)) for i,r in enumerate(paths)]} for c in [1,2,4,8,16,32]]}
    case={'id':'pilot-b1-s2','count':2,'streams':1,'bds':4,'events':1000,'kind':'tail','batch_size':200,'workers_per_bd':4}
    rows=[]
    for rank in range(3):
        rows.append({'rank':rank,'role':['smd0','eb','bd'][rank],'cleanup_complete':True,'host':'node','elapsed':10.,'affinity':list(range(3)) if rank==0 else list(range(3,6)) if rank==1 else list(range(6,13)),'runtime':{'psana':{'path':'/source/install_psana/psana.py'}},'timestamps':[],'device':None})
    row=rows[2];fds=[{'path':r['path'],'bytes':100,'dev':2,'ino':r['identity']['st_ino'],'fd':i} for i,r in enumerate(paths)]
    row.update(timestamps=list(range(1000)),callbacks=[{'timestamps':list(range(i,i+200)),'size':200,'stream':0} for i in range(0,1000,200)],cpu_reads=0,io={'requests':5000,'completed_requests':5000,'requested_bytes':1000*EVENT_BYTES,'completed_bytes':1000*EVENT_BYTES,'pressure_drains':0,'read_size_histogram':histogram(1000),'range_sha256':{r['path']:'x' for r in paths}},device={'uuid':'GPU-x','compat_mode':'ON','effective':dict(num_threads=4,task_size=1<<20,bounce_buffer_size=16<<20,auto_direct_io_read=False)},drain=[dict(closed=True,pending=0,active=0,held=0)],inherited_uuids=['GPU-x'],admission=[{'peers':4,'budget':6*2**30}],handles={'dgram':fds,'kvikio':[{'primary':f,'all_new_handles':[f]} for f in fds]},witness_values=96000,budget_high_water=[dict(limit=6*2**30,peak_committed=100,peak_committed_plus_held=200,observations=5)])
    for rank in range(3,6):
        empty=copy.deepcopy(row);empty.update(rank=rank,affinity=list(range(6+7*(rank-2),13+7*(rank-2))),timestamps=[],callbacks=[],drain=[],admission=[],witness_values=0,budget_high_water=[])
        empty['handles']['kvikio']=[];empty['io'].update(requests=0,completed_requests=0,requested_bytes=0,completed_bytes=0,read_size_histogram={},range_sha256={})
        rows.append(empty)
    scan=[{'path':r['path'],'resident':0} for g in info['sets'] for r in g['files']]
    r={'state':'pass','execution_sha':SHA,'parameters':case.copy(),'events':1000,'bytes':1000*EVENT_BYTES,'seconds':10.,'gib_s':1000*EVENT_BYTES/2**30/10,'rows':rows,'host':'node','measurement':{'cache':{'all30_scans':[scan,copy.deepcopy(scan)],'selected':[{'path':r['path'],'resident':0} for r in paths]}}}
    return r,case,info
class AuditTests(unittest.TestCase):
    def test_valid_receipt(self):
        r,c,i=fixture();self.assertEqual(audit_case(r,c,i)['events'],1000)
    def test_empty_bds_keep_global_partition(self):
        r,c,i=fixture();self.assertEqual(audit_case(r,c,i)['events'],1000)
        r['rows'][2]['timestamps'].pop()
        with self.assertRaises(ValueError):audit_case(r,c,i)
    def test_corrupt_receipts(self):
        changes=[lambda r:r.update(execution_sha='wrong'),lambda r:r.update(bytes=1),lambda r:r.update(seconds=0),lambda r:r.update(gib_s=1),lambda r:r['rows'][2]['timestamps'].__setitem__(0,1),lambda r:r['rows'][2].update(cpu_reads=1),lambda r:r['rows'][2]['io'].update(completed_bytes=1),lambda r:r['rows'][2]['admission'][0].update(peers=1),lambda r:r['rows'][2]['handles']['dgram'][0].update(ino=999),lambda r:r['rows'][2].update(witness_values=5),lambda r:r['measurement']['cache']['selected'][0].update(resident=1),lambda r:r['measurement']['cache']['all30_scans'].pop(),lambda r:r['rows'][2].update(cleanup_complete=False)]
        for n,change in enumerate(changes):
            with self.subTest(mutation=n):
                r,c,i=fixture();change(r)
                with self.assertRaises(ValueError):audit_case(r,c,i)
    def test_schedule(self):
        s=schedule('screen');self.assertEqual(len(s),18);self.assertEqual(s,schedule('screen'))
        for block in (1,2,3):self.assertEqual(sorted(r['count'] for r in s if r['block']==block),[1,2,4,8,16,32])
        self.assertEqual(len(schedule('qualify')),8)
    def test_snapshot_excludes_closed_directory_handle(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'payload';p.write_bytes(b'x')
            before=fd_snapshot()
            self.assertFalse(any(path.startswith('/proc/') and '/fd/' in path for path in before.values()))
            with p.open('rb') as f:
                after=fd_snapshot()
                self.assertEqual(after[f.fileno()],str(p))
                self.assertNotEqual(before.get(f.fileno()),str(p))
    def test_disjoint_affinity_from_actual_masks(self):
        def probe(rank,cpus):return {'rank':rank,'host':'node','cpus':cpus,'topology':{str(c):['0',str(c%64)] for c in cpus}}
        probes=[probe(rank,list(range(60)) if rank%2 else list(range(30))+list(range(32,62))) for rank in range(6)]
        mapping=select_affinity(probes);flat=[c for v in mapping.values() for c in v]
        self.assertEqual([len(mapping[str(i)]) for i in range(6)],[3,3,7,7,7,7])
        self.assertEqual(len(set(flat)),34);self.assertNotIn(30,flat);self.assertNotIn(31,flat)
        with self.assertRaises(ValueError):select_affinity([probe(rank,list(range(33))) for rank in range(6)])
        with self.assertRaises(ValueError):select_affinity([probe(rank,list(range(16))+list(range(64,80))) for rank in range(6)])
    def test_reject_new_profile_corruption(self):
        mutations=[lambda r:r['parameters'].update(batch_size=100),lambda r:r['rows'][3].update(affinity=r['rows'][2]['affinity']),lambda r:r['rows'][2]['device']['effective'].update(num_threads=8),lambda r:r['rows'][2]['budget_high_water'][0].update(peak_committed_plus_held=7*2**30)]
        for change in mutations:
            r,c,i=fixture();change(r)
            with self.assertRaises(ValueError):audit_case(r,c,i)
    def test_memory_sample_rejections(self):
        from memory_watch import validate_memory
        r=dict(samples=[dict(time=t,total=40*2**30,free=30*2**30,used=10*2**30) for t in (1.,1.1,1.2)],errors=[],max_used=10*2**30,min_free=30*2**30)
        validate_memory(r,1.02,1.18)
        for change in (lambda x:x['samples'][1].update(free=1),lambda x:x['samples'][1].update(time=5),lambda x:x.update(errors=['NVML failure']),lambda x:x.update(max_used=0)):
            bad=copy.deepcopy(r);change(bad)
            with self.assertRaises(ValueError):validate_memory(bad,1.02,1.18)
        with self.assertRaises(ValueError):validate_memory(r,0.,1.18)
    def test_exact_budget_peak_hooks(self):
        import importlib.util,sys
        from unittest.mock import patch
        spec=importlib.util.spec_from_file_location('psana.gpu.gpu_budget',SOURCE/'psana/psana/gpu/gpu_budget.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        with patch.dict(sys.modules,{'psana.gpu.gpu_budget':mod}):
            with budget_peaks() as peaks:
                b=mod._GpuBudget(1000);b.reserve(100);hold=b.hold(700)
                with hold:b.reserve(500)
                hold.close();b.release(600)
                self.assertEqual(b.committed(),0);self.assertEqual(b._held,0)
            r=list(peaks.values())[0];self.assertEqual(r['peak_committed'],600);self.assertEqual(r['peak_committed_plus_held'],800)
    def test_allocation_budget_guard(self):
        from launch_plan import make_plan,check_allocation,BASE_JOBS
        records=[dict(JobID=j,Account='lcls_g',NNodes='1',State='COMPLETED',ElapsedRaw=str(n)) for j,n in zip(sorted(BASE_JOBS),(908,277,930,927))]
        p=make_plan('qualify',records);self.assertEqual(p['prior_node_seconds'],3042)
        check_allocation(p,'TimeLimit=00:45:00','qualify')
        with self.assertRaises(ValueError):check_allocation(p,'TimeLimit=01:00:00','qualify')
        with self.assertRaises(ValueError):make_plan('qualify',records[:-1])
        with self.assertRaises(ValueError):make_plan('screen',records)
        records[0]['ElapsedRaw']='20000'
        with self.assertRaises(ValueError):make_plan('qualify',records)
    def test_real_wrong_fd(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'a';b=Path(d)/'b';a.write_bytes(b'a');b.write_bytes(b'b')
            row={'path':str(a),'identity':identity(a),'bytes':1}
            with a.open('rb') as fa,b.open('rb') as fb:
                self.assertEqual(fd_record(fa.fileno(),row)['ino'],a.stat().st_ino)
                with self.assertRaises(ValueError):fd_record(fb.fileno(),row)
if __name__=='__main__':unittest.main(verbosity=2)
