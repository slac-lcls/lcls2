import copy,json,tempfile,unittest
from pathlib import Path
from audit import audit_case
from campaign import schedule
from instrument import fd_snapshot
from support import SHA,EVENT_BYTES,histogram,identity,fd_record,select_affinity

def fixture():
    paths=[{'path':'/count2/f%d'%i,'bytes':100,'identity':{'st_dev':2,'st_ino':i+10}} for i in range(5)]
    info={'timestamps':list(range(1000)),'sets':[{'count':c,'files':paths if c==2 else [dict(r,path='/count%d/f%d'%(c,i)) for i,r in enumerate(paths)]} for c in [1,2,4,8,16,32]]}
    case={'id':'pilot-b1-s2','count':2,'streams':2,'bds':1,'events':1000,'kind':'pilot'}
    rows=[]
    for rank in range(3):
        rows.append({'rank':rank,'role':['smd0','eb','bd'][rank],'cleanup_complete':True,'host':'node','elapsed':10.,'affinity':list(range(56)),'runtime':{'psana':{'path':'/source/install_psana/psana.py'}},'timestamps':[],'device':None})
    row=rows[2];fds=[{'path':r['path'],'bytes':100,'dev':2,'ino':r['identity']['st_ino'],'fd':i} for i,r in enumerate(paths)]
    row.update(timestamps=list(range(1000)),callbacks=[{'timestamps':list(range(i,i+100)),'size':100,'stream':i//100%2} for i in range(0,1000,100)],cpu_reads=0,io={'requests':5000,'completed_requests':5000,'requested_bytes':1000*EVENT_BYTES,'completed_bytes':1000*EVENT_BYTES,'pressure_drains':0,'read_size_histogram':histogram(1000),'range_sha256':{r['path']:'x' for r in paths}},device={'uuid':'GPU-x','compat_mode':'ON','effective':dict(num_threads=8,task_size=1<<20,bounce_buffer_size=16<<20,auto_direct_io_read=False)},drain=[dict(closed=True,pending=0,active=0,held=0)],inherited_uuids=['GPU-x'],admission=[{'peers':1,'budget':6*2**30}],handles={'dgram':fds,'kvikio':[{'primary':f,'all_new_handles':[f]} for f in fds]},witness_values=0)
    scan=[{'path':r['path'],'resident':0} for g in info['sets'] for r in g['files']]
    r={'state':'pass','execution_sha':SHA,'parameters':case.copy(),'events':1000,'bytes':1000*EVENT_BYTES,'seconds':10.,'gib_s':1000*EVENT_BYTES/2**30/10,'rows':rows,'host':'node','measurement':{'cache':{'all30_scans':[scan,copy.deepcopy(scan)],'selected':[{'path':r['path'],'resident':0} for r in paths]}}}
    return r,case,info
class AuditTests(unittest.TestCase):
    def test_valid_receipt(self):
        r,c,i=fixture();self.assertEqual(audit_case(r,c,i)['events'],1000)
    def test_empty_bds_keep_global_partition(self):
        r,c,i=fixture();c['bds']=4;r['parameters']['bds']=4
        r['rows'][2]['admission'][0]['peers']=4
        for rank in range(3,6):
            row=copy.deepcopy(r['rows'][2]);row.update(rank=rank,timestamps=[],callbacks=[],drain=[],admission=[],witness_values=0)
            row['handles']['kvikio']=[]
            row['io'].update(requests=0,completed_requests=0,requested_bytes=0,completed_bytes=0,read_size_histogram={},range_sha256={})
            r['rows'].append(row)
        self.assertEqual(audit_case(r,c,i)['events'],1000)
        r['rows'][2]['timestamps'].pop()
        with self.assertRaises(ValueError):audit_case(r,c,i)
    def test_corrupt_receipts(self):
        changes=[lambda r:r.update(execution_sha='wrong'),lambda r:r.update(bytes=1),lambda r:r.update(seconds=0),lambda r:r.update(gib_s=1),lambda r:r['rows'][2]['timestamps'].__setitem__(0,1),lambda r:r['rows'][2].update(cpu_reads=1),lambda r:r['rows'][2]['io'].update(completed_bytes=1),lambda r:r['rows'][2]['admission'][0].update(peers=4),lambda r:r['rows'][2]['handles']['dgram'][0].update(ino=999),lambda r:r['rows'][2].update(witness_values=5),lambda r:r['measurement']['cache']['selected'][0].update(resident=1),lambda r:r['measurement']['cache']['all30_scans'].pop(),lambda r:r['rows'][2].update(cleanup_complete=False)]
        for n,change in enumerate(changes):
            with self.subTest(mutation=n):
                r,c,i=fixture();change(r)
                with self.assertRaises(ValueError):audit_case(r,c,i)
    def test_schedule(self):
        s=schedule('screen');self.assertEqual(len(s),72);self.assertEqual(s,schedule('screen'))
        for block in range(3):self.assertEqual(len({(r['count'],r['streams'],r['bds']) for r in s if r['block']==block}),24)
        self.assertEqual(len(schedule('pilot')),14)
    def test_snapshot_excludes_closed_directory_handle(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'payload';p.write_bytes(b'x')
            before=fd_snapshot()
            self.assertFalse(any(path.startswith('/proc/') and '/fd/' in path for path in before.values()))
            with p.open('rb') as f:
                after=fd_snapshot()
                self.assertEqual(after[f.fileno()],str(p))
                self.assertNotEqual(before.get(f.fileno()),str(p))
    def test_common_affinity_across_slurm_shapes(self):
        a=list(range(60));b=list(range(30))+list(range(32,62))
        def probe(cpus):return {'host':'node','cpus':cpus,'topology':{str(c):['0',str(c%64)] for c in cpus}}
        probes=[probe(a),probe(b)]
        self.assertEqual(select_affinity(probes),list(range(30))+list(range(32,58)))
        with self.assertRaises(ValueError):select_affinity([probe(list(range(50)))])
    def test_real_wrong_fd(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'a';b=Path(d)/'b';a.write_bytes(b'a');b.write_bytes(b'b')
            row={'path':str(a),'identity':identity(a),'bytes':1}
            with a.open('rb') as fa,b.open('rb') as fb:
                self.assertEqual(fd_record(fa.fileno(),row)['ino'],a.stat().st_ino)
                with self.assertRaises(ValueError):fd_record(fb.fileno(),row)
if __name__=='__main__':unittest.main(verbosity=2)
