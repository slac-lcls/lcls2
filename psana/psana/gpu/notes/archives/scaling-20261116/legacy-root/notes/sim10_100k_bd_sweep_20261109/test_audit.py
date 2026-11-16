"""Negative acceptance checks on copies of native sweep receipts."""
import json,sys,tempfile,unittest
from pathlib import Path
from audit import audit
SOURCE=Path(sys.argv.pop(1))
class Audit(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        for p in SOURCE.glob('*.json'):(self.root/p.name).write_bytes(p.read_bytes())
    def tearDown(self):self.tmp.cleanup()
    def mutate(self,fn):
        p=self.root/'03-bd4.json';r=json.loads(p.read_text());fn(r);p.write_text(json.dumps(r))
    def rejects(self,fn,pattern):
        self.mutate(fn)
        with self.assertRaisesRegex(ValueError,pattern):audit(self.root)
    def test_native(self):self.assertEqual(audit(self.root)['status'],'pass')
    def test_cache(self):self.rejects(lambda r:r['measurement']['initial_cache'][0].update(resident_pages=1),'Cold gate')
    def test_rate(self):self.rejects(lambda r:r.update(gib_s=r['gib_s']+1),'Rates')
    def test_partition(self):self.rejects(lambda r:r['rows'][2]['timestamps'].pop(),'Global partition')
    def test_callback(self):self.rejects(lambda r:r['rows'][2]['callbacks'].pop(),'Callback identity')
    def test_gpu(self):self.rejects(lambda r:r['rows'][3]['device'].update(pci='wrong'),'GPU identity')
    def test_peer(self):self.rejects(lambda r:r['rows'][2]['drain'][0].update(peers=1),'Drain/peer')
    def test_io(self):self.rejects(lambda r:r['rows'][2]['io'].update(completed_requests=0),'I/O count')
    def test_histogram(self):self.rejects(lambda r:r['rows'][2]['io']['read_size_histogram'].clear(),'I/O histogram')
    def test_pressure(self):self.rejects(lambda r:r['rows'][2]['io'].update(pressure_drains=1),'Pressure/CPU')
    def test_cpu(self):self.rejects(lambda r:r['rows'][2].update(cpu_reads=1),'Pressure/CPU')
    def test_streams(self):
        def change(r):
            for c in r['rows'][2]['callbacks']:c['stream']=1
        self.rejects(change,'Execution streams')
    def test_nic(self):self.rejects(lambda r:next(iter(r['measurement']['nic_delta'].values())).update(rx_bytes=0),'NIC bracket/delta')
    def test_settings(self):self.rejects(lambda r:r['framework'].update(streams=1),'Framework settings')
    def metadata(self,name,fn,pattern):
        p=self.root/name;r=json.loads(p.read_text());fn(r);p.write_text(json.dumps(r))
        with self.assertRaisesRegex(ValueError,pattern):audit(self.root)
    def test_source_provenance(self):self.metadata('provenance.json',lambda r:r['source_sha256'].clear(),'Source provenance')
    def test_staged_set(self):self.metadata('provenance.json',lambda r:r['staged'].pop(),'Staged file set')
    def test_fixture_identity(self):self.metadata('inputs.json',lambda r:r['timestamps'].__setitem__(0,0),'Fixture identity')
    def test_matrix(self):self.metadata('complete.json',lambda r:r['cases'].pop(),'Case matrix')
    def test_cleanup(self):self.metadata('cleanup.json',lambda r:r.update(removed=False),'Incomplete sweep/cleanup')
    def test_timer(self):self.rejects(lambda r:r['measurement'].update(seconds=r['seconds']+1),'Timer identity')
    def test_runtime(self):self.rejects(lambda r:r['rows'][2]['runtime_files'].clear(),'CPU/runtime changed')
    def test_helper(self):self.rejects(lambda r:r['rows'][0]['timestamps'].append(0),'Helper data')
if __name__=='__main__':unittest.main()
