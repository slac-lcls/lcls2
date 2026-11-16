"""Negative checks on copies of native framework evidence, after hardware run."""
import json,sys,tempfile,unittest
from pathlib import Path
from audit import audit
SOURCE=Path(sys.argv.pop(1))
class Audit(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        for p in SOURCE.glob('*.json'):(self.root/p.name).write_bytes(p.read_bytes())
    def tearDown(self):self.tmp.cleanup()
    def change(self,fn):
        p=self.root/'00-target-524288.json';r=json.loads(p.read_text());fn(r);p.write_text(json.dumps(r))
    def test_native(self):self.assertEqual(audit(self.root)['status'],'pass')
    def test_cached_page(self):
        self.change(lambda r:r['measurement']['initial_cache'][0].update(resident_pages=1))
        with self.assertRaisesRegex(ValueError,'Cold gate'):audit(self.root)
    def test_wrong_rate(self):
        self.change(lambda r:r.update(gib_s=r['gib_s']+1))
        with self.assertRaisesRegex(ValueError,'Rates'):audit(self.root)
    def test_short_gpu_io(self):
        self.change(lambda r:r['rows'][2]['io'].update(completed_requests=49999))
        with self.assertRaisesRegex(ValueError,'I/O count'):audit(self.root)
    def test_cpu_payload(self):
        self.change(lambda r:r['rows'][2].update(cpu_reads=1))
        with self.assertRaisesRegex(ValueError,'Pressure/CPU'):audit(self.root)
    def test_pressure(self):
        self.change(lambda r:r['rows'][2]['io'].update(pressure_drains=1))
        with self.assertRaisesRegex(ValueError,'Pressure/CPU'):audit(self.root)
    def test_missing_callback(self):
        self.change(lambda r:r['rows'][2]['callbacks'].pop())
        with self.assertRaisesRegex(ValueError,'Callback delivery'):audit(self.root)
    def test_nic_delta(self):
        self.change(lambda r:next(iter(r['measurement']['nic_delta'].values())).update(rx_bytes=-1))
        with self.assertRaisesRegex(ValueError,'NIC bracket/delta'):audit(self.root)
    def test_drain(self):
        self.change(lambda r:r['rows'][2]['drain'][0].update(held=1))
        with self.assertRaisesRegex(ValueError,'Resource drain'):audit(self.root)
    def test_batch_setting(self):
        self.change(lambda r:r['framework'].update(batch_size=7))
        with self.assertRaisesRegex(ValueError,'Framework settings'):audit(self.root)
    def test_sim10_reference(self):
        p=self.root/'inputs.json';r=json.loads(p.read_text());r['sim10']['timestamps'][0]+=1;p.write_text(json.dumps(r))
        with self.assertRaisesRegex(ValueError,'Sim10 reference'):audit(self.root)
    def test_histogram(self):
        self.change(lambda r:r['rows'][2]['io']['read_size_histogram'].clear())
        with self.assertRaisesRegex(ValueError,'Read histogram'):audit(self.root)
if __name__=='__main__':unittest.main()
