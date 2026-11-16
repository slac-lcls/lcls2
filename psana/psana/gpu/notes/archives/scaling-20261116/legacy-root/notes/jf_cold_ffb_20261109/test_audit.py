"""Negative acceptance checks applied to copies of actual native evidence."""
import json,sys,tempfile,unittest
from pathlib import Path
from audit import audit
SOURCE=Path(sys.argv.pop(1))
class Acceptance(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        for path in SOURCE.glob('*.json'):(self.root/path.name).write_bytes(path.read_bytes())
    def tearDown(self):self.tmp.cleanup()
    def mutate(self,name,fn):
        path=self.root/name;r=json.loads(path.read_text());fn(r);path.write_text(json.dumps(r))
    def test_accept_native(self):self.assertEqual(audit(self.root)['status'],'pass')
    def test_reject_one_cached_page(self):
        self.mutate('00-raw.json',lambda r:r['initial_cache'][0].update(resident_pages=1))
        with self.assertRaisesRegex(ValueError,'Cold gate'):audit(self.root)
    def test_reject_false_rate(self):
        self.mutate('01-sim10.json',lambda r:r.update(gib_s=r['gib_s']*1.1))
        with self.assertRaisesRegex(ValueError,'Rates'):audit(self.root)
    def test_reject_false_counter(self):
        self.mutate('00-raw.json',lambda r:next(iter(r['nic_delta'].values())).update(rx_bytes=next(iter(r['nic_delta'].values()))['rx_bytes']+1))
        with self.assertRaisesRegex(ValueError,'NIC delta'):audit(self.root)
    def test_reject_wrong_file(self):
        self.mutate('01-sim10.json',lambda r:r['initial_cache'][0].update(path='/tmp/wrong.xtc2'))
        with self.assertRaisesRegex(ValueError,'Cache file identity'):audit(self.root)
    def test_reject_short_count(self):
        self.mutate('03-raw.json',lambda r:r.update(requests=49999))
        with self.assertRaisesRegex(ValueError,'Totals'):audit(self.root)
    def test_reject_incomplete_cleanup(self):
        self.mutate('cleanup.json',lambda r:r.update(removed=False))
        with self.assertRaisesRegex(ValueError,'not cleaned'):audit(self.root)
if __name__=='__main__':unittest.main()
