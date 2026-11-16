"""Negative checks for the separate historical-configuration control audit."""
import json,sys,tempfile,unittest
from pathlib import Path
from audit import audit
SOURCE=Path(sys.argv.pop(1));PRIMARY=Path(sys.argv.pop(1))
class Audit(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        for p in SOURCE.glob('*.json'):(self.root/p.name).write_bytes(p.read_bytes())
    def tearDown(self):self.tmp.cleanup()
    def change(self,fn):
        p=self.root/'legacy-raw-cold-r1.json';r=json.loads(p.read_text());fn(r);p.write_text(json.dumps(r))
    def test_native(self):self.assertEqual(audit(self.root,PRIMARY)['status'],'pass')
    def test_cold_gate(self):
        self.change(lambda r:r['measurement']['conditioning']['cache']['files'][0].update(resident_pages=1))
        with self.assertRaisesRegex(ValueError,'Cold gate'):audit(self.root,PRIMARY)
    def test_bytes(self):
        self.change(lambda r:r.update(bytes=r['bytes']-1))
        with self.assertRaisesRegex(ValueError,'I/O totals'):audit(self.root,PRIMARY)
    def test_rate(self):
        self.change(lambda r:r.update(gib_s=r['gib_s']+1))
        with self.assertRaisesRegex(ValueError,'Rates'):audit(self.root,PRIMARY)
    def test_node(self):
        self.change(lambda r:r['rows'][2].update(host='other-node'))
        with self.assertRaisesRegex(ValueError,'Same-node topology'):audit(self.root,PRIMARY)
if __name__=='__main__':unittest.main()
