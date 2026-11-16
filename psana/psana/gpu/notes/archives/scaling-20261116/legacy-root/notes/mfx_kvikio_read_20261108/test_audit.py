import copy
import json
import os
from pathlib import Path
import unittest
from audit import check_case

class NativeReports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(os.environ['KVIKIO_READ_NATIVE_ROOT'])
        cls.native=json.loads((root/'anchor-first.json').read_text())
    def test_native(self):check_case(self.native)
    def test_bad_bytes(self):
        r=copy.deepcopy(self.native);r['total_bytes']+=1
        with self.assertRaisesRegex(ValueError,'Total accounting'):check_case(r)
    def test_wrong_effective(self):
        r=copy.deepcopy(self.native);r['effective']['num_threads']+=1
        with self.assertRaisesRegex(ValueError,'Effective settings'):check_case(r)
    def test_missing_witness(self):
        r=copy.deepcopy(self.native);r['validation']['witnesses'].pop()
        with self.assertRaisesRegex(ValueError,'Incomplete GPU witnesses'):check_case(r)
    def test_false_rate(self):
        r=copy.deepcopy(self.native);r['gib_s']*=2
        with self.assertRaisesRegex(ValueError,'Total rate'):check_case(r)

if __name__=='__main__':unittest.main()
