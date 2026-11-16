"""Negative audit checks against the three native measured reports."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from summarize import summarize

@unittest.skipUnless(os.environ.get('KVIKIO_NETWORK_NATIVE_ROOT'), 'native reports not supplied')
class SummaryAudit(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        for i in range(1,4):
            name='network-r%d.json'%i
            (self.root/name).write_bytes((Path(os.environ['KVIKIO_NETWORK_NATIVE_ROOT'])/name).read_bytes())
    def corrupt(self, fn):
        p=self.root/'network-r1.json';r=json.loads(p.read_text());fn(r);p.write_text(json.dumps(r))
        with self.assertRaises(ValueError): summarize(self.root)
    def test_native_reports(self):
        self.assertEqual(summarize(self.root)['status'],'pass')
    def test_wrong_nic_delta(self):
        def change(r):
            d=r['passes'][0]['network']['nic_delta'];d[next(iter(d))]['rx_bytes']+=1
        self.corrupt(change)
    def test_false_cold(self):
        self.corrupt(lambda r:r['passes'][0]['network']['cache_before'].update(resident_pages=1))
    def test_wrong_read_bytes(self):
        self.corrupt(lambda r:r['passes'][0].update(bytes=100))
