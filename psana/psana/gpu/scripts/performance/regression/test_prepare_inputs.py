"""Offline preparation integrity tests; no psana import or full-size data."""
import copy
import hashlib
import mmap
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import native_contract as n
import prepare_inputs as p


def frame(timestamp, service, payload=b''):
    return struct.pack('<6I', timestamp & 0xffffffff, timestamp >> 32, service << 24, 0, 0, 12+len(payload)) + payload


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def fixture(self):
        first, gap, last = frame(100,12,b'first'), frame(101,10), frame(102,11,b'last')
        path = self.root/'source'
        path.write_bytes(first+gap+last)
        rows = [[100,0,0,len(first),12],[102,0,len(first)+len(gap),len(last),11]]
        entry = dict(path=str(path), **p.stamp(path), ranges=p.merge_ranges(rows), range_bytes=len(first)+len(last))
        return entry, rows, first+last

    def test_disjoint_hash_copy_and_header_identity(self):
        entry, rows, selected = self.fixture()
        result, receipt = p.copy_checked(entry,self.root/'copy',rows)
        self.assertEqual(result['sha256'],hashlib.sha256(selected).hexdigest())
        self.assertEqual(result['range_bytes'],len(selected))
        self.assertEqual(receipt['source_sha256'],receipt['copy_sha256'])
        self.assertEqual((self.root/'copy').stat().st_mode & 0o777,0o444)
        self.assertEqual(n.input_ranges(result),entry['ranges'])

    def test_mutated_source_fails_before_copy(self):
        entry, rows, _ = self.fixture()
        Path(entry['path']).write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError,'source changed'):
            p.copy_checked(entry,self.root/'copy',rows)
        self.assertFalse((self.root/'copy').exists())

    def test_smd_bigdata_disagreement_rejected(self):
        entry, rows, _ = self.fixture()
        rows[0][0] += 1
        with self.assertRaisesRegex(ValueError,'header mismatch'):
            p.copy_checked(entry,self.root/'copy',rows)

    def test_copy_verification_rejects_corruption(self):
        entry, rows, _ = self.fixture()
        with patch.object(p,'hash_contents',return_value=('bad','bad')):
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                p.copy_checked(entry,self.root/'copy',rows)

    def test_range_validation_and_old_schema(self):
        entry, _, _ = self.fixture()
        self.assertEqual(n.input_ranges(dict(range_start=4,range_bytes=8,size=12)),[[4,8]])
        for ranges in ([], [[0,0]], [[0,10],[9,2]], [[0,1000]], [[0,True]], [[0,1]]):
            bad = dict(entry,ranges=ranges)
            with self.assertRaises(ValueError): n.input_ranges(bad)
        with self.assertRaisesRegex(ValueError,'ambiguous'):
            n.input_ranges(dict(entry,range_start=0))

    def test_cache_counts_exact_bytes_with_shared_boundary_page(self):
        entry, _, _ = self.fixture()
        path = Path(entry['path'])
        with path.open('rb') as stream:
            mapping = mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ)
            self.assertEqual(mapping[0],100)
            result = n.cache_ranges(dict(files=[dict(bigdata=entry)]))
            mapping.close()
        self.assertEqual(result['measured_bytes'],entry['range_bytes'])
        self.assertEqual(result['resident_bytes_before'],entry['range_bytes'])
        self.assertEqual(len(result['files']),2)

    def test_headers_and_ranges_reject_truncation_overlap(self):
        with self.assertRaisesRegex(ValueError,'short XTC'): p.header(b'x')
        with self.assertRaisesRegex(ValueError,'extent'): p.header(bytes(24))
        with self.assertRaisesRegex(ValueError,'overlapping'):
            p.merge_ranges([[1,0,0,24,12],[2,0,23,24,12]])

    def test_hash_crosses_block_boundary(self):
        entry, _, selected = self.fixture()
        with patch.object(p,'BLOCK',7):
            full, subset = p.hash_contents(Path(entry['path']),entry['ranges'])
        self.assertEqual(subset,hashlib.sha256(selected).hexdigest())
        self.assertEqual(full,n.file_hash(entry['path']))

    def test_compute_guard_and_existing_target(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(ValueError,'compute allocation'):
                p.prepare(self.root/'missing','a'*64,self.root/'target')
        entry, rows, _ = self.fixture()
        target=self.root/'copy'; target.write_bytes(b'keep')
        with self.assertRaises(FileExistsError): p.copy_checked(entry,target,rows)
        self.assertEqual(target.read_bytes(),b'keep')


if __name__ == '__main__':
    unittest.main()
