import hashlib
import contextlib
import io
import json
import sys
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import stage


class StagingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.src = self.root / 'source'
        self.dst = self.root / 'dest'
        self.data = b'payload' * 113 + b'tail'
        self.src.write_bytes(self.data)
        self.expected = {'bytes': len(self.data), 'sha256': hashlib.sha256(self.data).hexdigest()}
        self.layouts = {}

    def tearDown(self):
        self.tmp.cleanup()

    def command(self, *args):
        if args[1] == 'setstripe':
            path = Path(args[-1]); count = int(args[3])
            with path.open('xb'):
                pass
            self.layouts[str(path)] = count
            return ''
        path = args[-1]; count = self.layouts[path]
        return (f'{path}\nlmm_stripe_count: {count}\nlmm_stripe_size: 1048576\n'
                'lmm_pool: original\nlmm_pattern: raid0\n' +
                ''.join(f'  {i}  100  0x64  0x100\n' for i in range(count)))

    def copy(self):
        with patch.object(stage, 'command', self.command), patch.object(stage, 'CHUNK', 71):
            return stage.copy_one(self.src, self.dst, self.expected, 4)

    def test_full_copy_nondivisible_tail_and_independent_hash(self):
        original = stage.identity(self.src)
        receipt = self.copy()
        self.assertEqual(self.dst.read_bytes(), self.data)
        self.assertEqual(receipt['layout']['osts'], [0, 1, 2, 3])
        self.assertEqual(stage.digest(self.dst, self.expected)['sha256'], self.expected['sha256'])
        self.assertEqual(stage.identity(self.src), original)

    def test_partial_write_retried(self):
        real_write = os.write
        with patch.object(stage.os, 'write', lambda fd, data: real_write(fd, data[:13])):
            self.copy()
        self.assertEqual(self.dst.read_bytes(), self.data)

    def test_zero_write_fails(self):
        with patch.object(stage.os, 'write', return_value=0), self.assertRaisesRegex(RuntimeError, 'zero write'):
            self.copy()

    def test_existing_destination_preserved(self):
        self.dst.write_bytes(b'keep')
        with self.assertRaisesRegex(RuntimeError, 'destination exists'):
            self.copy()
        self.assertEqual(self.dst.read_bytes(), b'keep')

    def test_symlink_destination_preserved(self):
        self.dst.symlink_to(self.src)
        with self.assertRaises(RuntimeError):
            self.copy()
        self.assertEqual(self.src.read_bytes(), self.data)

    def test_bad_source_size_fails_before_create(self):
        self.expected['bytes'] += 1
        with self.assertRaisesRegex(RuntimeError, 'source size'):
            self.copy()
        self.assertFalse(self.dst.exists())

    def test_corruption_detected(self):
        self.copy()
        self.dst.write_bytes(b'x' + self.data[1:])
        with self.assertRaisesRegex(RuntimeError, 'SHA256 mismatch'):
            stage.digest(self.dst, self.expected)

    def test_truncation_detected(self):
        self.copy()
        self.dst.write_bytes(self.data[:-1])
        with self.assertRaisesRegex(RuntimeError, 'byte count'):
            stage.digest(self.dst, self.expected)

    def test_layout_wrong_count_size_pool_and_pattern_rejected(self):
        self.copy()
        raw = self.command(stage.LFS, 'getstripe', str(self.dst))
        for old, new in [('count: 4', 'count: 2'), ('1048576', '2097152'),
                         ('original', 'other'), ('raid0', 'other')]:
            with self.subTest(old=old), patch.object(stage, 'command', return_value=raw.replace(old, new)):
                with self.assertRaises(RuntimeError):
                    stage.layout(self.dst, 4)

    def test_atomic_record(self):
        path = self.root / 'record.json'
        stage.save(path, {'state': 'verified'})
        self.assertIn('verified', path.read_text())
        self.assertFalse(path.with_suffix('.json.tmp').exists())



    def run_main(self, *, fail=False, reuse=False, bad_manifest=False, outside=False, unpinned=False):
        source = self.root / 'inputs'
        source.mkdir()
        records = []
        for i in range(5, 10):
            name = f'mfx101555026-r0010-s00{i}-c000.xtc2'
            (source / name).write_bytes(self.data)
            records.append(dict(self.expected, relative_path=name, kind='bd'))
        if bad_manifest:
            records[0]['relative_path'] = '../escape'
        manifest = self.root / 'manifest.json'
        manifest.write_text(json.dumps({'files': records}))
        campaign = self.root / 'campaign'
        campaign.mkdir()
        output = (self.root if outside else campaign) / 'datasets'
        if reuse:
            output.mkdir()
            (output / 'keep').write_text('keep')
        argv = ['stage', '--source', str(source), '--output', str(output), '--manifest', str(manifest)]
        real_digest = stage.digest
        def digest(path, expected):
            if fail and 'count-02' in str(path):
                raise RuntimeError('injected verification failure')
            return real_digest(path, expected)
        try:
            with patch.object(sys, 'argv', argv), patch.object(stage, 'CAMPAIGN', campaign), \
                 patch.object(stage, 'MANIFEST_SHA256', 'bad' if unpinned else hashlib.sha256(manifest.read_bytes()).hexdigest()), \
                 patch.object(stage, 'command', self.command), patch.object(stage, 'digest', digest), \
                 patch.dict(os.environ, SLURM_JOB_ID='fixture'), contextlib.redirect_stdout(io.StringIO()):
                stage.main()
        finally:
            self.main_output = output
        return output

    def test_main_all_six_sets_complete(self):
        output = self.run_main()
        record = json.loads((output / 'complete.json').read_text())
        self.assertEqual(record['state'], 'complete')
        self.assertEqual(len(record['sets']), 6)
        self.assertEqual(len(list(output.glob('count-*/verified.json'))), 6)
        self.assertEqual(sum(len(r['verified']) for r in record['sets']), 30)

    def test_main_failed_verification_no_completion(self):
        with self.assertRaisesRegex(RuntimeError, 'injected verification'):
            self.run_main(fail=True)
        record = json.loads((self.main_output / 'status.json').read_text())
        self.assertEqual(record['state'], 'failed')
        self.assertIn('injected', record['error'])
        self.assertFalse((self.main_output / 'complete.json').exists())
        self.assertTrue((self.main_output / 'count-01/verified.json').exists())
        self.assertTrue((self.main_output / 'count-02').exists())

    def test_main_reused_output_refused(self):
        with self.assertRaisesRegex(RuntimeError, 'must be new'):
            self.run_main(reuse=True)
        self.assertEqual((self.main_output / 'keep').read_text(), 'keep')

    def test_main_bad_manifest_refused(self):
        with self.assertRaisesRegex(RuntimeError, 'wrong source manifest'):
            self.run_main(bad_manifest=True)
        self.assertFalse(self.main_output.exists())

    def test_main_outside_campaign_refused(self):
        with self.assertRaisesRegex(RuntimeError, 'campaign root'):
            self.run_main(outside=True)
        self.assertFalse(self.main_output.exists())

    def test_main_unpinned_manifest_refused(self):
        with self.assertRaisesRegex(RuntimeError, 'accepted transfer manifest'):
            self.run_main(unpinned=True)
        self.assertFalse(self.main_output.exists())

    def test_native_source_layout_fixture(self):
        raw = (Path(__file__).parent.parent / 'evidence.txt').read_text()
        sections = raw.strip().split('\n\n')
        self.assertEqual(len(sections), 5)
        for section, ost in zip(sections, [273, 106, 76, 91, 92]):
            with patch.object(stage, 'command', return_value=section):
                self.assertEqual(stage.layout(self.src, 1)['osts'], [ost])


if __name__ == '__main__':
    unittest.main(verbosity=2)
