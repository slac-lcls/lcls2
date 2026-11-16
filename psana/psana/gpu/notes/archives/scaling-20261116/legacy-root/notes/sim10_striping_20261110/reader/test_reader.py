import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import campaign


class ReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_binary = Path(os.environ['READER_TEST_BINARY'])
        cls.binary = Path(os.environ['READER_BINARY'])
        cls.cpus = sorted(os.sched_getaffinity(0))[:10]
        if len(cls.cpus) < 10:
            raise RuntimeError('fixture tests require10 allocated/login CPUs')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.paths = []
        self.expected = []
        self.checksum = 0
        for i, size in enumerate([1048576 + 3, 2 * 1048576 + 33, 3 * 1048576 - 1, 7919, 5 * 1048576 + 17]):
            path = self.root / f'f{i}'
            data = bytes((j + i) % 251 for j in range(size))
            with path.open('wb') as f:
                f.write(data); f.flush(); os.fsync(f.fileno())
            self.paths.append(path)
            self.expected.append({'path': str(path), 'bytes': size, 'identity': campaign.identity(path)})
            self.checksum += sum(data)

    def tearDown(self):
        self.temp.cleanup()

    def execute(self, fault='', workers=10, production=False, cpus=None):
        env = os.environ.copy()
        env['READER_TEST_FAULT'] = fault
        command = [str(self.binary if production else self.test_binary), str(workers), str(campaign.REQUEST),
                   ','.join(map(str, self.cpus[:workers] if cpus is None else cpus)), '1', *map(str, self.paths)]
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
        parsed = json.loads(result.stdout)
        return result, parsed

    def validate(self, data, workers):
        campaign.audit_reader(data, {'threads': workers}, self.expected, self.cpus[:workers], True)
        self.assertEqual(sum(w['test_checksum'] for w in data['workers']), self.checksum)
        # Independent interval union checks do not reproduce the partition formula.
        for f, expected in enumerate(self.expected):
            ranges = sorted((w['begin'], w['end']) for w in data['workers'] if w['file'] == f)
            self.assertEqual(ranges[0][0], 0)
            self.assertEqual(ranges[-1][1], expected['bytes'])
            for left, right in zip(ranges, ranges[1:]):
                self.assertEqual(left[1], right[0])

    def test_parallel_tails_unequal_and_empty_partitions(self):
        process, data = self.execute()
        self.assertEqual(process.returncode, 0, data)
        self.validate(data, 10)

    def test_one_worker_per_file(self):
        process, data = self.execute(workers=5)
        self.assertEqual(process.returncode, 0, data)
        self.validate(data, 5)

    def test_short_reads_retried(self):
        process, data = self.execute('short')
        self.assertEqual(process.returncode, 0, data)
        self.validate(data, 10)

    def test_interrupted_read_retried(self):
        process, data = self.execute('eintr')
        self.assertEqual(process.returncode, 0, data)
        self.validate(data, 10)

    def test_worker_setup_failure_exits_without_hang(self):
        process, data = self.execute('setup')
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(data['state'], 'failed')
        self.assertNotIn('gib_per_second', data)

    def test_eof_invalidates_trial(self):
        process, data = self.execute('eof')
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(data['state'], 'failed')
        self.assertNotIn('gib_per_second', data)

    def test_read_error_invalidates_trial(self):
        process, data = self.execute('readerror')
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(data['state'], 'failed')

    def test_duplicate_cpu_refused(self):
        process, data = self.execute(cpus=[self.cpus[0]] * 10)
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(data['state'], 'failed')

    def test_resident_cache_invalidates_before_start(self):
        process, data = self.execute('cache')
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(data['state'], 'failed')
        self.assertIn('cache not empty', data['error'])
        self.assertNotIn('gib_per_second', data)

    def test_production_has_no_fault_injection(self):
        process, data = self.execute('setup', production=True)
        self.assertEqual(process.returncode, 0, data)
        campaign.audit_reader(data, {'threads': 10}, self.expected, self.cpus, True)
        self.assertEqual(sum(w['test_checksum'] for w in data['workers']), 0)

    def test_audit_mutations_rejected(self):
        process, good = self.execute()
        self.assertEqual(process.returncode, 0, good)
        self.validate(good, 10)
        mutations = [lambda r: r.update(bytes=r['bytes']-1),
                     lambda r: r.update(threads=5),
                     lambda r: r.update(request_bytes=4096),
                     lambda r: r.update(seconds=r['seconds']*2),
                     lambda r: r.update(gib_per_second=999),
                     lambda r: r['files'][0].update(cold_resident_pages=1),
                     lambda r: r['files'][0].update(warm_resident_pages=0),
                     lambda r: r['files'][0].update(pages=0),
                     lambda r: r['workers'][0].update(begin=1),
                     lambda r: r['workers'][0].update(bytes=999),
                     lambda r: r['workers'][0].update(actual_end_cpu=999),
                     lambda r: r['workers'].pop()]
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                changed=copy.deepcopy(good); mutate(changed)
                with self.assertRaises(RuntimeError):
                    campaign.audit_reader(changed, {'threads': 10}, self.expected, self.cpus, True)

    def test_randomized_complete_blocks(self):
        cases = campaign.schedule('sweep')
        self.assertEqual(len(cases), 54)
        expected = {(c, t) for c in campaign.COUNTS for t in campaign.THREADS}
        for block in range(3):
            self.assertEqual({(c['count'], c['threads']) for c in cases if c['block'] == block}, expected)
        self.assertEqual(cases, campaign.schedule('sweep'))
        self.assertNotEqual([(c['count'],c['threads']) for c in cases[:18]], [(c['count'],c['threads']) for c in cases[18:36]])

    def test_all_file_eviction_and_receipt(self):
        process = subprocess.run([str(self.binary), '--evict-all', *map(str, self.paths)], capture_output=True, text=True, timeout=30)
        self.assertEqual(process.returncode, 0, process.stdout)
        record = json.loads(process.stdout)
        campaign.audit_global_cache(record, self.expected)
        record['files'][0]['resident_pages'] = 1
        with self.assertRaisesRegex(RuntimeError, 'global cache not empty'):
            campaign.audit_global_cache(record, self.expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)
