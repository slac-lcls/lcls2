"""Deliberate-corruption checks for the archived application record."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from paths import find_repository
from sim10_cube import validate

REPO = find_repository()
RECORD = 'psana/psana/gpu/docs/performance/campaigns/perlmutter/sim10-xpp-cube/2026-11-16-se-ec-ecc-100k'


class CompletedCubeChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='sim10-archive-test-')
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        self.record = json.loads((REPO/RECORD/'manifest.json').read_text())
        self.archive = self.repo/self.record['archive_root']
        shutil.copytree(REPO/self.record['archive_root'], self.archive)
        shutil.copytree(REPO/RECORD, self.repo/RECORD)
        self.report = self.repo/RECORD/'README.md'

    def check(self):
        return validate(self.record, self.report, self.repo)

    def change(self, path, mutation):
        data = json.loads(path.read_text())
        mutation(data)
        path.write_text(json.dumps(data))

    def test_original_passes(self):
        self.assertEqual(self.check(), 35)

    def test_dropped_or_duplicated_case(self):
        path = self.archive/'case-receipts.json'
        original = json.loads(path.read_text())
        for variant in [original[:-1], original[:-1]+[copy.deepcopy(original[0])]]:
            with self.subTest(length=len(variant)):
                path.write_text(json.dumps(variant))
                with self.assertRaisesRegex(ValueError, 'population|identity'):
                    self.check()

    def test_changed_timing(self):
        self.change(self.archive/'case-receipts.json', lambda d: d[3].update(seconds=d[3]['seconds']*2))
        with self.assertRaisesRegex(ValueError, 'Original timing'):
            self.check()

    def test_changed_batch(self):
        self.change(self.archive/'case-receipts.json', lambda d: d[3]['case'].update(batch=200))
        with self.assertRaisesRegex(ValueError, 'identity'):
            self.check()

    def test_warm_final_scan(self):
        self.change(self.archive/'case-receipts.json', lambda d: d[3]['cache_gate']['attempts'][-1]['scans'][-1]['files'][0].update(resident_pages=1))
        with self.assertRaisesRegex(ValueError, 'Client-cold'):
            self.check()

    def test_cube_nan_flag(self):
        self.change(self.archive/'case-receipts.json', lambda d: d[0]['ranks'][2]['cube_verification']['bins'][0].update(finite=False))
        with self.assertRaisesRegex(ValueError, 'Cube content'):
            self.check()

    def test_wrong_cube_bin_counts_preserving_total(self):
        def change(data):
            counts = data[0]['ranks'][2]['cube_verification']['counts']
            counts[0] += 1
            counts[1] -= 1
        self.change(self.archive/'case-receipts.json', change)
        with self.assertRaisesRegex(ValueError, 'Global ordinal'):
            self.check()

    def test_csv_wrong_byte_denominator(self):
        path = self.repo/RECORD/'runs.csv'
        path.write_text(path.read_text().replace('335820000000', '671640000000', 1))
        with self.assertRaisesRegex(ValueError, 'CSV requested_bytes'):
            self.check()

    def test_authored_table_drift(self):
        self.report.write_text(self.report.read_text().replace('5.181', '9.842', 1))
        with self.assertRaisesRegex(ValueError, 'Report table'):
            self.check()

    def test_bundle_modification(self):
        path = self.archive/'bundle/run.sbatch'
        path.write_text(path.read_text()+'\n# changed\n')
        with self.assertRaisesRegex(ValueError, 'Frozen bundle'):
            self.check()

    def test_memory_summary_corruption(self):
        self.change(self.archive/'memory-summary.json',
                    lambda d: d.update(max_task_d2h_destination_capacity_bytes=0))
        with self.assertRaisesRegex(ValueError, 'Memory summary'):
            self.check()

    def test_wrong_configuration(self):
        self.record['configuration']['workers_per_bd'] = 8
        with self.assertRaisesRegex(ValueError, 'Worker/storage'):
            self.check()


if __name__ == '__main__':
    unittest.main()
