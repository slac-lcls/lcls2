"""Acceptance/rejection checks using copies of completed native reports."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from summarize import summarize


class SummaryChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        native = Path(os.environ['BASELINE_NATIVE_ROOT'])
        for repeat in (1, 2, 3):
            for streams in (1, 2):
                name = f'clean-s{streams}-r{repeat}'
                (self.root / name).mkdir()
                shutil.copyfile(native / name / 'report.json', self.root / name / 'report.json')

    def change(self, edit):
        path = self.root / 'clean-s1-r1/report.json'
        report = json.loads(path.read_text())
        edit(report)
        path.write_text(json.dumps(report))

    def test_native_reports_pass(self):
        self.assertEqual(summarize(self.root)['status'], 'pass')

    def test_missing_case_rejected(self):
        (self.root / 'clean-s2-r3/report.json').unlink()
        with self.assertRaises(FileNotFoundError):
            summarize(self.root)

    def test_pressure_rejected(self):
        self.change(lambda r: r['admission_pressure_trims'].update(steady=1))
        with self.assertRaisesRegex(ValueError, 'admission drains'):
            summarize(self.root)

    def test_bank_mismatch_rejected(self):
        self.change(lambda r: r['checks']['per_bank_counts'][0].__setitem__(0, 0))
        with self.assertRaisesRegex(ValueError, 'bank ownership'):
            summarize(self.root)

    def test_wrong_rate_rejected(self):
        self.change(lambda r: r['window'].update(completed_events_per_second=35000.0))
        with self.assertRaisesRegex(ValueError, 'rate arithmetic'):
            summarize(self.root)


if __name__ == '__main__':
    unittest.main()
