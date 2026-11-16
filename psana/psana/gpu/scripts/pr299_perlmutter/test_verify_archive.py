"""Offline checks for the packaging verifier, separate from campaign tests."""

import tempfile
import unittest
from pathlib import Path

import verify_archive


class ArchiveVerificationTests(unittest.TestCase):
    def test_complete_archive(self):
        result = verify_archive.verify(Path(__file__).resolve().parents[5])
        self.assertEqual(result['timed_samples'], 48)
        self.assertEqual(result['pairs'], 24)

    def test_checksum_input_rejects_unsafe_and_duplicate_paths(self):
        digest = 'a' * 64
        invalid = [f'{digest}  ../escape\n', f'{digest}  /absolute\n',
                   f'{digest}  sample\n{digest}  sample\n', 'bad  sample\n']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checksums'
            for content in invalid:
                with self.subTest(content=content):
                    path.write_text(content)
                    with self.assertRaises(ValueError):
                        verify_archive.read_checksums(path)

    def test_checksum_accepts_valid_relative_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checksums'
            path.write_text('a' * 64 + '  nested/sample\n')
            self.assertEqual(verify_archive.read_checksums(path), {'nested/sample': 'a' * 64})

    def test_numeric_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            verify_archive.near(1.0, 2.0, 'changed sample')


if __name__ == '__main__':
    unittest.main()
