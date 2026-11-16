"""Offline checks for the packaging verifier, separate from campaign tests."""

import shutil
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

    def test_archive_rejects_tampering(self):
        source = Path(__file__).resolve().parents[5]
        for mutation in ('modified', 'missing', 'extra', 'symlink', 'mapped_hash'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                repository = Path(directory) / 'repository'
                for relative in (verify_archive.SCRIPTS, verify_archive.EVIDENCE):
                    shutil.copytree(source / relative, repository / relative)
                shutil.copy2(source / verify_archive.REPORT, repository / verify_archive.REPORT)
                target = repository / verify_archive.EVIDENCE / 'campaign-summary.json'
                expected = 'Checksum mismatch'
                if mutation in ('modified', 'mapped_hash'):
                    target.write_bytes(target.read_bytes() + b'\n')
                    if mutation == 'mapped_hash':
                        manifest = repository / verify_archive.EVIDENCE / 'SHA256SUMS'
                        checksums = verify_archive.read_checksums(manifest)
                        checksums[str(target.relative_to(repository))] = verify_archive.sha256(target)
                        manifest.write_text(''.join(f'{digest}  {name}\n' for name, digest in sorted(checksums.items())))
                        expected = 'Mapped hash mismatch'
                elif mutation == 'missing':
                    target.unlink()
                    expected = 'Checksum coverage differs'
                elif mutation == 'extra':
                    (target.parent / 'unexpected.txt').write_text('unexpected\n')
                    expected = 'Checksum coverage differs'
                else:
                    outside = Path(directory) / 'outside.json'
                    shutil.copy2(target, outside)
                    target.unlink()
                    target.symlink_to(outside)
                    expected = 'Unsafe package path'
                with self.assertRaisesRegex(ValueError, expected):
                    verify_archive.verify(repository)


if __name__ == '__main__':
    unittest.main()
