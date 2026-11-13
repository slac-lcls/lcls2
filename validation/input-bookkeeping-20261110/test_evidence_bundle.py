"""CPU-only checks of the portable evidence verifier, not GPU pool invariants."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('pr298_evidence_verifier', HERE / 'verify.py')
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


class EvidenceBundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'bundle'
        shutil.copytree(str(HERE), str(self.root), ignore=shutil.ignore_patterns('__pycache__'))

    def manifest(self):
        return json.loads((self.root / 'manifest.json').read_text())

    def save_manifest(self, manifest):
        (self.root / 'manifest.json').write_text(json.dumps(manifest))

    def test_offline_copy_passes(self):
        result = VERIFIER.verify_bundle(self.root)
        self.assertEqual(result['status'], 'pass')
        self.assertEqual(result['inventory']['files'], 61)

    def test_modified_artifact_fails(self):
        path = self.root / self.manifest()['files'][0]['path']
        path.write_bytes(path.read_bytes().replace(b'old', b'bad', 1))
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            VERIFIER.verify_bundle(self.root)

    def test_missing_artifact_fails(self):
        (self.root / self.manifest()['files'][0]['path']).unlink()
        with self.assertRaises(OSError):
            VERIFIER.verify_bundle(self.root)

    def test_duplicate_manifest_entry_fails(self):
        manifest = self.manifest()
        manifest['files'].append(manifest['files'][0])
        self.save_manifest(manifest)
        with self.assertRaisesRegex(ValueError, 'Duplicate manifest path'):
            VERIFIER.verify_bundle(self.root)

    def test_path_escape_fails(self):
        manifest = self.manifest()
        manifest['files'][0]['path'] = '../outside.json'
        self.save_manifest(manifest)
        with self.assertRaisesRegex(ValueError, 'Unsafe manifest path'):
            VERIFIER.verify_bundle(self.root)

    def test_unlisted_artifact_fails(self):
        (self.root / 'ada-10k/receipts/unlisted.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'inventory mismatch'):
            VERIFIER.verify_bundle(self.root)

    def test_symlink_escape_fails(self):
        path = self.root / self.manifest()['files'][0]['path']
        outside = Path(self.temporary.name) / 'outside.json'
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
        with self.assertRaises(ValueError):
            VERIFIER.verify_bundle(self.root)

    def test_rehashed_bad_median_still_fails(self):
        relative = 'ada-10k/receipts/results.json'
        path = self.root / relative
        result = json.loads(path.read_text())
        result['summary'][0]['median_gib_s'] *= 2
        data = json.dumps(result).encode()
        path.write_bytes(data)
        manifest = self.manifest()
        entry = next(item for item in manifest['files'] if item['path'] == relative)
        entry.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                     git_blob_sha1=hashlib.sha1(
                         b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest())
        self.save_manifest(manifest)
        with self.assertRaisesRegex(ValueError, 'Ada median mismatch'):
            VERIFIER.verify_bundle(self.root)


if __name__ == '__main__':
    unittest.main()
