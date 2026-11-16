"""Bind the 10k runner to the reviewed manifest, never just raise the old cap."""
import json
from pathlib import Path

NOTES = Path(__file__).resolve().parents[1]
ROOT = Path('/sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/sample10_10k')


def check_10k(root, sha, *, full_hash=False):
    receipt = json.loads((NOTES/'mfx_expand10_10k_20261106/files.json').read_text())
    root = Path(root)
    if root.resolve() != Path(receipt['root']).resolve():
        raise ValueError('Use the accepted project-scratch fixture')
    if sha(root/'generation.json') != receipt['generation_sha256']:
        raise ValueError('Generation manifest differs from reviewed Stage 1')
    manifest = json.loads((root/'generation.json').read_text())
    stamps = [r['timestamp'] for r in manifest['schedule'] if r['service'] == 12]
    if (manifest['event_count'] != 10000 or len(stamps) != 10000 or len(set(stamps)) != 10000
            or stamps != sorted(stamps) or len(manifest['streams']) != 10):
        raise ValueError('Wrong event/stream identity')
    expected = {Path(r['path']).resolve():r for r in receipt['files']}
    seen = set()
    for stream in manifest['streams']:
        for record in stream['files'].values():
            path = Path(record['path']).resolve()
            if path in seen or path not in expected:
                raise ValueError('Duplicate or unexpected file')
            seen.add(path)
            if path.stat().st_size != expected[path]['bytes'] or record['sha256'] != expected[path]['sha256']:
                raise ValueError('Fixture size/hash receipt differs')
            if full_hash and sha(path) != expected[path]['sha256']:
                raise ValueError('Fixture bytes changed')
    if seen != set(expected) or len(seen) != 20:
        raise ValueError('Incomplete fixture')
    return manifest, dict(generation_sha256=receipt['generation_sha256'], files=20,
                          validation='fresh_full_file_sha256' if full_hash else 'fresh_manifest_sha256_and_file_sizes; file hashes from Stage 1',
                          cache='uncontrolled; full-file verification warms input cache' if full_hash else 'uncontrolled; no cold-cache claim')
