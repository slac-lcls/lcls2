"""Verify the pinned PR-study import without psana, CUDA, data or a scheduler."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit

RECORD = Path('psana/psana/gpu/notes/pr_campaign_import_20261116')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root, source_git=False):
    root = root.resolve()
    ledger = json.loads((root / RECORD / 'imports.json').read_text())
    require(ledger['schema_version'] == 1, 'Import schema')
    files = ledger['files']
    paths = [entry['path'] for entry in files]
    require(len(paths) == 703 and len(set(paths)) == 703, 'Import coverage/duplicates')
    counts = {}
    for entry in files:
        path = root / entry['path']
        require(not Path(entry['path']).is_absolute() and path.resolve().is_relative_to(root), 'Unsafe import path')
        require(path.is_file() and not path.is_symlink(), 'Missing or symlinked import: ' + str(path))
        require(digest(path) == entry['imported_sha256'], 'Changed import: ' + str(path))
        mode = '100755' if path.stat().st_mode & 0o111 else '100644'
        require(mode == entry['git_mode'], 'Changed Git executable bit: ' + str(path))
        counts[entry['policy']] = counts.get(entry['policy'], 0) + 1
        if entry['policy'] == 'byte-preserved':
            require(digest(path) == entry['source_sha256'] and path.stat().st_size == entry['bytes'], 'Changed frozen bytes')
        else:
            require(entry['policy'] == 'reconciled-navigation', 'Unknown import policy')
    require(counts == {'byte-preserved': 698, 'reconciled-navigation': 5}, 'Import policy coverage')

    source_checks = 0
    if source_git:
        for source in ledger['sources']:
            selected = [e for e in files if e['study'] == source['study']]
            actual = subprocess.check_output(['git', '-C', str(root), 'diff', '--name-status',
                                              source['delta_base'], source['commit']], text=True).splitlines()
            expected = {e['source_change'] + '\t' + e['path'] for e in selected}
            require(set(actual) == expected and len(actual) == len(selected), 'Source delta differs from ledger')
            # One cat-file process verifies all objects without hundreds of Git startups.
            request = ''.join(source['commit'] + ':' + e['path'] + '\n' for e in selected).encode()
            raw = subprocess.check_output(['git', '-C', str(root), 'cat-file', '--batch'], input=request)
            pos = 0
            for entry in selected:
                end = raw.index(b'\n', pos)
                header = raw[pos:end].split()
                require(len(header) == 3 and header[1] == b'blob', 'Missing source blob')
                size = int(header[2]); start = end + 1
                payload = raw[start:start + size]; pos = start + size + 1
                require(len(payload) == entry['bytes'] and hashlib.sha256(payload).hexdigest() == entry['source_sha256'], 'Wrong source identity')
                source_checks += 1

    archived = root / 'psana/psana/gpu/scripts/pr187_pr208'
    manifest = json.loads((archived / 'manifest.json').read_text())
    require(len(manifest['files']) == 47, 'PR187 snapshot count')
    for entry in manifest['files']:
        path = archived / entry['path']
        require(path.resolve().is_relative_to(archived) and not path.is_symlink(), 'Unsafe PR187 snapshot')
        require(path.stat().st_size == entry['bytes'] and digest(path) == entry['sha256'], 'PR187 snapshot hash')

    # Check report navigation. Historic absolute scratch paths remain provenance.
    docs = [root / e['path'] for e in files if e['path'].endswith('.md')]
    docs += [root / RECORD / 'README.md']
    links = 0
    for doc in docs:
        for target in re.findall(r'\[[^\]]*\]\(([^)\s]+)\)', doc.read_text()):
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path or parsed.path.startswith('/'):
                continue
            path = (doc.parent / unquote(parsed.path)).resolve()
            require(path.exists(), 'Broken relative link: ' + str(doc.relative_to(root)) + ' -> ' + target)
            links += 1
    return dict(imported_paths=len(files), policies=counts, source_git_blobs=source_checks,
                pr187_snapshots=47, report_links=links,
                limitations=['Integrity/navigation check, not a new performance result or original raw-data replay.',
                             'Historical absolute scratch links and remote URLs are not availability-checked.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path)
    parser.add_argument('--source-git', action='store_true', help='Also verify source Git objects and exact branch-delta coverage')
    args = parser.parse_args()
    root = args.repository or next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
    print(json.dumps(verify(root, args.source_git), indent=2))


if __name__ == '__main__':
    main()
