"""Verify the source-only scaling relocation and complete original-addition accounting."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main(repository=None, source_git=False):
    root = Path(repository).resolve() if repository else next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
    note = root/'psana/psana/gpu/notes/scaling_migration_20261116'
    ledger = json.loads((note/'relocation.json').read_text())
    entries = ledger['files']
    old_paths = [e['old_path'] for e in entries]
    new_paths = [e['new_path'] for e in entries]
    require(len(old_paths) == len(set(old_paths)) and len(new_paths) == len(set(new_paths)), 'Duplicate relocation')
    frozen = 0
    for e in entries:
        p = root/e['new_path']
        require(not Path(e['new_path']).is_absolute() and p.resolve().is_relative_to(root), 'Unsafe path')
        require(p.is_file() and not p.is_symlink(), 'Missing or symlinked relocation: '+str(p))
        require(digest(p.read_bytes()) == e['destination_sha256'], 'Destination hash: '+e['new_path'])
        require(('100755' if p.stat().st_mode & 0o111 else '100644') == e['git_mode'], 'Executable bit')
        if e['action'] == 'moved-byte-exact':
            require(digest(p.read_bytes()) == e['source_sha256'] and p.stat().st_size == e['bytes'], 'Frozen bytes')
            frozen += 1
        else:
            require(e['action'] == 'integrated', 'Unknown relocation action')
    additions = {e['old_path'] for e in entries if e['category'] == 'original-addition'}
    require(len(additions) == ledger['original_additions'] == 359, 'Original addition accounting')
    # Frozen destination coverage rejects accidental extra files, including bytecode.
    archive = root/'psana/psana/gpu/notes/archives/scaling-20261116'
    expected = {e['new_path'] for e in entries if e['action'] == 'moved-byte-exact'}
    actual = {str(p.relative_to(root)) for p in archive.rglob('*') if p.is_file()}
    require(actual == expected | {str((archive/'README.md').relative_to(root))}, 'Frozen archive coverage')
    checked = 0
    if source_git:
        git = ['git','-C',str(root)]
        original = set(subprocess.check_output(git+['diff','--name-only','--diff-filter=A',ledger['source_addition_base'],ledger['source_commit']],text=True).splitlines())
        require(additions == original, 'Source addition inventory mismatch')
        requests = ''.join(ledger['source_commit']+':'+e['old_path']+'\n' for e in entries).encode()
        raw = subprocess.check_output(git+['cat-file','--batch'],input=requests)
        pos = 0
        for e in entries:
            end = raw.index(b'\n',pos)
            header = raw[pos:end].split()
            require(len(header)==3 and header[1]==b'blob','Source object')
            size = int(header[2]); start = end+1
            data = raw[start:start+size]; pos = start+size+1
            require(size == e['bytes'] and digest(data) == e['source_sha256'],'Source identity')
            checked += 1
        # Selected legacy-root directories contain every source file, not only obvious scripts.
        closure = json.loads((note/'dependency-closure.json').read_text())
        wanted = set(subprocess.check_output(git+['ls-tree','-r','--name-only',ledger['source_commit']]+list(closure['roots']),text=True).splitlines())
        require(wanted == {p for p in old_paths if p.startswith('notes/')}, 'Dependency root coverage')
        all_notes = subprocess.check_output(git+['ls-tree','-r','--name-only',ledger['source_commit'],'notes'],text=True).splitlines()
        known = {'/'.join(p.split('/')[:2]) for p in all_notes}
        refs = set()
        for e in entries:
            if e['action']=='moved-byte-exact' and Path(e['new_path']).suffix in {'.py','.sh','.sbatch','.md','.json'}:
                refs.update(re.findall(r'notes/[A-Za-z0-9_.-]+',(root/e['new_path']).read_text(errors='replace')))
        require(refs & known <= set(closure['roots']), 'Unclosed transitive notes reference')
    result = dict(status='pass',paths=len(entries),frozen_files=frozen,original_additions=len(additions),
                  inherited_dependencies=len(entries)-len(additions),source_git_blobs=checked,
                  limitations=['Literal-reference closure includes full source roots and provenance references; external runtime/data paths are not reconstructed.',
                               'Historical SDF missing-input receipt is preserved; no new cross-site availability claim.'])
    print(json.dumps(result,indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository',type=Path)
    parser.add_argument('--source-git',action='store_true')
    args = parser.parse_args()
    main(args.repository,args.source_git)
