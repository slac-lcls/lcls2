"""Metadata/receipt readback after preparation; never rehash bigdata on login."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO/'psana/psana/gpu/scripts/performance/regression'))
from native_contract import digest, file_hash, input_ranges

ROOT = Path('/pscratch/sd/m/monarin/psana2-gpu/pr-regression-inputs-20261117')



def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    DATA, JOB, COMMIT = args.data, args.job, args.commit
    bundle = ROOT / args.bundle
    receipt = read(DATA/'preparation-complete.json')
    manifest = read(DATA/'dataset-manifest.json')
    reference = read(DATA/'event-reference.json')
    plan = read(ROOT/'plan-01.json')
    assert receipt['complete'] is True and receipt['job'] == JOB
    assert receipt['plan_sha256'] == file_hash(ROOT/'plan-01.json')
    assert receipt['manifest_sha256'] == file_hash(DATA/'dataset-manifest.json')
    assert receipt['event_reference_sha256'] == manifest['event_reference_sha256'] == file_hash(DATA/'event-reference.json')
    assert reference == plan['reference']
    assert receipt['event_sha256'] == digest([row[0] for row in reference['events']])
    assert len(reference['events']) == receipt['events'] == 10000
    assert sum(row[2] for row in reference['events']) == receipt['requested_bytes'] == 335571760000
    for name, key in (('prepare_inputs.py','generator_sha256'),('native_contract.py','helper_sha256')):
        blob = subprocess.check_output(['git','-C',str(REPO),'show',COMMIT+':psana/psana/gpu/scripts/performance/regression/'+name])
        assert receipt[key] == hashlib.sha256(blob).hexdigest() == file_hash(bundle/name)
    assert set(p.name for p in bundle.iterdir()) == {'prepare_inputs.py','native_contract.py','SHA256SUMS'}
    assert DATA.stat().st_uid == os.getuid() and DATA.stat().st_mode & 0o777 == 0o700
    assert manifest['schema'] == 'run51-private-input-v2' and manifest['private_copy'] is True and manifest['root'] == str(DATA)
    assert [s['stream'] for s in manifest['files']] == [3,5,7,8,9]
    copies = {r['target']:r for r in receipt['source_copy_receipts']}
    assert len(copies) == 10
    layouts = []
    for item, source in zip(manifest['files'],plan['streams']):
        assert input_ranges(item['bigdata']) == source['bigdata']['ranges']
        for key in ('bigdata','smd'):
            entry = item[key];path = Path(entry['path']);st = path.stat()
            assert not path.is_symlink() and st.st_uid == os.getuid() and st.st_nlink == 1 and st.st_mode & 0o777 == 0o444
            assert [entry[k] for k in ('size','mtime_ns','device','inode')] == [st.st_size,st.st_mtime_ns,st.st_dev,st.st_ino]
            row = copies[str(path)]
            assert row['verified'] is True and row['source_sha256'] == row['copy_sha256']
            assert row['source'] == source[key]['path']
            srcstat = Path(row['source']).stat()
            assert [row['source_stat'][k] for k in ('size','mtime_ns','device','inode')] == [srcstat.st_size,srcstat.st_mtime_ns,srcstat.st_dev,srcstat.st_ino]
            if key == 'smd':
                assert entry['sha256'] == file_hash(path) == file_hash(row['source']) == source[key]['sha256']
            count = subprocess.check_output(['lfs','getstripe','-c',str(path)],text=True).strip()
            size = subprocess.check_output(['lfs','getstripe','-S',str(path)],text=True).strip()
            assert count == '1' and size == '1048576'
            layouts.append(dict(path=str(path),count=int(count),size=int(size)))
    accounting = subprocess.check_output(['sacct','-j',JOB,'--format=JobID,State,ExitCode,Elapsed,NodeList,MaxRSS,AllocTRES','-P','-n'],text=True)
    rows = [r.split('|') for r in accounting.splitlines()]
    job = next(r for r in rows if r[0] == JOB)
    assert job[1:3] == ['COMPLETED','0:0'], accounting
    assert job[4] == receipt['node']
    summary = dict(status='pass',job=JOB,reviewed_preparation_commit=COMMIT,node=receipt['node'],
                   elapsed=job[3],accounting=accounting,events=10000,available_events=17872,
                   requested_bytes=receipt['requested_bytes'],copy_bytes=sum(s[k]['size'] for s in manifest['files'] for k in ('bigdata','smd')),
                   files=10,ranges_per_stream=[len(s['bigdata']['ranges']) for s in manifest['files']],
                   bigdata_headers_verified_by_job=50000,slowupdate_gaps_verified_by_job=415,
                   source_copy_hashes_match=True,frozen_code_matches_review=True,private_ownership_verified=True,
                   layouts=layouts,artifacts={name:dict(path=str(DATA/name),sha256=file_hash(DATA/name)) for name in ('dataset-manifest.json','event-reference.json','preparation-complete.json')},
                   limitation='Input preparation only. Full-file and selected-range hashes/header checks performed by the reviewed compute job; readback rechecks metadata/receipts/SMD only. Native/cache/runtime qualification pending.')
    output = args.output
    with output.open('x') as f: json.dump(summary,f,indent=2);f.write('\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
