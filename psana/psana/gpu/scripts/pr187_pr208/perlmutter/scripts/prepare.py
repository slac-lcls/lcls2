"""Compute-only new private stage and CPU-derived 1,000-event reference."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct

ROOT = Path(os.environ['CAMPAIGN_ROOT'])
OLD = Path('/pscratch/sd/m/monarin/psana2-gpu/pr187-pr208-perlmutter-20261026/attempt01')
assert os.environ.get('SLURM_JOB_ID')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(16 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ready = json.loads((OLD / 'inputs-ready.json').read_text())
    assert ready['complete']
    reference = json.loads((OLD / 'reference.json').read_text())
    assert sha(OLD / 'reference.json') == ready['reference_sha256']
    assert sha(OLD / 'pixels.json') == ready['pixels_sha256']
    expected_constants = '8df983bf1a5ab4ece8f744666ded1f9387b5a29d2c6505150b587b76f8c78a03'
    assert sha(OLD / 'constants-normalized.pkl.gz') == expected_constants
    for name in ('constants-normalized.pkl.gz', 'constants-normalization.json',
                 'constants.json', 'pixels.json'):
        shutil.copy2(OLD / name, ROOT / name)
        assert sha(ROOT / name) == sha(OLD / name)
    stage = ROOT / 'inputs'
    stage.mkdir(exist_ok=False)
    (stage / 'smalldata').mkdir()
    assert shutil.disk_usage(stage).free > sum(s['bytes'] for s in ready['streams']) + 20*2**30

    def copy(entry):
        src = Path(ready['stage']) / entry['name']
        dst = stage / entry['name']
        before = src.stat()
        h = hashlib.sha256()
        with src.open('rb', buffering=0) as f, dst.open('xb', buffering=0) as out:
            for block in iter(lambda: f.read(16 << 20), b''):
                assert out.write(block) == len(block)
                h.update(block)
            os.fsync(out.fileno())
        after = src.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        assert before.st_size == dst.stat().st_size == entry['bytes']
        assert (before.st_dev, before.st_ino) != (dst.stat().st_dev, dst.stat().st_ino)
        assert h.hexdigest() == sha(dst) == entry['sha256']
        print('PRIVATE_COPY_VERIFIED', entry['name'], flush=True)
        return entry

    with ThreadPoolExecutor(max_workers=5) as pool:
        copied = list(pool.map(copy, ready['streams']))
    for name, digest in reference['10000']['smd_hashes'].items():
        src = Path(ready['stage']) / 'smalldata' / name
        assert sha(src) == digest
        shutil.copy2(src, stage / 'smalldata' / name)
        assert sha(stage / 'smalldata' / name) == digest

    # Decode every stream independently on the CPU; verify known long and short
    # references before accepting the new 1,000-event timestamp and byte counts.
    from psana import dgram
    decoded = {}
    for name in sorted(reference['10000']['prefixes']):
        path = stage / 'smalldata' / name.replace('.xtc2', '.smd.xtc2')
        fd = os.open(path, os.O_RDONLY)
        rows = []
        try:
            cfg = dgram.Dgram(file_descriptor=fd)
            while len(rows) < 10000:
                event = dgram.Dgram(config=cfg)
                if event.service() != 12:
                    continue
                info = event.smdinfo[0].offsetAlg
                offset, size = int(info.intOffset), int(info.intDgramSize)
                assert size > 0 and offset >= 0
                if rows:
                    assert offset >= rows[-1][1] + rows[-1][2]
                rows.append((int(event.timestamp()), offset, size))
        finally:
            os.close(fd)
        decoded[name] = rows
    for n in (200, 1000, 10000):
        stamps = [r[0] for r in next(iter(decoded.values()))[:n]]
        assert len(stamps) == len(set(stamps)) == n and stamps == sorted(stamps)
        assert all([r[0] for r in rows[:n]] == stamps for rows in decoded.values())
        result = dict(timestamp_sha256=hashlib.sha256(struct.pack(f'<{n}Q', *stamps)).hexdigest(),
                      payload_bytes=sum(r[2] for rows in decoded.values() for r in rows[:n]),
                      prefixes={name: rows[n-1][1]+rows[n-1][2] for name, rows in decoded.items()},
                      stage_bytes=reference['10000']['stage_bytes'],
                      smd_hashes=reference['10000']['smd_hashes'])
        if str(n) in reference:
            for key in ('timestamp_sha256', 'payload_bytes', 'prefixes'):
                assert result[key] == reference[str(n)][key], (n, key)
        reference[str(n)] = result
    (ROOT / 'reference.json').write_text(json.dumps(reference, indent=2)+'\n')
    record = dict(complete=True, job=os.environ['SLURM_JOB_ID'], node=os.uname().nodename,
                  source=ready['stage'], stage=str(stage), streams=copied,
                  experiment='mfx101232725', run=442, events=[200,1000,10000],
                  constants_sha256=expected_constants, pixels_sha256=sha(ROOT/'pixels.json'),
                  reference_sha256=sha(ROOT/'reference.json'),
                  reference_derivation='CPU dgram decoding of five hash-verified SMD streams; '
                  'known 200/10000 timestamps, byte totals and prefixes independently rechecked; '
                  '1000-event reference derived identically. Reused 19 validated CPU calibration pixels.')
    (ROOT / 'inputs-ready.json').write_text(json.dumps(record, indent=2)+'\n')
    print('PREPARATION_COMPLETE', json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
