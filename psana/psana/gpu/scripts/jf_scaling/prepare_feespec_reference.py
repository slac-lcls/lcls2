"""Build independent six-stream SMD and CPU feespec references before freezing."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np
from psana import dgram
from common import digest, sha
from feespec import coalesced_requests


def smd_rows(directory, name, events):
    fd = os.open(directory/'smalldata'/name.replace('.xtc2', '.smd.xtc2'), os.O_RDONLY)
    rows = []
    fence = 0
    try:
        cfg = dgram.Dgram(file_descriptor=fd)
        while len(rows) < events:
            evt = dgram.Dgram(config=cfg)
            if evt.service() != 12:
                fence += 1
                continue
            info = evt.smdinfo[0].offsetAlg
            rows.append(dict(timestamp=int(evt.timestamp()), offset=int(info.intOffset),
                             size=int(info.intDgramSize), fence=fence))
    finally:
        os.close(fd)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--jf-reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    previous = json.loads(a.jf_reference.read_text())
    stage_manifest = json.loads((a.directory/'manifest.json').read_text())
    streams = {r['name']: smd_rows(a.directory, r['name'], 10000)
               for r in stage_manifest['streams']}
    name = 'mfx101210926-r0387-s000-c000.xtc2'
    stamps, sums, arrays = [], [], []
    fd = os.open(a.directory/name, os.O_RDONLY)
    try:
        cfg = dgram.Dgram(file_descriptor=fd)
        while len(stamps) < 10000:
            evt = dgram.Dgram(config=cfg)
            if evt.service() != 12:
                continue
            values = evt.feespec[0].raw.hproj
            assert values.shape == (2048,) and values.dtype == np.int32
            stamps.append(int(evt.timestamp()))
            sums.append(int(values.sum(dtype=np.int64)))
            if len(stamps) <= 200:
                arrays.append(dict(timestamp=stamps[-1], digest=digest(values)))
    finally:
        os.close(fd)
    refs = {}
    for n in (200, 10000):
        ts = stamps[:n]
        assert all([r['timestamp'] for r in rows[:n]] == ts for rows in streams.values())
        timestamp_hash = hashlib.sha256(struct.pack(f'<{n}Q', *ts)).hexdigest()
        assert timestamp_hash == previous[str(n)]['timestamp_sha256']
        for key, rows in streams.items():
            assert all(r['size'] > 0 and r['offset'] >= 0 for r in rows[:n])
            if key != name:
                assert all(r['size'] > 1 << 20 for r in rows[:n])
        jf_bytes = sum(r['size'] for key, rows in streams.items() if key != name for r in rows[:n])
        assert jf_bytes == previous[str(n)]['payload_bytes']
        payload = sum(r['size'] for rows in streams.values() for r in rows[:n])
        if n == 10000:
            assert payload == stage_manifest['payload_bytes']
        requests = dict(off=n*6, on=sum(coalesced_requests(rows[:n]) for rows in streams.values()))
        refs[str(n)] = dict(timestamp_sha256=timestamp_hash, payload_bytes=payload,
            prefixes={key: rows[n-1]['offset']+rows[n-1]['size'] for key, rows in streams.items()},
            stage_bytes={r['name']: r['stage_bytes'] for r in stage_manifest['streams']},
            smd_hashes={key.replace('.xtc2', '.smd.xtc2'): sha(a.directory/'smalldata'/key.replace('.xtc2', '.smd.xtc2'))
                        for key in streams}, requests=requests,
            feespec=dict(timestamps=ts, sums=sums[:n], arrays=arrays))
        print(n, 'events', payload, 'bytes', requests, flush=True)
    a.output.write_text(json.dumps(refs, indent=2)+'\n')


if __name__ == '__main__':
    main()
