"""Bounded 1..1000-event MFX sample10 writer; no database writes or GPU work."""
import argparse
import json
import os
from pathlib import Path
import struct
import sys

import numpy as np
from psana import dgram
from psana.dgramedit import AlgDef, DetectorDef, DgramEdit

HELPERS = Path(__file__).resolve().parents[1] / 'synthetic_mfx_stage3_20261104'
sys.path.insert(0, str(HELPERS))
from xtc_fixture import (SNAPSHOT, XTC, check_stat, children, file_sha, header,
                         insert_times, load_sources, read_record, require,
                         save_json, schemas, service, sha, smd_offsets, spans)

PANELS = {5: [3, 7, 0, 1, 2], 6: [19, 15, 11, 31, 27, 23],
          7: [18, 14, 10, 6, 30, 26, 22], 8: [16, 12, 8, 4, 28, 24, 20],
          9: [17, 13, 9, 5, 29, 25, 21]}
PIXELS, WORDS, COUNT = 512 * 1024, 52429, 10
FIELDS = [('raw', 1, 3), ('frame_cnt', 3, 0), ('timestamp', 3, 0),
          ('hotPixelThresh', 1, 0), ('numHotPixels', 2, 0), ('maxHotPixels', 2, 0)]
META = {'factor': 10, 'rows': 512, 'columns': 1024, 'dtype_code': 1, 'encoding_version': 1}


def schedule(rows, count=COUNT, start=0):
    require(isinstance(count, int) and 1 <= count <= 1000, 'Count must be 1..1000')
    events = [i for i, r in enumerate(rows) if r['service'] == 12]
    require(isinstance(start, int) and 0 <= start <= len(events)-count, 'Invalid source window')
    first, last = events[start], events[start+count-1]
    # Preserve original opening transitions and SlowUpdates, including state
    # updates before a selected tail window. Omit only unselected L1Accepts.
    indices = [i for i in range(last+1) if rows[i]['service'] != 12 or first <= i <= last]
    out = [{'source_index': i, 'timestamp': rows[i]['timestamp'], 'service': rows[i]['service']}
           for i in indices]
    if start+count == len(events):
        out.extend({'source_index': i, 'timestamp': rows[i]['timestamp'], 'service': rows[i]['service']}
                   for i in range(last+1, len(rows)))
    else:
        for svc, ts in zip((9, 7, 5), insert_times(rows[last]['timestamp'], rows[last+1]['timestamp'], 3)):
            template = next(j for j, t in enumerate(rows) if t['service'] == svc)
            out.append({'source_index': template, 'timestamp': ts, 'service': svc})
    require(sum(r['service'] == 12 for r in out) == count, 'Incorrect event count')
    require([r['service'] for r in out if r['service'] not in (10, 12)] == [2, 4, 6, 8, 9, 7, 5],
            'This bounded writer expects one enabled native step')
    require(all(a['timestamp'] < b['timestamp'] for a, b in zip(out, out[1:])), 'Bad timeline')
    return out


def map_xtc(data, leaf):
    """Preserve headers and sibling order; recompute only changed extents."""
    def walk(block):
        src, damage, contains, extent = XTC.unpack_from(block)
        require(extent == len(block) and extent >= 12, 'Bad XTC block')
        if contains & 4095 == 0:
            parts, p = [], 12
            while p < len(block):
                require(p + 12 <= len(block), 'Short child')
                n = XTC.unpack_from(block, p)[-1]
                require(n >= 12 and p + n <= len(block), 'Bad child size')
                parts.append(walk(block[p:p+n]))
                p += n
            out = bytearray(block[:12] + b''.join(parts))
        else:
            out = bytearray(leaf(block, src, contains & 4095))
        struct.pack_into('<I', out, 8, len(out))
        return bytes(out)
    return data[:12] + walk(data[12:])


def metadata(used, stream):
    key = next(k for k in range(255, -1, -1) if k not in used)
    cfg = DgramEdit(transition_id=2, ts=1, bufsize=65536)
    det = cfg.Detector(DetectorDef('sim10meta', 'sim10meta', 'sample10-v1'),
                       AlgDef('config', 1, 0, 0), {k: (np.uint32, 0) for k in META},
                       nodeId=0, namesId=key, segment=stream)
    for k, v in META.items():
        setattr(det.config, k, v)
    cfg.adddata(det.config)
    buf = bytearray(65536)
    cfg.save(buf)
    blob = bytes(buf[:cfg.size])
    decoded = dgram.Dgram(view=blob)
    require(all(getattr(decoded.sim10meta[stream].config, k) == v for k, v in META.items()),
            'Metadata creation failed')
    return key, blob[24:]


def jf_schemas(config, stream):
    sc = {k: v for k, v in schemas(config).items() if v['detname'] == 'jungfrau' and v['alg'] == 'raw'}
    require([v['segment'] for v in sc.values()] == PANELS.get(stream, []), 'Unexpected JF panel order')
    require(all(v['fields'] == FIELDS and v['dettype'] == 'jungfrau' for v in sc.values()),
            'Unexpected JF schema')
    return sc


def config_transform(raw, jf, meta):
    seen = set()
    def leaf(block, src, kind):
        if kind != 4 or src not in jf:
            return block
        seen.add(src)
        b = bytearray(block)
        b[784:1040] = b'sim10'.ljust(256, b'\0')
        struct.pack_into('<I', b, 1040, 1 << 16)
        b[1048+260:1048+516] = b'seed'.ljust(256, b'\0')
        struct.pack_into('<I', b, 1048+520, 1)
        return b
    out = bytearray(map_xtc(raw, leaf))
    require(seen == set(jf), 'Missing JF Names')
    out.extend(meta)
    struct.pack_into('<I', out, 20, len(out) - 12)
    list(children(out))
    return bytes(out)


def event_transform(raw, jf):
    seen = []
    def leaf(block, src, kind):
        if kind != 1 or src not in jf:
            return block
        seen.append(src)
        # Wrap this ShapesData in a dgram header for the checked field parser.
        wrapped = raw[:12] + block
        fields = spans(wrapped, src, jf[src])
        p, n, dtype, rank = fields['raw']
        require((n, dtype, rank) == (PIXELS*2, 1, 3), 'Wrong raw field size')
        parts = list(children(wrapped))[1:]
        shapes = [(q, length) for q, _, k, length in parts if k == 2]
        payloads = [(q, length) for q, _, k, length in parts if k == 3]
        require(len(shapes) == len(payloads) == 1, 'Unexpected ShapesData children')
        require(shapes[0][0] < payloads[0][0], 'Expected Shapes before Data')
        q, length = shapes[0]
        require(length == 32 and struct.unpack_from('<5I', wrapped, q+12)[:3] == (1, 512, 1024),
                'Wrong source panel shape')
        values = np.frombuffer(wrapped, dtype='<u2', count=PIXELS, offset=p)
        seed = values[::10].tobytes()
        # Preserve scalar fields following the raw array; update Data and ShapesData extents.
        b = bytearray(wrapped[:p] + seed + wrapped[p+n:])
        struct.pack_into('<5I', b, q+12, WORDS, 0, 0, 0, 0)
        data_pos, data_len = payloads[0]
        struct.pack_into('<I', b, data_pos+8, data_len - n + len(seed))
        struct.pack_into('<I', b, 20, len(b)-12)
        return b[12:]
    out = map_xtc(raw, leaf)
    require(seen == list(jf), 'Missing/reordered JF event segments')
    list(children(out))
    return out


def make(root, count=COUNT, start=0):
    require(os.environ.get('PS_FAKESTEP_FLAG') == '0', 'Set PS_FAKESTEP_FLAG=0')
    sources = load_sources()
    plan = schedule(sources[0]['rows'], count, start)
    root.mkdir(mode=0o700, exist_ok=False)
    (root/'smalldata').mkdir()
    report = {'format': 'sample10-v1', 'source': {'exp': 'mfx101555026', 'run': 10},
              'identity_policy': 'source identity retained in private scratch; not a published run',
              'event_count': count, 'source_start_event': start, 'schedule': plan, 'metadata': META, 'streams': [],
              'runtime': {'python': sys.version, 'executable': sys.executable, 'dgram': dgram.__file__},
              'code_sha256': {str(p): file_sha(p) for p in [Path(__file__), HELPERS/'xtc_fixture.py']},
              'calibration_manifest_sha256': file_sha(SNAPSHOT/'manifest.json')}
    for stream, source in enumerate(sources):
        jf = jf_schemas(source['bd_config'], stream)
        require(jf == jf_schemas(source['smd_config'], stream), 'BD/SMD JF schema mismatch')
        key, meta = (None, b'')
        if jf:
            key, meta = metadata(set(schemas(source['bd_config'])) | set(source['schemas']), stream)
        stem = f'mfx101555026-r0010-s{stream:03d}-c000'
        paths = {'bd': root/(stem+'.xtc2'), 'smd': root/'smalldata'/(stem+'.smd.xtc2')}
        trace = []
        with source['path'].open('rb') as original, paths['bd'].open('xb') as bd, paths['smd'].open('xb') as smd:
            for item in plan:
                row = source['rows'][item['source_index']]
                old = {'bd': read_record(original, row['bd_offset']),
                       'smd': source['smd'][row['smd_offset']:row['smd_offset']+row['smd_size']]}
                require(len(old['bd']) == row['bd_size'] and header(old['bd'])[:2] == header(old['smd'])[:2],
                        'Source headers/size mismatch')
                new = {}
                positions = {'bd': bd.tell(), 'smd': smd.tell()}
                for kind in ('bd', 'smd'):
                    raw = old[kind]
                    if item['service'] == 2:
                        raw = config_transform(raw, jf, meta)
                    elif item['service'] == 12 and kind == 'bd':
                        raw = event_transform(raw, jf)
                    b = bytearray(raw)
                    struct.pack_into('<Q', b, 0, item['timestamp'])
                    if item['service'] == 12 and kind == 'smd':
                        (p, _), (q, _) = smd_offsets(b, source['schemas'])
                        struct.pack_into('<Q', b, p, positions['bd'])
                        struct.pack_into('<Q', b, q, len(new['bd']))
                    header(b)
                    new[kind] = bytes(b)
                bd.write(new['bd']); smd.write(new['smd'])
                trace.append({'source_index': item['source_index'], 'offsets': positions,
                              'sizes': {k: len(v) for k, v in new.items()},
                              'source_sha256': {k: sha(v) for k, v in old.items()}})
            for f in (bd, smd):
                f.flush(); os.fsync(f.fileno())
        check_stat(source['path'], source['entry']['xtc'])
        report['streams'].append({'stream': stream, 'panels': PANELS.get(stream, []),
                                  'metadata_names_id': key, 'source': source['entry'], 'records': trace,
                                  'files': {k: {'path': str(p), 'bytes': p.stat().st_size, 'sha256': file_sha(p)}
                                            for k, p in paths.items()}})
        print('wrote stream', stream, flush=True)
    report['total_file_bytes'] = sum(f['bytes'] for s in report['streams'] for f in s['files'].values())
    report['seed_bytes'] = count*32*WORDS*2
    save_json(root/'generation.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--events', type=int, default=COUNT)
    parser.add_argument('--start-event', type=int, default=0)
    args = parser.parse_args()
    make(args.root, args.events, args.start_event)
