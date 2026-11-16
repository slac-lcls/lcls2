"""Full readback plus CPU reference, independently decoded with psana Dgram."""
import argparse
import json
import os
from pathlib import Path
import struct

import numpy as np
from psana import DataSource, dgram

from fixture import (COUNT, FIELDS, META, PANELS, PIXELS, WORDS, SNAPSHOT, HELPERS,
                     check_stat, children, file_sha, header, load_sources, read_record,
                     require, save_json, schemas, service, sha, smd_offsets, spans)


def expand(seed):
    require(seed.dtype == np.uint16 and seed.shape == (WORDS,), 'Expected 52429 uint16 seed words')
    return np.repeat(seed, 10)[:PIXELS].reshape(512, 1024)


def leaves(raw):
    out = {}
    for p, src, kind, size in children(raw):
        # Shapes and Data are included in their owning ShapesData bytes.
        if kind in (1, 4):
            require((kind, src) not in out, 'Duplicate leaf')
            out[(kind, src)] = raw[p:p+size]
    return out


def check_config(old, new, stream, meta_id):
    before, after = schemas(old), schemas(new)
    expected_ids = set(before) | ({meta_id} if stream in PANELS else set())
    require(set(after) == expected_ids, 'Unexpected Names IDs')
    for k, sc in before.items():
        expected = dict(sc)
        if sc['detname'] == 'jungfrau' and sc['alg'] == 'raw':
            expected.update(alg='sim10', version=65536, fields=[('seed', 1, 1)] + FIELDS[1:])
        require(after[k] == expected, 'Unexpected Configure schema mutation')
    old_leaves, new_leaves = leaves(old), leaves(new)
    expected_leaves = set(old_leaves) | ({(4, meta_id), (1, meta_id)} if stream in PANELS else set())
    require(set(new_leaves) == expected_leaves, 'Unexpected Configure XTC children')
    for key, value in old_leaves.items():
        kind, src = key
        if kind == 4 and before[src]['detname'] == 'jungfrau' and before[src]['alg'] == 'raw':
            # Restore only declared Names changes and compare all bytes, including padding.
            restored = bytearray(new_leaves[key])
            restored[784:1044] = value[784:1044]
            restored[1048+260:1048+516] = value[1048+260:1048+516]
            restored[1048+520:1048+524] = value[1048+520:1048+524]
            require(restored == value, 'Extra Names changes')
        else:
            require(new_leaves[key] == value, 'Source config payload changed')
    cfg = dgram.Dgram(view=new)
    if stream in PANELS:
        m = cfg.sim10meta[stream].config
        require(all(getattr(m, k) == v for k, v in META.items()), 'Wrong expansion metadata')
        require(after[meta_id] == {
            'detname': 'sim10meta', 'dettype': 'sim10meta', 'detid': 'sample10-v1',
            'alg': 'config', 'version': 65536, 'segment': stream,
            'fields': [(k, 2, 0) for k in META], 'arrays': 0}, 'Wrong metadata schema')
    return cfg


def check_event(old, new, old_cfg, new_cfg, stream, source_schemas):
    src = dgram.Dgram(config=old_cfg, view=old)
    dst = dgram.Dgram(config=new_cfg, view=new)
    source = getattr(src, 'jungfrau', {})
    target = getattr(dst, 'jungfrau', {})
    require(list(source) == list(target) == PANELS.get(stream, []), 'Missing/reordered/duplicate panels')
    checks = []
    for panel in source:
        original = source[panel].raw
        synthetic = target[panel].sim10
        raw = original.raw
        require(raw.dtype == np.uint16 and raw.shape == (1, 512, 1024), 'Wrong source pixel layout')
        seed = synthetic.seed
        require(seed.shape == (WORDS,) and seed.dtype == np.uint16, 'Wrong seed layout')
        require(np.array_equal(seed, raw.reshape(-1)[::10]), 'Seed differs from source stride')
        for name, _, _ in FIELDS[1:]:
            require(getattr(original, name) == getattr(synthetic, name), 'Changed scalar ' + name)
        expanded = expand(seed)
        direct = raw.reshape(-1)[(np.arange(PIXELS)//10)*10].reshape(512, 1024)
        require(np.array_equal(expanded, direct), 'Wrong CPU expansion, including tail')
        checks.append({'panel': panel, 'seed_sha256': sha(seed.tobytes()),
                       'expanded_sha256': sha(expanded.tobytes()),
                       'raw_gain_counts': np.bincount(raw.reshape(-1) >> 14, minlength=4).tolist(),
                       'expanded_gain_counts': np.bincount(expanded.reshape(-1) >> 14, minlength=4).tolist()})
    # Compare all non-JF event leaves byte-for-byte. For JF, restore pixel data and shape.
    old_parts, new_parts = leaves(old), leaves(new)
    require(set(old_parts) == set(new_parts), 'Event leaf set changed')
    ids = {k: v['segment'] for k, v in source_schemas.items()
           if v['detname'] == 'jungfrau' and v['alg'] == 'raw'}
    for key, block in old_parts.items():
        if key[0] != 1 or key[1] not in ids:
            require(new_parts[key] == block, 'Non-JF event payload changed')
            continue
        # A source panel has one Shapes child and one Data child. This check is
        # independent of the writer's field-spans parser and expansion indexing.
        new_block = new_parts[key]
        oparts = list(children(old[:12] + block))[1:]
        nparts = list(children(new[:12] + new_block))[1:]
        require([(k, src) for _, src, k, _ in oparts] == [(k, src) for _, src, k, _ in nparts],
                'JF child structure changed')
        require(len(oparts) == 2 and sorted(k for _, _, k, _ in oparts) == [2, 3], 'Unexpected children')
        for (op, _, k, on), (np_, _, _, nn) in zip(oparts, nparts):
            a, b = block[op-12:op-12+on], new_block[np_-12:np_-12+nn]
            require(a[:8] == b[:8], 'JF child src/damage/type changed')
            if k == 2:
                require(struct.unpack_from('<5I', b, 12) == (WORDS, 0, 0, 0, 0), 'Wrong seed shape words')
            else:
                require(a[12+PIXELS*2:] == b[12+WORDS*2:], 'JF scalar bytes changed')
        require(block[:8] == new_block[:8], 'JF ShapesData src/damage/type changed')
    return checks


def check_provenance(generation):
    required = [Path(__file__).with_name('fixture.py'), HELPERS/'xtc_fixture.py']
    require(set(generation['code_sha256']) == {str(p) for p in required}, 'Wrong writer dependency set')
    for path in required:
        require(generation['code_sha256'][str(path)] == file_sha(path), 'Writer/helper digest mismatch')
    require(generation['calibration_manifest_sha256'] == file_sha(SNAPSHOT/'manifest.json'),
            'Frozen calibration manifest digest mismatch')


def verify(root):
    generation = json.loads((root/'generation.json').read_text())
    check_provenance(generation)
    count, start = generation['event_count'], generation['source_start_event']
    require(type(count) is int and 1 <= count <= 10000 and type(start) is int and start >= 0, 'Invalid window')
    sources = load_sources()
    require(len(generation['streams']) == 10 and generation['event_count'] == count, 'Wrong stream/event count')
    expected_times = [r['timestamp'] for r in sources[0]['rows'] if r['service'] == 12][start:start+count]
    all_timeline = None
    samples, file_bytes, steps = [], 0, None
    witnesses = {}
    for stream, (source, saved) in enumerate(zip(sources, generation['streams'])):
        require(saved['stream'] == stream and saved['source'] == source['entry'], 'Wrong source identity')
        for kind, item in saved['files'].items():
            path = Path(item['path'])
            require(path.parent == (root if kind == 'bd' else root/'smalldata'), 'Escaped output directory')
            require(path.stat().st_size == item['bytes'] and file_sha(path) == item['sha256'], 'File hash/size mismatch')
            file_bytes += item['bytes']
        old_cfg = dgram.Dgram(view=source['bd_config'])
        source_schemas = schemas(source['bd_config'])
        configs = {}
        times, timeline, counts, state = [], [], [], 'start'
        cursors = {'bd': 0, 'smd': 0}
        with Path(saved['files']['bd']['path']).open('rb') as bd, Path(saved['files']['smd']['path']).open('rb') as smd, source['path'].open('rb') as original:
            for trace in saved['records']:
                row = source['rows'][trace['source_index']]
                old = {'bd': read_record(original, row['bd_offset']),
                       'smd': source['smd'][row['smd_offset']:row['smd_offset']+row['smd_size']]}
                new = {'bd': read_record(bd, cursors['bd']), 'smd': read_record(smd, cursors['smd'])}
                svc, ts = service(new['bd']), header(new['bd'])[0]
                require((service(new['smd']), header(new['smd'])[0]) == (svc, ts), 'BD/SMD identity mismatch')
                timeline.append((ts, svc))
                for kind in ('bd', 'smd'):
                    require(trace['offsets'][kind] == cursors[kind] and trace['sizes'][kind] == len(new[kind]), 'Bad trace')
                    require(trace['source_sha256'][kind] == sha(old[kind]), 'Changed source bytes')
                    require(header(new[kind])[1:5] == header(old[kind])[1:5], 'env/src/damage/type changed')
                    list(children(new[kind]))
                    if svc == 2:
                        configs[kind] = check_config(old[kind], new[kind], stream, saved['metadata_names_id'])
                    else:
                        dgram.Dgram(config=configs[kind], view=new[kind])
                    if svc not in (2, 12):
                        require(new[kind][8:] == old[kind][8:], 'Transition/SlowUpdate payload changed')
                        if svc not in (9, 7, 5):
                            require(new[kind] == old[kind], 'Opening/SlowUpdate timestamp changed')
                    if svc == 12:
                        require(header(old[kind])[0] == ts, 'Original event timestamp changed')
                        if kind == 'smd':
                            b = bytearray(new[kind])
                            actual_refs = smd_offsets(b, source['schemas'])
                            source_refs = smd_offsets(old[kind], source['schemas'])
                            for (p, val), (_, original_val), expected in zip(
                                    actual_refs, source_refs, (cursors['bd'], len(new['bd']))):
                                require(val == expected, 'Wrong SMD reference')
                                struct.pack_into('<Q', b, p, original_val)
                            require(b == old[kind], 'Extra SMD change')
                if svc == 12:
                    samples.append({'stream': stream, 'timestamp': ts,
                                    'panels': check_event(old['bd'], new['bd'], old_cfg, configs['bd'], stream, source_schemas)})
                    for panel in samples[-1]['panels']:
                        key = (ts, panel['panel'])
                        require(key not in witnesses, 'Duplicate event/panel witness')
                        witnesses[key] = panel['seed_sha256']
                    if len(times) % 1000 == 0:
                        print('verified source stream', stream, 'event', len(times), flush=True)
                    require(state == 'enabled', 'Event outside enabled state')
                    counts[-1] += 1
                    times.append(ts)
                elif svc == 10:
                    require(state == 'enabled', 'SlowUpdate outside enabled state')
                else:
                    states = {2: ('start', 'configured'), 4: ('configured', 'run'), 6: ('run', 'step'),
                              8: ('step', 'enabled'), 9: ('enabled', 'step'), 7: ('step', 'run'), 5: ('run', 'done')}
                    require(svc in states and state == states[svc][0], 'Invalid transition sequence')
                    state = states[svc][1]
                    if svc == 6:
                        counts.append(0)
                for kind in cursors:
                    cursors[kind] += len(new[kind])
            require(bd.read(1) == smd.read(1) == b'', 'Unaccounted file tail')
        require(state == 'done' and times == expected_times and counts == [count], 'Incorrect run/step/event count')
        require(all(a[0] < b[0] for a, b in zip(timeline, timeline[1:])), 'Timeline not strictly ordered')
        if all_timeline is None:
            all_timeline = timeline
        require(timeline == all_timeline, 'Unaligned streams')
        check_stat(source['path'], source['entry']['xtc'])
        steps = counts
    require(file_bytes == generation['total_file_bytes'], 'Wrong file byte total')
    require(sum(len(s['panels']) for s in samples) == count*32, 'Not all event panels validated')
    # Run the normal serial SMD/EventBuilder/BigData path, with DB loading disabled.
    # sim10 has no detector adapter yet; access decoded Dgrams only in this fixture check.
    ds = DataSource(exp='mfx101555026', run=10, dir=str(root), live=False, skip_calib_load='all')
    public_times, public_counts, run_count, dg_checks = [], [], 0, 0
    for run in ds.runs():
        run_count += 1
        require(run.expt == 'mfx101555026' and run.runnum == 10, 'Wrong scratch source identity')
        for step in run.steps():
            step_count = 0
            for evt in step.events():
                public_times.append(int(evt.timestamp))
                seen = set()
                for dg in evt._dgrams:
                    require(dg is not None, 'Missing stream in DataSource event')
                    for panel, container in getattr(dg, 'jungfrau', {}).items():
                        require(panel not in seen, 'Duplicate DataSource panel')
                        seen.add(panel)
                        seed = container.sim10.seed
                        require(seed.shape == (WORDS,) and seed.dtype == np.uint16, 'DataSource seed layout')
                        require(sha(seed.tobytes()) == witnesses[(int(evt.timestamp), panel)],
                                'DataSource seed content mismatch')
                        dg_checks += 1
                require(seen == set(range(32)), 'DataSource missing physical panels')
                step_count += 1
                if step_count % 1000 == 0:
                    print('DataSource verified events', step_count, flush=True)
            public_counts.append(step_count)
    require(run_count == 1 and public_counts == steps and public_times == expected_times and dg_checks == count*32,
            'DataSource traversal mismatch')
    return {'status': 'pass', 'scope': '10k fixture CPU/format acceptance; GPU acceptance reported separately',
            'root': str(root), 'event_count': count, 'step_counts': steps, 'stream_count': 10,
            'jf_stream_count': 5, 'panel_events_checked': dg_checks, 'total_file_bytes': file_bytes,
            'seed_bytes': count*32*WORDS*2, 'expanded_bytes': count*32*PIXELS*2,
            'timestamps': expected_times, 'timeline': all_timeline, 'samples': samples,
            'generation_sha256': file_sha(root/'generation.json'),
            'code_sha256': {str(p): file_sha(p) for p in [Path(__file__), Path(__file__).with_name('fixture.py'), HELPERS/'xtc_fixture.py']},
            'PS_FAKESTEP_FLAG': os.environ.get('PS_FAKESTEP_FLAG'), 'PS_PARALLEL': os.environ.get('PS_PARALLEL')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get('PS_FAKESTEP_FLAG') == '0' and os.environ.get('PS_PARALLEL') == 'none', 'Use runtime.sh')
    save_json(args.report, verify(args.root))
    print('Fixture verification PASS', flush=True)
