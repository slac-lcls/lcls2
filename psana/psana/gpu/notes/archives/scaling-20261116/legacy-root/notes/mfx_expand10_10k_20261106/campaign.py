"""Generate and independently verify one 10k sample10 fixture; no GPU/DB writes."""
import ast
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
import psana
import fixture as f
import verify

NOTES = Path(__file__).resolve().parent
PROJECT = Path('/sdf/data/lcls/ds/prj/prjdatpsana/scratch')
ROOT = PROJECT / 'synthetic_mfx/sample10_10k'
EVIDENCE = PROJECT / 'synthetic_mfx/sample10_10k_evidence_20261106'


def preflight():
    sources = f.load_sources()
    plan = f.schedule(sources[0]['rows'], 10000)
    streams = []
    for stream, source in enumerate(sources):
        jf = f.jf_schemas(source['bd_config'], stream)
        f.require(jf == f.jf_schemas(source['smd_config'], stream), 'BD/SMD schema mismatch')
        extra = 0
        if jf:
            _, blob = f.metadata(set(f.schemas(source['bd_config'])) | set(source['schemas']), stream)
            extra = 2 * len(blob)
        shrink = len(jf) * (f.PIXELS - f.WORDS) * 2
        sizes = {'bd': 0, 'smd': 0}
        for item in plan:
            row = source['rows'][item['source_index']]
            sizes['bd'] += row['bd_size'] - (shrink if item['service'] == 12 else 0)
            sizes['smd'] += row['smd_size']
        sizes['bd'] += extra // 2
        sizes['smd'] += extra // 2
        streams.append({'stream': stream, 'panels': f.PANELS.get(stream, []), 'bytes': sizes})
    constants = {}
    manifest = json.loads((f.SNAPSHOT/'manifest.json').read_text())
    for item in manifest['calibration']:
        doc = item['source_document']
        if item['detector'] != 'jungfrau' or doc['ctype'] not in ('pedestals', 'pixel_gain', 'pixel_offset'):
            continue
        path = f.SNAPSHOT / item['file']
        shape = ast.literal_eval(doc['data_shape'])
        dtype = np.dtype(doc['data_dtype'])
        f.require(shape == (3, 32, 512, 1024), 'Wrong full-panel calibration shape')
        f.require(path.stat().st_size == int(np.prod(shape)) * dtype.itemsize == item['bytes'], 'Calibration size')
        f.require(f.file_sha(path) == item['sha256'], 'Frozen calibration hash mismatch')
        constants[doc['ctype']] = {'path': str(path), 'shape': list(shape), 'dtype': str(dtype),
                                  'id': doc['_id'], 'sha256': item['sha256']}
    f.require(len(constants) == 3, 'Missing frozen calibration')
    total = sum(sum(s['bytes'].values()) for s in streams)
    stat = os.statvfs(PROJECT)
    free = stat.f_bavail * stat.f_frsize
    f.require(os.access(PROJECT, os.W_OK), 'Project scratch not writable')
    f.require(not ROOT.exists(), 'Refusing to overwrite existing fixture')
    f.require(free >= total + 5 * 2**30, 'Insufficient available space plus 5 GiB headroom')
    return {'status': 'pass', 'root': str(ROOT), 'event_count': 10000, 'source_start_event': 0,
            'source_events': 36037, 'records_per_stream': len(plan), 'streams': streams,
            'expected_file_bytes': total, 'seed_bytes': 10000*32*f.WORDS*2,
            'calibration': constants, 'calibration_manifest_sha256': f.file_sha(f.SNAPSHOT/'manifest.json'),
            'storage': {'resolved_project': str(PROJECT.resolve()), 'available_bytes': free,
                        'quota': 'Filesystem availability only; project-specific quota not exposed by quota command'},
            'closing_transitions': plan[-3:], 'first_timestamp': next(x['timestamp'] for x in plan if x['service']==12),
            'last_event_timestamp': next(x['timestamp'] for x in reversed(plan) if x['service']==12),
            'runtime': {'host': socket.gethostname(), 'python': sys.version, 'executable': sys.executable,
                        'psana': psana.__file__, 'psana_sha256': f.file_sha(psana.__file__),
                        'dgram': f.dgram.__file__, 'dgram_sha256': f.file_sha(f.dgram.__file__)},
            'code_sha256': {str(p): f.file_sha(p) for p in
                           [NOTES/'campaign.py', NOTES/'fixture.py', NOTES/'verify.py', NOTES/'run.sbatch',
                            NOTES/'test_fixture.py', NOTES.parent/'mfx_expand10_stage2_20261105/runtime.sh',
                            f.HELPERS/'xtc_fixture.py']}}


def main():
    f.require(sys.argv[1:] in (['preflight'], ['run']), 'Choose preflight or run')
    f.require(os.environ.get('PS_FAKESTEP_FLAG') == '0' and os.environ.get('PS_PARALLEL') == 'none', 'Use runtime.sh')
    before = preflight()
    if sys.argv[1] == 'preflight':
        print(json.dumps(before, indent=2))
        return
    os.umask(0o007)
    EVIDENCE.parent.mkdir(mode=0o770, exist_ok=True)
    EVIDENCE.mkdir(mode=0o770, exist_ok=False)
    started = time.monotonic()
    status = {'phase': 'preflight', 'root': str(ROOT), 'job': os.environ.get('SLURM_JOB_ID'),
              'host': socket.gethostname(), 'accepted': False}

    def checkpoint(phase):
        status.update(phase=phase, elapsed_seconds=time.monotonic()-started)
        pending = EVIDENCE/'status.pending.json'
        with pending.open('w') as out:
            json.dump(status, out, indent=2)
            out.write('\n'); out.flush(); os.fsync(out.fileno())
        os.replace(pending, EVIDENCE/'status.json')
        print('PHASE', phase, flush=True)

    try:
        f.save_json(EVIDENCE/'preflight.json', before)
        checkpoint('generating')
        generation = f.make(ROOT, 10000)
        f.require(generation['total_file_bytes'] == before['expected_file_bytes'], 'Size projection mismatch')
        for saved, expected in zip(generation['streams'], before['streams']):
            f.require({k: v['bytes'] for k, v in saved['files'].items()} == expected['bytes'], 'Stream size mismatch')
        checkpoint('verifying')
        report = verify.verify(ROOT)
        f.save_json(EVIDENCE/'verification.json', report)
        compact = {k: v for k, v in report.items() if k not in ('samples', 'timestamps', 'timeline')}
        compact.update(records_per_stream=len(report['timeline']), first_timestamp=report['timestamps'][0],
                       last_timestamp=report['timestamps'][-1],
                       verification_sha256=f.file_sha(EVIDENCE/'verification.json'),
                       per_stream_event_checks={str(s): sum(x['stream']==s for x in report['samples']) for s in range(10)},
                       per_panel_checks={str(p): sum(any(y['panel']==p for y in x['panels']) for x in report['samples']) for p in range(32)})
        f.save_json(EVIDENCE/'summary.json', compact)
        status['accepted'] = True
        checkpoint('complete')
    except BaseException as error:
        status['error'] = repr(error)
        checkpoint('failed')
        raise


if __name__ == '__main__':
    main()
