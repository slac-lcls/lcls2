import hashlib
import json
import os
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/pscratch/sd/p/psdatmgr/psdm/mfx/mfx100848724/xtc')
CONDA = Path('/global/homes/m/monarin/.conda-envs/psana-build')
BASE = '11e4a13b541e2aefbfcb7168ab4d2040e2bcd4ba'
HEAD = 'acfd029a8d27898b5f165e02facd3c3bbc9c4ad2'
HEAD_SOURCE = ROOT / 'source-head'
HEAD_INSTALL = ROOT / 'install-head'
EXPERIMENTS = {51: 'mfx100848724', 77: 'mfx100852324'}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_hashes(directory):
    return {str(path.relative_to(directory)): file_hash(path)
            for path in sorted(Path(directory).rglob('*'))
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc'}


def scan_smd(payload):
    offset = 0
    events = []
    step = -1
    endruns = 0
    while offset < len(payload):
        if offset + 24 > len(payload):
            raise ValueError('truncated SMD header')
        nanoseconds, seconds, environment, source, damage, contains, extent = struct.unpack_from('<IIIIHHI', payload, offset)
        length = 12 + extent
        if extent < 12 or offset + length > len(payload):
            raise ValueError('invalid SMD extent')
        service = (environment >> 24) & 15
        if service > 12:
            raise ValueError('invalid SMD service')
        if service == 6:
            step += 1
        elif service == 5:
            endruns += 1
        elif service == 12:
            if step < 0:
                raise ValueError('event precedes BeginStep')
            events.append(((seconds << 32) | nanoseconds, step))
        offset += length
    if len(set(timestamp for timestamp, step in events)) != len(events):
        raise ValueError('duplicate timestamp in SMD file')
    if events != sorted(events):
        raise ValueError('out-of-order SMD file')
    return events, endruns


def reference(run_number):
    prefix = f'{EXPERIMENTS[run_number]}-r{run_number:04d}'
    files = sorted((DATA / 'smalldata').glob(prefix + '-*.smd.xtc2'))
    if len(files) != 10:
        raise ValueError(f'expected ten SMD files for {prefix}')
    previous = None
    sources = []
    for path in files:
        payload = path.read_bytes()
        events, endruns = scan_smd(payload)
        if previous is not None and previous != events:
            raise ValueError('SMD streams disagree')
        previous = events
        if run_number == 77 and endruns != 1:
            raise ValueError('run77 lacks EndRun')
        bigdata = DATA / path.name.replace('.smd.xtc2', '.xtc2')
        if not os.access(bigdata, os.R_OK):
            raise ValueError(f'unreadable bigdata: {bigdata}')
        sources.append(dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest(),
                            bigdata=str(bigdata), bytes=bigdata.stat().st_size,
                            mtime_ns=bigdata.stat().st_mtime_ns, endruns=endruns))
    expected = {51: 17872, 77: 3000}[run_number]
    if len(previous) != expected:
        raise ValueError(f'run{run_number} event total changed: {len(previous)}')
    selected = previous[:10000] if run_number == 51 else previous
    return dict(events=[list(event) for event in selected],
                available_events=len(previous), files=sources)


def cases():
    result = []
    for bds in (1, 2):
        for mode in ('gpu', 'cpu'):
            for iterator in (('events',) if mode == 'gpu' else ('events', 'steps')):
                result.append(dict(name=f'r51-{mode}-{iterator}-bd{bds}', run=51, bds=bds,
                                   variants=[dict(runtime=runtime, mode=mode, iterator=iterator)
                                             for runtime in ('base', 'head')]))
        result.append(dict(name=f'r77-gpu-events-steps-bd{bds}', run=77, bds=bds,
                           variants=[dict(runtime='head', mode='gpu', iterator=iterator)
                                     for iterator in ('events', 'steps')]))
    return result


def pair_order(round_number):
    return (0, 1) if round_number % 2 == 0 else (1, 0)


def check_rows(rows, expected, iterator):
    expected_timestamps = [item[0] for item in expected]
    actual = sorted(timestamp for row in rows for timestamp in row['timestamps'])
    if actual != expected_timestamps:
        raise ValueError('delivered timestamps do not match independent SMD reference')
    per_step = {}
    for row in rows:
        if row['role'] != 'bd' and row['timestamps']:
            raise ValueError('non-BD delivered events')
        if row['role'] == 'bd' and not row['timestamps']:
            raise ValueError('idle BD')
        if row['timestamps'] != sorted(row['timestamps']):
            raise ValueError('per-rank ordering changed')
        for timestamp, step in row['step_records']:
            per_step[timestamp] = step
    if iterator == 'steps' and per_step != dict(expected):
        raise ValueError('step boundaries differ from independent SMD reference')
    return hashlib.sha256(json.dumps(actual, separators=(',', ':')).encode()).hexdigest()
