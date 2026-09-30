"""Count native CUDA activity in accepted reports and each BD's steady range.

Steady counts select activity starting inside the NVTX time window, including
worker-thread APIs in that process. They do not attribute activity to an event.
API durations are inclusive and may overlap; they are not GPU execution times.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3


MARKER = 'psana.benchmark.steady'


def read_profile(path):
    with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
        tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
        if 'CUPTI_ACTIVITY_KIND_KERNEL' not in tables:
            raise ValueError(f'{path}: no CUDA kernel records')
        strings = dict(db.execute('select id, value from StringIds'))
        windows = [(start, end) for start, end, text, text_id in db.execute(
            'select start, end, text, textId from NVTX_EVENTS')
            if (text or strings.get(text_id)) == MARKER and end is not None]
        if len(windows) != 1:
            raise ValueError(f'{path}: expected one completed {MARKER} range, got {windows}')
        start, end = windows[0]
        if end <= start:
            raise ValueError(f'{path}: invalid steady range')
        result = dict(path=str(path.resolve()), steady_window_ns=[start, end], scopes={})
        for scope, where, params in (('full', '', ()), ('steady', ' where start >= ? and start < ?', (start, end))):
            kernels = {}
            for name, count, duration in db.execute(
                    'select demangledName, count(*), sum(end-start) from CUPTI_ACTIVITY_KIND_KERNEL'
                    + where + ' group by demangledName', params):
                kernels[strings[name]] = dict(count=count, duration_ns=duration)
            if not kernels:
                raise ValueError(f'{path}: no {scope} kernels')
            apis = {}
            for table in ('CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_DRIVER'):
                if table not in tables:
                    continue
                for name, count, duration in db.execute(
                        'select nameId, count(*), sum(end-start) from ' + table
                        + where + ' group by nameId', params):
                    apis[table.rsplit('_', 1)[1].lower() + '/' + strings[name]] = dict(
                        count=count, duration_ns=duration)
            copies = {}
            if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables:
                labels = dict(db.execute('select id, label from ENUM_CUDA_MEMCPY_OPER'))
                for kind, count, size, duration in db.execute(
                        'select copyKind, count(*), sum(bytes), sum(end-start) '
                        'from CUPTI_ACTIVITY_KIND_MEMCPY' + where + ' group by copyKind', params):
                    copies[labels[kind]] = dict(count=count, bytes=size, duration_ns=duration)
            result['scopes'][scope] = dict(kernels=kernels, apis=apis, copies=copies)
        return result


def summarize(directory):
    complete = json.loads((directory/'complete.json').read_text())
    rows = json.loads((directory/'results.json').read_text())
    if complete != dict(complete=True, samples=8) or len(rows) != 8:
        raise ValueError('Expected all eight accepted profile cases')
    expected = {f'{variant}-{workload}-bd{bds}' for bds in (1, 4)
                for variant, workload in (('parent', 'calib'), ('stage1', 'calib'),
                                          ('stage1', 'input'), ('stage1b', 'input'))}
    if {row['tag'] for row in rows} != expected:
        raise ValueError('Unexpected profile case matrix')
    cases = []
    for row in rows:
        reports = row['reports']
        bds = row['result']['nbds']
        if len(reports) != bds or len(set(reports)) != bds:
            raise ValueError(f"{row['tag']}: missing or duplicate rank reports")
        ranks = [read_profile(Path(report).with_suffix('.sqlite')) for report in reports]
        scopes = {}
        for scope in ('full', 'steady'):
            categories = {}
            for category in ('kernels', 'apis', 'copies'):
                totals = defaultdict(lambda: defaultdict(int))
                for rank in ranks:
                    for name, values in rank['scopes'][scope][category].items():
                        for key, value in values.items():
                            totals[name][key] += value
                categories[category] = {k: dict(v) for k, v in sorted(totals.items())}
            scopes[scope] = categories
        cases.append(dict(tag=row['tag'], ranks=ranks, aggregate=scopes))
    return dict(provenance=json.loads((directory/'provenance.json').read_text()),
                selection='Activity start within each BD NVTX window; APIs include worker threads. '
                          'Durations may overlap and must not be summed as wall time.', cases=cases)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.directory)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    for case in result['cases']:
        print(case['tag'], json.dumps(case['aggregate']['steady']['kernels'], sort_keys=True))
