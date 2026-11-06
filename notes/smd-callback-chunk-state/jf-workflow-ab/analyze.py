"""Compare actual application inputs/outputs against direct-file timestamps."""

import json
from pathlib import Path

import numpy as np

root = Path('/tmp/jf-callback-ab/matrix')
steps = json.loads(Path('/tmp/jf-callback-ab/oracle.json').read_text())
location = {ts: (step, event) for step, events in enumerate(steps)
            for event, ts in enumerate(events)}
assert len(location) == sum(map(len, steps)), 'Oracle timestamps must be unique'
commands = json.loads((root / 'commands.json').read_text())
result = []


def compare_arrays(left, right):
    if left.shape != right.shape:
        return dict(shape_match=False, actual_shape=list(left.shape),
                    expected_shape=list(right.shape))
    equal = (left == right) | (np.isnan(left) & np.isnan(right))
    diff = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return dict(shape_match=True, equal=bool(equal.all()),
                unequal_pixels=int((~equal).sum()),
                max_abs_diff=float(np.nanmax(diff)),
                allclose=bool(np.allclose(left, right, rtol=1e-12, atol=1e-12,
                                         equal_nan=True)))


for command in commands:
    case, stage = command['case'], command['stage']
    path = root / case / stage
    reports = [json.loads(p.read_text()) for p in sorted(path.glob('rank*.json'))]
    actual = sorted(ts for report in reports for ts in report['raw_timestamps'])
    if stage == 'stage2':
        expected = sorted(ts for events in steps for ts in events[2:12])
    else:
        expected = steps[int(stage.rsplit('-', 1)[1])][:10]
    item = dict(case=case, stage=stage, exit_code=command['exit_code'],
                timed_out=command['timed_out'], rank_reports=len(reports),
                completed=all(r['completed'] for r in reports) and bool(reports),
                expected_count=len(expected), actual_count=len(actual),
                timestamps_exact=actual == expected,
                callback_calls=sum(r['callback_calls'] for r in reports),
                chunks=sum(len(r['chunks']) for r in reports),
                per_step_counts=[sum(ts in set(events) for ts in actual)
                                 for events in steps],
                first_indices=[location.get(ts) for ts in actual[:15]],
                exceptions=[r['exception'] for r in reports if 'exception' in r])
    arrays = {}
    if stage == 'stage2':
        for gain in range(3):
            files = list(path.glob('*-gain%d.npz' % gain))
            refs = list((root / 'fixed-large/stage2').glob('*-gain%d.npz' % gain))
            if len(files) != 1 or len(refs) != 1:
                arrays[str(gain)] = dict(missing_or_ambiguous=True)
                continue
            with np.load(files[0]) as left, np.load(refs[0]) as right:
                arrays[str(gain)] = {name: compare_arrays(left[name], right[name])
                                     for name in right.files}
    else:
        # Gate basenames encode gain, not the callback's local step index.
        gates = root / case / 'stage1-repo/jungfrau/block_results'
        refs = root / 'fixed-large/stage1-repo/jungfrau/block_results'
        gain = int(stage.rsplit('-', 1)[1])
        for reference in sorted(refs.glob('*-g%d-gate_*.npy' % gain)):
            file = gates / reference.name
            arrays[reference.name] = (compare_arrays(np.load(file), np.load(reference))
                                      if file.exists() else dict(missing=True))
    item['arrays'] = arrays
    result.append(item)
(root / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
for item in result:
    print('{case:15s} {stage:8s} exit={exit_code:3d} inputs={actual_count:4d}/{expected_count:<3d}'
          ' exact={timestamps_exact!s:5s} calls={callback_calls:3d} chunks={chunks:3d}'
          ' steps={per_step_counts}'.format(**item))

# Acceptance checks include failing pre-fix controls; process exit alone is
# insufficient because the repeated-window pedestal job exits successfully.
assert len(result) == 20
for item in result:
    assert not item['timed_out']
    assert item['rank_reports'] == (5 if item['stage'] == 'stage2' else 1)
    missing_step = (item['case'] in ('unfixed-small', 'unfixed-step')
                    and item['stage'] == 'stage1-2')
    repeated_window = item['case'] == 'unfixed-small' and item['stage'] == 'stage2'
    if missing_step:
        assert item['exit_code'] != 0 and item['actual_count'] == 0
        assert not item['timestamps_exact']
        assert all(value.get('missing') for value in item['arrays'].values())
    elif repeated_window:
        assert item['exit_code'] == 0 and item['completed']
        assert item['per_step_counts'] == [100, 100, 100]
        assert not item['timestamps_exact']
        assert all(not value['arr_av1']['equal'] for value in item['arrays'].values())
    else:
        assert item['exit_code'] == 0 and item['completed'] and item['timestamps_exact']
        values = (list(item['arrays'].values()) if item['stage'] != 'stage2' else
                  [value for arrays in item['arrays'].values() for value in arrays.values()])
        assert values and all(value.get('equal') for value in values)
print('All 20 A/B acceptance checks passed, including the expected pre-fix failures.')
