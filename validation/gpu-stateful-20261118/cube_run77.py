"""StepCube on run 77 against a CPU reference: per-step pixel sums and
per-segment counts must match exactly (PR #270 phase 1, step D).

GPU pass: two execution streams, StepCube emitting one record per step per
stream, merged by the per-BD reducer. CPU pass: run.steps() on the plain CPU
path, summing each present segment's raw words per step. Missing segments are
handled the same way on both sides -- zero data, zero presence -- which is why
the reference reads the per-segment dict rather than det.raw.raw(evt), which
returns None for any partial event.
"""
import sys
from functools import partial

import numpy as np

from psana import DataSource
from psana.gpu import GpuTask
from psana.gpu.examples.step_cube import StepCube

OPTS = dict(exp='mfx100852324', run=77, dir='/sdf/data/lcls/ds/prj/public01/xtc',
            detectors=['jungfrau', 'timing'])
SINK = []

# ---- GPU --------------------------------------------------------------------
ds = DataSource(gpu_det='jungfrau', n_gpu_streams=2,
                gpu_fn=GpuTask(factory=partial(StepCube, 'jungfrau'), inputs=['jungfrau.raw'],
                               state_bytes_per_stream=StepCube.state_bytes(32),
                               aggregate_outputs=StepCube.OUTPUTS, aggregate_sink=SINK.append),
                **OPTS)
run = next(ds.runs())
n_gpu_events = sum(1 for _ in run.events())
gpu_steps = sorted((r for r in SINK if r.begin_ts is not None), key=lambda r: r.begin_ts)
print(f'CUBE77 gpu events={n_gpu_events} records={len(SINK)} '
      f'steps={[(r.fields["counts"].tolist()[:3], r.complete, r.streams, r.key[:3]) for r in gpu_steps]}',
      flush=True)

# ---- CPU reference ----------------------------------------------------------
ds = DataSource(**OPTS)
run = next(ds.runs())
det = run.Detector('jungfrau')
key = ('jungfrau', 'raw')
ok, n_cpu_events, i_step = True, 0, 0
for step in run.steps():
    rec = gpu_steps[i_step] if i_step < len(gpu_steps) else None
    ids = list(rec.key) if rec is not None else None       # dense row -> segment id
    sums = counts = None
    for evt in step.events():
        n_cpu_events += 1
        # Private attribute, deliberately: det.raw.raw(evt) returns None for a
        # partial event, and this check must mirror the GPU's per-segment
        # presence. If Event's internals change, this reference breaks loudly.
        segs = evt._det_segments.get(key, {})
        if sums is None and rec is not None:
            sums = np.zeros(rec.fields['sums'].shape, np.uint64)
            counts = np.zeros(len(ids), np.uint64)
        for seg_id, seg in segs.items():
            if rec is None:
                continue
            row = ids.index(int(seg_id))
            # A jungfrau segment's raw is (1, 512, 1024); the bank row is (512, 1024).
            sums[row] += seg.raw.reshape(sums.shape[1:]).astype(np.uint64)
            counts[row] += 1
    if rec is None or sums is None:
        print(f'CUBE77 step={i_step + 1} no GPU record / no CPU events', flush=True)
        ok = False
    else:
        same = np.array_equal(sums, rec.fields['sums']) and np.array_equal(counts, rec.fields['counts'])
        print(f'CUBE77 step={i_step + 1} counts cpu={counts.tolist()[:3]} gpu={rec.fields["counts"].tolist()[:3]} '
              f'sums_equal={np.array_equal(sums, rec.fields["sums"])} max_cpu={int(sums.max())}', flush=True)
        ok &= same
    i_step += 1

ok &= (i_step == len(gpu_steps) == 3 and n_cpu_events == n_gpu_events == 3000
       and all(r.complete and r.streams == (0, 1) for r in gpu_steps))
print(f'CUBE77 cpu events={n_cpu_events} steps={i_step}', flush=True)
print('CUBE77_PASS' if ok else 'CUBE77_FAIL', flush=True)
sys.exit(0 if ok else 1)
