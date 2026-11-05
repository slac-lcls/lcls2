"""Serial run.events() to EndRun on run 77: every event must arrive.

Run from the repository root:

    python3 validation/gpu-steps-20261103/serial_events.py


The companion to the uncapped MPI steps case. RunSerial.events() returns on
EndRun (run.py), so a transition released ahead of the events it follows
truncates the run silently -- this counts to the known run size instead.

No max_events: the cap is what hid the defect, since a capped run stops
before EndRun is ever reached.
"""
import os
import sys
from psana import DataSource
from psana.gpu import GpuTask


def unused(batch, stream):
    pass


EXPECTED = 3000          # run 77: 3 steps x 1000 (job 42149659)

ds = DataSource(exp='mfx100852324', run=77,
                dir='/sdf/data/lcls/ds/prj/public01/xtc',
                detectors=['jungfrau', 'timing'],
                gpu_det='jungfrau',
                gpu_fn=GpuTask(unused, ['jungfrau.raw'], []),
                batch_size=20)
run = next(ds.runs())

n_events = 0
for _ in run.events():
    n_events += 1

print(f'SERIAL77 events={n_events} expected={EXPECTED}', flush=True)

# And the same stream through steps(), for the per-step structure.
ds2 = DataSource(exp='mfx100852324', run=77,
                 dir='/sdf/data/lcls/ds/prj/public01/xtc',
                 detectors=['jungfrau', 'timing'],
                 gpu_det='jungfrau',
                 gpu_fn=GpuTask(unused, ['jungfrau.raw'], []),
                 batch_size=20)
run2 = next(ds2.runs())
per_step = [sum(1 for _ in step.events()) for step in run2.steps()]
print(f'SERIAL77 per_step={per_step} total={sum(per_step)}', flush=True)

ok = n_events == EXPECTED and sum(per_step) == EXPECTED and per_step == [1000] * 3
print('SERIAL77_PASS' if ok else 'SERIAL77_FAIL', flush=True)
sys.exit(0 if ok else 1)
