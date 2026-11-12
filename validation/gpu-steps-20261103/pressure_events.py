"""Uncapped run 77 under memory pressure: every event must still arrive.

The delivery-order defects fixed in 2fdae3ae3 lived on the memory-pressure
path -- _retire_issue_and_yield catches GpuMemoryPressureError and calls
_flush_event_pool() MID-PACKET, with later subbatches unsubmitted. Neither
uncapped acceptance run provokes that, so both passed before the fix; this
driver exists to give that path device evidence.

It counts the recovery path explicitly rather than assuming pressure
happened: a run that never raised proves nothing, so it FAILS if the budget
was too generous to trigger any recovery.

PS_PRESSURE_GB sets the per-BD budget in GiB (default 1.5). Too small and
admission raises "cannot fit alone", which is not recoverable and aborts the
run -- that is a different, legitimate error. The useful band is where a
single event fits but concurrency does not.
"""
import os
import sys

from psana import DataSource
from psana.gpu import GpuTask
from psana.gpu import gpu_events as ge
from psana.gpu.gpu_budget import GpuMemoryPressureError


def unused(batch, stream):
    pass


EXPECTED = 3000          # run 77: 3 steps x 1000 (job 42149659)
BUDGET_GB = float(os.environ.get('PS_PRESSURE_GB', '1.5'))
# batch_size matters as much as the budget. _retire_issue_and_yield only runs
# for subbatch index > 0, so a packet must SPLIT before the recovery path can
# be reached at all -- and splitting is driven by batch bytes against
# _compute_subbatch_budget. A large batch with a small budget is what forces
# both the split and the admission failure.
BATCH = int(os.environ.get('PS_PRESSURE_BATCH', '20'))
STREAMS = int(os.environ.get('PS_PRESSURE_STREAMS', '2'))

# Count recoveries by wrapping the method that performs them. Counting the
# exception at its raise site would also catch the unrecoverable
# "cannot fit alone" case, which is not what this measures.
recoveries = [0]
_original = ge.GpuEventManager._retire_issue_and_yield


def counting_retire_issue_and_yield(self, subbatch):
    ready = self.event_pool.begin_retire_next()
    try:
        yield from self._yield_ready(ready)
    finally:
        self.event_pool.finish_retire_next()
    slot = self.event_pool.next_slot_id
    try:
        return self._issue_gpu_read(subbatch, slot)
    except GpuMemoryPressureError:
        recoveries[0] += 1
        yield from self._flush_event_pool()
        self._trim_gpu_caches()
        return self._issue_gpu_read(subbatch, self.event_pool.next_slot_id)


ge.GpuEventManager._retire_issue_and_yield = counting_retire_issue_and_yield

print(f'=== run 77 uncapped, budget={BUDGET_GB} GiB, batch_size={BATCH}, '
      f'n_gpu_streams={STREAMS} ===', flush=True)
opts = dict(exp='mfx100852324', run=77,
            dir='/sdf/data/lcls/ds/prj/public01/xtc',
            detectors=['jungfrau', 'timing'],
            gpu_det='jungfrau',
            gpu_fn=GpuTask(unused, ['jungfrau.raw'], []),
            gpu_memory_budget_gb=BUDGET_GB,
            n_gpu_streams=STREAMS,
            batch_size=BATCH)

ds = DataSource(**opts)
run = next(ds.runs())
n_events = sum(1 for _ in run.events())
print(f'PRESSURE77 events={n_events} expected={EXPECTED} '
      f'recoveries={recoveries[0]}', flush=True)

# And through steps(), for the per-step structure under the same pressure.
ds2 = DataSource(**opts)
run2 = next(ds2.runs())
before = recoveries[0]
per_step = [sum(1 for _ in step.events()) for step in run2.steps()]
print(f'PRESSURE77 per_step={per_step} total={sum(per_step)} '
      f'recoveries={recoveries[0] - before}', flush=True)

ok = (n_events == EXPECTED
      and per_step == [1000] * 3
      and recoveries[0] > 0)
if recoveries[0] == 0:
    print('PRESSURE77_INCONCLUSIVE: no recovery path was taken -- the budget '
          'was too generous, so this run says nothing about the pressure '
          'path. Lower PS_PRESSURE_GB.', flush=True)
print('PRESSURE77_PASS' if ok else 'PRESSURE77_FAIL', flush=True)
sys.exit(0 if ok else 1)
