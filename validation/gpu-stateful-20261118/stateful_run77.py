"""Uncapped run 77 through a stateful GpuTask: the worker's own state must
show every event, in the right step (PR #270 phase 1, step A).

Two execution streams, so two worker instances; each counts the events of
every subbatch it processes into bins[step_scope.index] held in ctx.state,
and reports them from end_run. Summed over both instances the bins must be
[1000, 1000, 1000] and the total must equal what run.events() delivered. That
is the end-to-end statement that setup, per-stream dispatch and finish at
close all happen on the real pipeline, not just the EventPool.
"""
import sys
from functools import partial

from psana import DataSource
from psana.gpu import GpuTask, GpuTaskWorker

EXPECTED_PER_STEP = [1000, 1000, 1000]     # run 77 (job 42149659)
# StepScope.index is 0 BEFORE the first BeginStep and counts BeginSteps from
# there, so run 77's three steps are scopes 1, 2 and 3; scope 0 holds any
# pre-step events, of which run 77 has none. Bins are indexed by scope.
FIRST_STEP = 1
N_BINS = 8
RESULTS = {}


class StepCounter(GpuTaskWorker):
    def __init__(self, results):
        self.results = results
        self.counts = None

    def begin_run(self, ctx, stream):
        import cupy as cp
        self.counts = ctx.state.zeros((N_BINS,), cp.uint64)

    def process(self, batch, stream):
        self.counts[batch.step_scope.index] += batch.size

    def end_run(self, ctx, stream):
        import cupy as cp
        self.results[ctx.stream_index] = (ctx.run_instance, ctx.run_status,
                                          cp.asnumpy(self.counts).tolist())

    def close(self):
        self.counts = None


ds = DataSource(exp='mfx100852324', run=77,
                dir='/sdf/data/lcls/ds/prj/public01/xtc',
                detectors=['jungfrau', 'timing'], gpu_det='jungfrau',
                gpu_fn=GpuTask(factory=partial(StepCounter, RESULTS),
                               inputs=['jungfrau.raw'], state_bytes_per_stream=1 << 20),
                n_gpu_streams=2)
run = next(ds.runs())
n_events = sum(1 for _ in run.events())       # exhausting it closes the manager

print(f'STATEFUL77 events={n_events} instances={len(RESULTS)}', flush=True)
for i in sorted(RESULTS):
    ri, status, counts = RESULTS[i]
    print(f'STATEFUL77 stream={i} status={status} counts={counts[:5]} run_instance={tuple(ri)}',
          flush=True)
bins = [sum(RESULTS[i][2][b] for i in RESULTS) for b in range(N_BINS)]
per_step = bins[FIRST_STEP:FIRST_STEP + 3]
rest = bins[0] + sum(bins[FIRST_STEP + 3:])
print(f'STATEFUL77 per_step={per_step} total={sum(per_step)} pre_step_or_beyond={rest}', flush=True)

ok = (len(RESULTS) == 2
      and all(status == 'complete' for _, status, _ in RESULTS.values())
      and per_step == EXPECTED_PER_STEP and rest == 0
      and sum(per_step) == n_events == 3000)
print('STATEFUL77_PASS' if ok else 'STATEFUL77_FAIL', flush=True)
sys.exit(0 if ok else 1)
