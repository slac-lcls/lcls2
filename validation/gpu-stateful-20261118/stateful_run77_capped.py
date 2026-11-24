"""Run 77 cut by max_events, through a stateful GpuTask (PR #344, R1).

Every stateful acceptance run so far went to EndRun with no cap, so the late
cutoff never fired on real data. This is the capped counterpart.

The cap lands INSIDE step 2, which is the case R1 is about. The manager reads
ahead: once the cap is reached it still submits the remaining subbatches of
the packet, and those carry no events. Counting them as progress reported a
fully delivered step as truncated and gave the NEXT step hooks and exported
records for events the user never received.

With PS_CAP=1500 on run 77 (three steps of 1000):

  step 1  delivered whole          -> hooks, record, complete, not truncated
  step 2  the run ended inside it  -> hooks, record, truncated
  step 3  never eligible           -> no hooks, no record at all

PS_CAP overrides the cap; it must land strictly inside a step for the test to
mean anything.
"""
import os
import sys
from functools import partial

from psana import DataSource
from psana.gpu import GpuTask, GpuTaskWorker

CAP = int(os.environ.get('PS_CAP', '1500'))
STREAMS = int(os.environ.get('PS_STREAMS', '2'))
PER_STEP = 1000                            # run 77 (job 42149659)
FIRST_STEP = 1
N_BINS = 8
RESULTS = {}
SINK = []

if CAP % PER_STEP == 0:
    # A cap on a step boundary is not this check's case and would make it
    # report a failure that is not one. The cutoff is found on the NEXT
    # step's first event, after its read-ahead, so that step has no eligible
    # events at all: the frontier stays on the step before it, which is then
    # truncated. The arithmetic below would expect the later step instead.
    sys.exit(f'PS_CAP={CAP} lands on a step boundary; this check needs a cap '
             f'strictly inside a step')


class StepCounter(GpuTaskWorker):
    def __init__(self, results):
        self.results = results
        self.counts = None
        self.hooks = []

    def begin_run(self, ctx, stream):
        import cupy as cp
        self.counts = ctx.state.zeros((N_BINS,), cp.uint64)

    def begin_step(self, ctx, stream):
        self.hooks.append(('begin_step', ctx.step_scope.index))

    def process(self, batch, stream):
        self.counts[batch.step_scope.index] += batch.size

    def end_step(self, ctx, stream):
        k = ctx.step_scope.index
        self.hooks.append(('end_step', k))
        ctx.emit('cube', key='events', count=self.counts[k:k + 1])

    def end_run(self, ctx, stream):
        import cupy as cp
        self.results[ctx.stream_index] = (ctx.run_status,
                                          cp.asnumpy(self.counts).tolist(),
                                          self.hooks)

    def close(self):
        self.counts = None


ds = DataSource(exp='mfx100852324', run=77,
                dir='/sdf/data/lcls/ds/prj/public01/xtc',
                detectors=['jungfrau', 'timing'], gpu_det='jungfrau',
                gpu_fn=GpuTask(factory=partial(StepCounter, RESULTS),
                               inputs=['jungfrau.raw'], state_bytes_per_stream=1 << 20,
                               aggregate_outputs={'cube': {'count': 'sum'}},
                               aggregate_sink=SINK.append),
                n_gpu_streams=STREAMS, max_events=CAP)
run = next(ds.runs())
n_events = sum(1 for _ in run.events())

cut_step = FIRST_STEP + CAP // PER_STEP     # the step the cap lands inside
print(f'CAPPED77 cap={CAP} events={n_events} instances={len(RESULTS)} '
      f'cut_step={cut_step}', flush=True)

scopes_hooked = set()
status_ok = True
for i in sorted(RESULTS):
    status, counts, hooks = RESULTS[i]
    print(f'CAPPED77 stream={i} status={status} counts={counts[:5]}', flush=True)
    print(f'CAPPED77 stream={i} hooks={hooks}', flush=True)
    scopes_hooked |= {k for _, k in hooks}
    status_ok &= status == 'incomplete'     # a capped run never reaches EndRun

steps = sorted((r for r in SINK if r.begin_ts is not None), key=lambda r: r.begin_ts)
print('CAPPED77 step_records=' +
      str([(r.key, r.fields['count'].tolist(), r.complete, r.truncated) for r in steps]),
      flush=True)

bins = [sum(RESULTS[i][1][b] for i in RESULTS) for b in range(N_BINS)]
print(f'CAPPED77 bins={bins[:5]}', flush=True)

# R1, stated three ways: no hooks past the cut step, no record past it, and
# the step the run ended inside is the only truncated one.
beyond_hooks = {k for k in scopes_hooked if k > cut_step}
truncated = [i for i, r in enumerate(steps) if r.truncated]
# `complete`, not `not truncated`: a record the host byte cap delivered
# early is final=False and truncated=False, and would pass the weaker test.
complete_before = all(r.complete for r in steps[:cut_step - FIRST_STEP])

ok = (status_ok
      and len(RESULTS) == STREAMS
      and n_events == CAP
      and not beyond_hooks
      and len(steps) == cut_step - FIRST_STEP + 1
      and complete_before
      and truncated == [cut_step - FIRST_STEP]
      # closed on every stream, not merely exported once
      and all(('end_step', cut_step) in RESULTS[i][2] for i in RESULTS)
      and bins[FIRST_STEP:cut_step] == [PER_STEP] * (cut_step - FIRST_STEP)
      and sum(bins) == CAP)

if beyond_hooks:
    print(f'CAPPED77 FAIL: hooks for ineligible scopes {sorted(beyond_hooks)}', flush=True)
if len(steps) != cut_step - FIRST_STEP + 1:
    print(f'CAPPED77 FAIL: {len(steps)} step records, expected '
          f'{cut_step - FIRST_STEP + 1}', flush=True)
print('CAPPED77_PASS' if ok else 'CAPPED77_FAIL', flush=True)
sys.exit(0 if ok else 1)
