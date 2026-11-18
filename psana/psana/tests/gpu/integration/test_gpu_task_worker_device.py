"""A stateful task on a real device (PR #270 phase 1, step A).

Two execution streams, one GpuTaskWorker each. Each worker allocates its bins
through ctx.state inside the setup reservation, accumulates the pixel sum and
event count of every subbatch it is handed into bins[step_scope.index] on its
own stream, and reads them back in end_run. Two submissions in different
scopes land on different slots, so each instance sees exactly one, and the
second instance's bins must show scope 2 only -- the same statement
test_gpu_step_scope_device.py makes for a closure variable, now for
budget-visible, run-owned state. Every hook asserts it runs with its stream
current, which is what makes the in-hook CuPy work stream-ordered.

Marked `gpu`; skipped without a CUDA device.
"""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from test_batched_gather import _gpu_available
from test_gpu_producer_device import bindings, envelopes, fixture

from psana.gpu import GpuTask, GpuTaskWorker, RunInstance
from psana.gpu.gpu_plan import StepScope
from psana.gpu.gpu_stream import EventPool
from psana.gpu.gpu_task_worker import TaskWorkers

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not _gpu_available(), reason='no CUDA device')]

N_BINS = 4


class _Acc(GpuTaskWorker):
    def __init__(self, results):
        self.results = results
        self.sums = self.counts = None
        self.hooks = []

    def begin_step(self, ctx, stream):
        self._on_stream(stream)
        self.hooks.append(('begin_step', ctx.step_scope.index))

    def end_step(self, ctx, stream):
        self._on_stream(stream)
        k = ctx.step_scope.index
        self.hooks.append(('end_step', k))
        ctx.emit('cube', key='events', count=self.counts[k:k + 1])

    def _on_stream(self, stream):
        import cupy as cp
        assert cp.cuda.get_current_stream().ptr == stream.ptr, 'hook not on its stream'

    def begin_run(self, ctx, stream):
        import cupy as cp
        self._on_stream(stream)
        assert ctx.input_spec('camera.raw').dtype == np.dtype(np.uint16)
        self.sums = ctx.state.zeros((N_BINS,), cp.uint64)
        self.counts = ctx.state.zeros((N_BINS,), cp.uint64)

    def process(self, batch, stream):
        import cupy as cp
        self._on_stream(stream)
        i = batch.step_scope.index
        self.sums[i] += batch.input('camera.raw').astype(cp.uint64).sum()
        self.counts[i] += batch.size
        assert batch.run_instance == RunInstance('test', 77, 0, 0)

    def end_run(self, ctx, stream):
        import cupy as cp
        self._on_stream(stream)
        self.results[ctx.stream_index] = (ctx.run_status, cp.asnumpy(self.sums),
                                          cp.asnumpy(self.counts), self.hooks)

    def close(self):
        self.sums = self.counts = None


def test_two_stream_worker_accumulates_run_owned_state_by_scope():
    import cupy as cp
    parser, detector, budget, specs, expected, window = fixture(cp, ((0,),) * 3)
    mapping = bindings(parser, detector)
    results, sink = {}, []
    task = GpuTask(factory=lambda: _Acc(results), inputs=['camera.raw'],
                   state_bytes_per_stream=64 << 10,
                   aggregate_outputs={'cube': {'count': 'sum'}}, aggregate_sink=sink.append)
    pool = EventPool(n=2, budget=budget)
    committed_before = budget.committed()
    ri = RunInstance('test', 77, 0, 0)

    scopes = [StepScope(run=77, index=k, begin_ts=0 if k == 0 else 900 + 50 * k)
              for k in range(3)]
    workers = TaskWorkers.build(task, streams=pool.streams, budget=budget, run_instance=ri,
                                input_preparers={'camera.raw': detector},
                                margin=budget.limit() // 10, scopes=scopes)
    assert workers.n == 2 and budget.committed() == committed_before + 2 * (64 << 10)
    assert all(workers.context(i).state.used > 0 for i in range(2)), 'bins came from ctx.state'

    def submit(scope):
        return pool.submit(
            NS(iter_events=lambda: iter(specs)), None, envelopes(specs),
            {'camera.raw': detector}, input_windows=(window,), batch_id=7,
            task=task, detector_bindings=mapping, task_constants=None,
            run=77, step_generation=scope.index, step_scope=scope,
            run_instance=ri, task_workers=workers)

    try:
        submit(scopes[0])      # slot 0 -> instance 0
        submit(scopes[2])      # slot 1 -> instance 1: never saw 0 or 1
        list(pool.flush())
        # Submissions commit storage of their own (preparer slot buffers) that
        # outlives the task, so measure what finish_run releases, not a return
        # to the pre-submission total.
        committed_after_flush = budget.committed()
        workers.finish_run('complete')

        per_submission = int(sum(int(e.sum()) for e in expected))
        n = len(specs)
        status0, sums0, counts0, hooks0 = results[0]
        status1, sums1, counts1, hooks1 = results[1]
        assert status0 == status1 == 'complete'
        # Hook placement: catch-up pairs for unseen scopes, open/close around
        # the one each instance processed, the rest completed at finish.
        assert hooks0 == [('begin_step', 0),
                          ('end_step', 0), ('begin_step', 1), ('end_step', 1),
                          ('begin_step', 2), ('end_step', 2)]
        assert hooks1 == [('begin_step', 0), ('end_step', 0), ('begin_step', 1), ('end_step', 1),
                          ('begin_step', 2),
                          ('end_step', 2)]
        assert sums0.tolist() == [per_submission, 0, 0, 0] and counts0.tolist() == [n, 0, 0, 0]
        assert sums1.tolist() == [0, 0, per_submission, 0] and counts1.tolist() == [0, 0, n, 0]
        assert committed_after_flush - budget.committed() == 2 * (64 << 10), 'reservation returned at finish'
        # The sink: one complete record per scope, both streams, merged counts.
        assert [(r.begin_ts, r.fields['count'].tolist(), r.complete, r.streams) for r in sink] == [
            (scopes[0].begin_ts, [n], True, (0, 1)),
            (scopes[1].begin_ts, [0], True, (0, 1)),
            (scopes[2].begin_ts, [n], True, (0, 1))]
    finally:
        list(pool.flush())
        workers.finish_run('incomplete')     # no-op after the first finish
        window.close()
