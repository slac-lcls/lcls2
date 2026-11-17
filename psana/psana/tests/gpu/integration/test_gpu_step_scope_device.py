"""Step scope as a CUDA bin index: one output bin per step (issue #153).

Everything else that reads `step_generation` does so host-side and throws it
away -- `calls.append((batch.size, batch.step_generation))` and the like. That
leaves the thing #153 actually promises untested: that the scope can *drive* a
per-step accumulation on the device, which is what "one output bin per step"
means for a user kernel.

So here a real GpuTask callback launches a real RawKernel that atomically adds
each event's pixel sum into `bins[step_scope.index]`. If the scope were wrong
-- the GPU-148-01 defect, where events before a trailing BeginStep are
submitted with the next step's generation -- the counts land in the wrong bin
and the assertions below fail. Run 77 of mfx100852324 is the real-data version
of exactly this shape: 3 steps, 1000 events each, one bin per step.

Marked `gpu`; skipped without a CUDA device.
"""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from test_batched_gather import _gpu_available
from test_gpu_producer_device import bindings, envelopes, fixture
from psana.gpu import GpuTask
from psana.gpu.gpu_plan import StepScope
from psana.gpu.gpu_stream import EventPool

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not _gpu_available(), reason='no CUDA device')]


# One bin per step. atomicAdd because events from several subbatches of the
# same step accumulate into the same bin, on different streams.
_BIN_KERNEL = r'''
extern "C" __global__
void bin_by_step(const unsigned short* pixels, unsigned long long n_pixels,
                 unsigned int step_index, unsigned long long* bins) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x
                         + threadIdx.x;
    if (i < n_pixels) {
        atomicAdd(bins + step_index, (unsigned long long)pixels[i]);
    }
}
'''


def _accumulate(cp, bins, n_bins):
    """A task callback that bins its events by the scope it was submitted with."""
    kernel = cp.RawKernel(_BIN_KERNEL, 'bin_by_step')
    kernel.compile()
    seen = []

    def callback(batch, stream):
        scope = batch.step_scope
        # The contract under test: the scope arrives on the device-side batch,
        # and its index is the bin. step_generation must agree with it.
        assert scope is not None, 'no step scope reached the callback'
        assert batch.step_generation == scope.index
        assert 0 <= scope.index < n_bins
        seen.append((batch.size, scope.index, scope.begin_ts))

        pixels = batch.input('camera.raw')
        flat = pixels.reshape(-1)
        threads = 256
        blocks = (flat.size + threads - 1) // threads
        kernel((blocks,), (threads,),
               (flat, np.uint64(flat.size), np.uint32(scope.index), bins),
               stream=stream)
        # Publish per-event so the record retires like any other task batch.
        batch.publish('step_index',
                      cp.full((batch.size,), scope.index, cp.uint32))

    return callback, seen


def _submit(cp, pool, specs, window, mapping, detector, callback, scope):
    task = GpuTask(callback, ['camera.raw'], [])
    return pool.submit(
        NS(iter_events=lambda: iter(specs)), None, envelopes(specs),
        {'camera.raw': detector}, input_windows=(window,), batch_id=7,
        task=task, detector_bindings=mapping, task_constants=None,
        run=scope.run, step_generation=scope.index, step_scope=scope)


def test_a_kernel_bins_by_step_scope_not_by_manager_state():
    """Two submissions in different scopes land in different bins.

    This is the device-side statement of GPU-148-01: the bin comes from the
    scope the subbatch carries, so a submission made *after* the manager's
    generation advanced still bins under the step its events belong to.
    """
    import cupy as cp
    parser, detector, budget, specs, expected, window = fixture(cp, ((0,),) * 3)
    mapping = bindings(parser, detector)
    bins = cp.zeros(3, cp.uint64)
    callback, seen = _accumulate(cp, bins, n_bins=3)
    # n=1: begin_retire_next() retires next_slot_id, so with a deeper pool the
    # first submission's slot is not the one it looks at and it returns None.
    # Retiring each submission before the next also makes the bin counts below
    # unambiguous.
    pool = EventPool(n=1, budget=budget)

    try:
        scope0 = StepScope(run=77, index=0, begin_ts=0)
        record = _submit(cp, pool, specs, window, mapping, detector,
                         callback, scope0)
        assert pool.begin_retire_next() is record
        pool.finish_retire_next()

        # The pixel sum the kernel should have accumulated, computed on the
        # host from the same input the gather produced.
        per_submission = int(sum(int(e.sum()) for e in expected))
        counts = cp.asnumpy(bins)
        assert counts[0] == per_submission, 'scope 0 did not accumulate'
        assert counts[1] == 0 and counts[2] == 0, 'bled into another step'

        # Now a second submission in scope 2, skipping 1 -- an empty step, as
        # a BD that missed one sees after catch-up history.
        scope2 = StepScope(run=77, index=2, begin_ts=950)
        record = _submit(cp, pool, specs, window, mapping, detector,
                         callback, scope2)
        assert pool.begin_retire_next() is record
        pool.finish_retire_next()

        counts = cp.asnumpy(bins)
        assert counts[0] == per_submission, 'scope 0 changed retroactively'
        assert counts[1] == 0, 'the skipped step must stay empty'
        assert counts[2] == per_submission, 'scope 2 did not accumulate'

        assert [index for _, index, _ in seen] == [0, 2]
        assert [begin_ts for _, _, begin_ts in seen] == [0, 950]
    finally:
        list(pool.flush())
        window.close()


def test_the_published_step_index_follows_the_scope():
    """The scope is also visible per event in the published output.

    A kernel binning into device memory proves the index reached the device;
    this proves the same index is what a consumer reads back per event, which
    is how a user builds a per-step cube.
    """
    import cupy as cp
    parser, detector, budget, specs, expected, window = fixture(cp, ((0,),) * 3)
    mapping = bindings(parser, detector)
    bins = cp.zeros(2, cp.uint64)
    callback, _ = _accumulate(cp, bins, n_bins=2)
    pool = EventPool(n=1, budget=budget)

    try:
        scope = StepScope(run=77, index=1, begin_ts=900)
        record = _submit(cp, pool, specs, window, mapping, detector,
                         callback, scope)
        assert pool.begin_retire_next() is record
        for _, outputs in record.publications_by_ts.items():
            assert int(outputs['step_index'].array.get()) == 1
        pool.finish_retire_next()
    finally:
        list(pool.flush())
        window.close()
