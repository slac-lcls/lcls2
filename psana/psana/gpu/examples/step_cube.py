"""A step-scan cube as a stateful GPU task: per-pixel sums and per-segment
counts of one dense input, one bank per execution stream, exported per step.

Copy this file outside psana and adapt it. It shows the mechanics PR #270
provides -- run-owned budgeted state, begin_step/end_step placed per stream
by the framework, ctx.emit to a per-BD reducer -- with integer arithmetic a
CPU reference can match exactly. It sums raw ADC words: calibration,
thresholds and masks are the user's science and belong in process() before
the accumulation.

    task = GpuTask(factory=partial(StepCube, 'jungfrau'), inputs=['jungfrau.raw'],
                   state_bytes_per_stream=StepCube.state_bytes(n_segments=8),
                   aggregate_outputs=StepCube.OUTPUTS, aggregate_sink=my_sink)

Each sink record is one step on one BD, merged over its streams: key = the
dense segment ids in bank order, fields sums (segments, rows, cols) uint64
and counts (segments,) uint64 -- the number of events in which that segment
was present. Missing segments contribute zero data and zero presence.
"""
from functools import partial  # noqa: F401  (re-exported for the docstring's example)

import numpy as np

from psana.gpu import GpuTaskWorker, state_bytes


class StepCube(GpuTaskWorker):
    OUTPUTS = {'cube': {'sums': 'sum', 'counts': 'sum'}}

    @staticmethod
    def state_bytes(n_segments, rows=512, cols=1024):
        """Bank size for state_bytes_per_stream: sums plus counts, uint64,
        at the allocator's rounding -- never sum nbytes by hand."""
        return state_bytes(((n_segments, rows, cols), np.uint64), ((n_segments,), np.uint64))

    def __init__(self, detector, output='cube'):
        self.detector, self.input, self.output = detector, detector + '.raw', output
        self.sums = self.counts = None
        self.segment_ids = None

    def begin_run(self, ctx, stream):
        spec = ctx.input_spec(self.input)
        self.sums = ctx.state.zeros(spec.event_shape, np.uint64)
        self.counts = ctx.state.zeros((spec.event_shape[0],), np.uint64)

    def begin_step(self, ctx, stream):
        self.sums.fill(0)
        self.counts.fill(0)
        self.segment_ids = None

    def process(self, batch, stream):
        images = batch.input(self.input)          # (N, segments, rows, cols)
        present = batch.present(self.input)       # (N, segments) uint8
        if images is None:
            return
        # Absent segments arrive as zeros, so the plain sum is already the
        # masked sum; only the counts need the presence array. The reduction
        # accumulates in uint64 without materialising a widened copy.
        self.sums += images.sum(axis=0, dtype=np.uint64)
        self.counts += present.sum(axis=0, dtype=np.uint64)
        self.segment_ids = tuple(int(s) for s in batch.segment_ids(self.detector))

    def end_step(self, ctx, stream):
        if self.segment_ids is None:
            return          # this stream saw no data for the step; completion is the framework's
        ctx.emit(self.output, key=self.segment_ids, sums=self.sums, counts=self.counts)

    def close(self):
        self.sums = self.counts = None
