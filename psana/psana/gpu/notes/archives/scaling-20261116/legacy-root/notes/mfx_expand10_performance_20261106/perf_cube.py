"""Performance orchestration around the unchanged reviewed sample10/cube kernels."""
from pathlib import Path
import sys

import numpy as np

CAPACITY = Path(__file__).resolve().parents[1]/'mfx_expand10_capacity_20261106'
sys.path.insert(0, str(CAPACITY))
import dense_cube
from timing import Kernel, KernelCache, Window


class Capture(dense_cube.reference.Capture):
    def publish(self, name, array):
        self.batch.keepalive(array)
        self.arrays[name] = array
        # No publications/D2H in this runner. accept_events checks every panel's
        # errors on device; post-drain per-bin counts must match all inputs.


class Cube(dense_cube.Cube):
    def __init__(self, timestamps, *, warmup, recorder, **kwargs):
        super().__init__(timestamps, **kwargs)
        self.window = Window(warmup, len(self.ordinal))
        self.recorder = recorder
        self.recorder.phase = lambda: self.window.phase
        self.callbacks = []

    def prepare(self):
        if self.kernel is not None or self.seen:
            raise RuntimeError('Prepare once before processing')
        import cupy as cp
        for name, source, symbol, owner, attr, options in (
            ('expansion_kernel', dense_cube.adapter.kernel_source(), 'expand10', self.sample10, 'kernel', ('--std=c++17',)),
            ('cube_sum_kernel', dense_cube.reference.KERNEL, 'accumulate', self, 'kernel', ('--std=c++17','--fmad=false')),
            ('cube_count_kernel', dense_cube.reference.KERNEL, 'accept_events', self, 'accept_kernel', ('--std=c++17',))):
            kernel = cp.RawKernel(source, symbol, options=options)
            kernel.compile()
            setattr(owner, attr, Kernel(kernel, name, self.recorder) if self.recorder.enabled else kernel)
        if self.recorder.enabled:
            self.sample10.calibration._kernels = KernelCache(self.recorder)

    def join(self):
        for bank in self.banks:
            bank.stream.synchronize()

    def _warm_join(self):
        if len(self.banks) != self.max_streams:
            raise RuntimeError('Warmup did not exercise every configured stream')
        self.join()

    def __call__(self, batch, stream):
        self.window.before(self._warm_join)
        self.recorder.callback = len(self.callbacks)
        try:
            self._identify(batch.timestamps)
            capture = Capture(batch)
            with self.recorder.cpu('expand_calibrate_call'):
                self.sample10(capture, stream)
            with self.recorder.cpu('cube_bins_upload_accumulate_call'):
                self.add(capture.arrays['jungfrau.calib'], capture.arrays['sim10.errors'],
                         batch.timestamps, stream, batch.keepalive, batch.step_generation)
            self.callbacks.append(dict(size=batch.size, stream=int(stream.ptr), phase=self.window.phase))
            self.window.after(batch.size)
        finally:
            self.recorder.callback = None

    def check_counts(self):
        """Call after EOF and explicit join; validates rejection and bin routing."""
        if len(self.seen) != len(self.ordinal) or len(self.banks) != self.max_streams:
            raise RuntimeError('Incomplete cube or missing stream')
        counts = np.zeros(self.nbins, np.uint64)
        bank_counts = []
        for bank in self.banks:
            got = bank.counts.get()
            expected = np.bincount(np.asarray(bank.events, np.int64) % self.active_bins,
                                   minlength=self.nbins).astype(np.uint64)
            if not np.array_equal(got, expected):
                raise RuntimeError('Rejected/mis-binned event or wrong bank count')
            counts += got
            bank_counts.append(got.tolist())
        return dict(status='pass', counts=counts.tolist(), per_bank_counts=bank_counts,
                    events=int(counts.sum()),
                    scope='Exact accepted event/bin counts; no new full-pixel numerical acceptance claim')
