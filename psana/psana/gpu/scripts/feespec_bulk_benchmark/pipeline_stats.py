"""Diagnostic-only launch/subbatch counts and exact charged-allocation peak.

Installed in separate samples, never in controls or native traces. No CUDA
synchronization is added. Charged memory includes cached owned allocations;
it excludes CUDA context, allocator cache and unowned user allocations.
"""
from collections import Counter
from functools import wraps
from weakref import WeakSet


class PipelineStats:
    def __init__(self):
        from psana.gpu.gpudgram import parser
        from psana.gpu import gpu_detector, gpu_calib
        from psana.gpu.gpu_budget import _GpuBudget
        from psana.gpu.gpu_stream import EventPool
        self.active = False
        self.budgets = WeakSet()
        self.launches = Counter()
        self.subbatches = []
        self.peak = 0
        self.reserve_calls = 0
        for module, name, label in (
            (parser, '_walk_kernel', 'walk'),
            (parser, '_init_locators_kernel', 'init'),
            (parser, '_locate_fields_kernel', 'locate'),
            (gpu_detector, '_batched_gather_kernel', 'gather'),
            (gpu_calib, '_jungfrau_calib_kernel', 'calibration')):
            self._kernel(module, name, label)
        reserve, submit = _GpuBudget.reserve, EventPool.submit

        @wraps(reserve)
        def reserved(budget, n):
            result = reserve(budget, n)
            self.budgets.add(budget)
            if self.active:
                self.reserve_calls += 1
                self.peak = max(self.peak, budget.committed())
            return result

        @wraps(submit)
        def submitted(pool, gv, *args, **kwargs):
            result = submit(pool, gv, *args, **kwargs)
            if self.active:
                self.subbatches.append(int(gv.n_events))
            return result

        _GpuBudget.reserve, EventPool.submit = reserved, submitted

    def _kernel(self, module, name, label):
        factory = getattr(module, name)

        @wraps(factory)
        def wrapped(*args, **kwargs):
            kernel = factory(*args, **kwargs)

            def launch(*args, **kwargs):
                if self.active:
                    self.launches[label] += 1
                return kernel(*args, **kwargs)
            return launch
        setattr(module, name, wrapped)

    def begin(self):
        self.launches.clear()
        self.subbatches.clear()
        self.reserve_calls = 0
        self.peak = max((b.committed() for b in self.budgets), default=0)
        self.active = True

    def end(self):
        self.active = False
        return dict(launches=dict(self.launches), subbatches=list(self.subbatches),
                    peak_charged_bytes=self.peak, allocation_reserve_calls=self.reserve_calls)
