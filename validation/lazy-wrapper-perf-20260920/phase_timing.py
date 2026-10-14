"""Opt-in, benchmark-only host timers/NVTX for the frozen A/B implementations.

No CUDA synchronization is added. Coarse statement ranges (not per-segment
wrappers) preserve generator yields and the original asynchronous execution.
AST matching deliberately fails closed if the historical implementation changes.
"""
import ast
from collections import Counter
from contextlib import contextmanager
from functools import wraps
import hashlib
import inspect
import textwrap
import time


class Recorder:
    def __init__(self, variant, nvtx=None, clock=time.perf_counter_ns):
        self.variant, self.nvtx, self.clock = variant, nvtx, clock
        self.active = False
        self.steady = False
        self.stack = []
        self.stats = {}
        self.inventory = []

    @contextmanager
    def phase(self, name):
        if not self.active:
            yield
            return
        label = 'steady' if self.steady else 'startup'
        if self.nvtx is not None:
            self.nvtx.RangePush('psana.' + self.variant + '.' + name)
        frame = [self.clock(), 0]
        self.stack.append(frame)
        try:
            yield
        finally:
            elapsed = self.clock() - frame[0]
            assert self.stack.pop() is frame
            if self.stack:
                self.stack[-1][1] += elapsed
            stat = self.stats.setdefault(label + '/' + name,
                                         dict(calls=0, total_ns=0, self_ns=0, max_ns=0))
            stat['calls'] += 1
            stat['total_ns'] += elapsed
            stat['self_ns'] += elapsed - frame[1]
            stat['max_ns'] = max(stat['max_ns'], elapsed)
            if self.nvtx is not None:
                self.nvtx.RangePop()

    def snapshot(self):
        assert not self.stack, 'range leaked across a yield'
        return dict(phases=self.stats, patches=self.inventory,
                    units='nanoseconds; host wall time, not CUDA execution time',
                    steady_definition='after first event is delivered; excludes first submitted subbatch')


def _call_name(node):
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
    return None


class Ranges(ast.NodeTransformer):
    def __init__(self, kind, variant):
        self.kind, self.variant = kind, variant
        self.matches = Counter()

    def range(self, node, name):
        self.matches[name] += 1
        context = ast.Call(func=ast.Attribute(value=ast.Name(id='_bench_phases', ctx=ast.Load()),
                                             attr='phase', ctx=ast.Load()),
                           args=[ast.Constant(name)], keywords=[])
        return ast.copy_location(ast.With(items=[ast.withitem(context_expr=context)], body=[node]), node)

    def visit_Assign(self, node):
        call = _call_name(node.value)
        if self.kind == 'parse':
            if call == 'prepare':
                return self.range(node, 'xtc.metadata')
            if call == 'GpuEventBatch':
                return self.range(node, 'xtc.walk')
        if self.kind == 'detector' and self.variant in ('B', 'O', 'G', 'Z'):
            if any(isinstance(t, ast.Name) and t.id == 'events_info' for t in node.targets):
                return self.range(node, 'detector.sources')
        return self.generic_visit(node)

    def visit_If(self, node):
        if (self.kind == 'parse' and isinstance(node.test, ast.Attribute)
                and node.test.attr == 'field_handles'):
            return self.range(node, 'xtc.locate_all')
        return self.generic_visit(node)

    def visit_With(self, node):
        if self.kind == 'detector' and self.variant in ('G', 'Z') and any(
                isinstance(child, ast.Call) and _call_name(child) == 'gather'
                for child in ast.walk(node)):
            return self.range(node, 'detector.gather')
        return self.generic_visit(node)

    def visit_For(self, node):
        call = _call_name(node.iter)
        if self.kind == 'parse' and isinstance(node.iter, ast.Attribute) and node.iter.attr == 'field_handles':
            return self.range(node, 'xtc.locate_all')
        if self.kind == 'detector':
            if call == 'iter_events':
                return self.range(node, 'detector.sources')
            if call == 'iter_sources' or (isinstance(node.target, ast.Tuple) and
                    isinstance(node.target.elts[0], ast.Name) and node.target.elts[0].id == 'desc_row'):
                return self.range(node, 'detector.gather')
        return self.generic_visit(node)

    def visit_Expr(self, node):
        call = _call_name(node.value)
        if self.kind == 'detector':
            if call == 'fused_calib_gpu':
                return self.range(node, 'detector.calib')
            if call == '_zero_missing_rows_gpu':
                return self.range(node, 'detector.zero_missing')
            if call == 'fill' and isinstance(node.value.func.value, ast.Name):
                target = node.value.func.value.id
                if target in ('target', 'present'):
                    return self.range(node, 'detector.zero_' + target)
        return self.generic_visit(node)


def transform(source, kind, variant):
    tree = ast.parse(textwrap.dedent(source))
    fn = tree.body[0]
    fn.decorator_list = []
    visitor = Ranges(kind, variant)
    tree = visitor.visit(tree)
    expected = ({'xtc.metadata': 1, 'xtc.walk': 1, 'xtc.locate_all': 1} if kind == 'parse'
                else {'detector.sources': 1, 'detector.gather': 1, 'detector.calib': 1,
                      'detector.zero_target': 1})
    if kind == 'detector' and variant in ('B', 'O', 'G', 'Z'):
        expected.update({'detector.zero_present': 1, 'detector.zero_missing': 1})
    if kind == 'detector' and variant in ('G', 'Z'):
        expected.pop('detector.zero_target')
        expected.pop('detector.zero_present')
    if dict(visitor.matches) != expected:
        raise RuntimeError(f'unexpected instrumentation matches: {visitor.matches}, expected {expected}')
    ast.fix_missing_locations(tree)
    # No newly introduced range may encompass a generator yield.
    for node in ast.walk(tree):
        if isinstance(node, ast.With) and isinstance(node.items[0].context_expr, ast.Call):
            if _call_name(node.items[0].context_expr) == 'phase':
                assert not any(isinstance(child, (ast.Yield, ast.YieldFrom)) for child in ast.walk(node))
    return tree, dict(visitor.matches)


def install(variant, nvtx=None):
    if variant not in ('A', 'B', 'O', 'G', 'Z'):
        raise ValueError('timing hooks support A/B/B+ revisions')
    from psana.gpu.gpu_detector import GPUDetector
    from psana.gpu.gpu_events import GpuEventManager
    from psana.gpu.gpu_kvikio_read import KvikioGpuReader
    from psana.gpu.gpu_stream import EventPool
    recorder = Recorder(variant, nvtx)

    def remember(fn, ranges):
        source = inspect.getsource(fn)
        recorder.inventory.append(dict(function=fn.__qualname__, path=inspect.getfile(fn),
                                       source_sha256=hashlib.sha256(source.encode()).hexdigest(), ranges=ranges))
        return source

    def wrap(owner, method, phase):
        descriptor = inspect.getattr_static(owner, method)
        fn = descriptor.__func__ if isinstance(descriptor, classmethod) else descriptor
        assert not inspect.isgeneratorfunction(fn)
        remember(fn, [phase])
        @wraps(fn)
        def measured(*args, **kwargs):
            with recorder.phase(phase):
                return fn(*args, **kwargs)
        setattr(owner, method, classmethod(measured) if isinstance(descriptor, classmethod) else measured)

    def rewrite(owner, method, kind):
        fn = getattr(owner, method)
        assert not fn.__code__.co_freevars
        source = remember(fn, kind)
        tree, matches = transform(source, kind, variant)
        namespace = {}
        fn.__globals__['_bench_phases'] = recorder
        exec(compile(tree, inspect.getfile(fn) + ':benchmark-ranges', 'exec'), fn.__globals__, namespace)
        setattr(owner, method, wraps(fn)(namespace[fn.__name__]))
        recorder.inventory[-1]['ranges'] = matches

    wrap(GpuEventManager, '_setup_detectors', 'setup.detectors')
    wrap(GpuEventManager, '_next_batch', 'upstream.next_batch')
    wrap(KvikioGpuReader, 'issue_batch', 'read.submit')
    wrap(KvikioGpuReader, 'wait_batch', 'read.wait')
    wrap(EventPool, 'submit', 'submit')
    wrap(EventPool, 'begin_retire_next', 'retire.producer_wait')
    wrap(EventPool, 'finish_retire_next', 'retire.consumer_wait')
    wrap(GPUDetector, '_slot_buffer', 'detector.buffers')
    rewrite(GPUDetector, 'process_batch', 'detector')
    if variant in ('B', 'O', 'G', 'Z'):
        from psana.gpu.gpudgram.batch import GpuXtcBatchPool
        from psana.gpu.gpu_input import GpuEventDgrams
        rewrite(GpuXtcBatchPool, 'parse', 'parse')
        wrap(GpuEventDgrams, 'from_batch', 'event.views')
    return recorder
