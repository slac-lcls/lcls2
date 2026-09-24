"""Opt-in host timing for the integrated JF benchmark; no extra CUDA waits.

Inclusive times describe call trees. Exclusive times partition the BD thread's
wall time. Generator timers cover resume/close only, never time at a yield.
Installed functions and statement matches are recorded for reproducibility.
"""
import ast
from collections import Counter
from contextlib import contextmanager
from functools import wraps
import hashlib
import inspect
import sys
import textwrap
import time


class Recorder:
    def __init__(self, variant, nvtx=None, clock=time.perf_counter_ns):
        self.variant, self.nvtx, self.clock = variant, nvtx, clock
        self.active = self.steady = False
        self.stack, self.inventory = [], []
        self.stats = {}
        self.counters = {}

    @contextmanager
    def phase(self, name):
        if not self.active:
            yield
            return
        label = ('steady/' if self.steady else 'startup/') + name
        if self.nvtx is not None:
            self.nvtx.RangePush('psana.' + name)
        frame = [self.clock(), 0]
        self.stack.append(frame)
        try:
            yield
        finally:
            elapsed = self.clock() - frame[0]
            assert self.stack.pop() is frame
            if self.stack:
                self.stack[-1][1] += elapsed
            stat = self.stats.setdefault(label, dict(calls=0, total_ns=0, self_ns=0, max_ns=0))
            stat['calls'] += 1
            stat['total_ns'] += elapsed
            stat['self_ns'] += elapsed - frame[1]
            stat['max_ns'] = max(stat['max_ns'], elapsed)
            if self.nvtx is not None:
                self.nvtx.RangePop()

    def snapshot(self):
        assert not self.stack, 'timer leaked across generator yield'
        return dict(phases=self.stats, patches=self.inventory, counters=self.counters,
                    units='host wall nanoseconds, not CUDA kernel duration',
                    steady_definition='after first delivered event')


class TimedIterator:
    """Preserve yield-from return, send, throw, and close semantics."""
    def __init__(self, iterator, recorder, phase):
        self.iterator, self.recorder, self.phase = iterator, recorder, phase

    def __iter__(self):
        return self

    def __next__(self):
        with self.recorder.phase(self.phase):
            return next(self.iterator)

    def send(self, value):
        with self.recorder.phase(self.phase):
            return self.iterator.send(value)

    def throw(self, *args):
        with self.recorder.phase(self.phase):
            return self.iterator.throw(*args)

    def close(self):
        with self.recorder.phase(self.phase):
            return self.iterator.close()


def measured_function(fn, recorder, phase):
    if inspect.isgeneratorfunction(fn):
        @wraps(fn)
        def measured(*args, **kwargs):
            return (yield from TimedIterator(fn(*args, **kwargs), recorder, phase))
    else:
        @wraps(fn)
        def measured(*args, **kwargs):
            with recorder.phase(phase):
                return fn(*args, **kwargs)
    return measured


def accumulate_read(counters, before, after, requests):
    """Reader totals reset at every EB batch; accumulate completion deltas."""
    current = counters.setdefault('reader', dict(
        io_path=after['io_path'], compat_mode=after['compat_mode'],
        bulk_read=after['bulk_read'], total_requests=0,
        total_bytes=0, useful_bytes=0, total_ns=0, issue_to_complete_ns=0))
    current['total_requests'] += requests
    for name in ('total_bytes', 'useful_bytes', 'total_ns', 'issue_to_complete_ns'):
        delta = after[name] - before[name]
        assert delta >= 0, 'reader reset during a wait'
        current[name] += delta


def timed_statements(source, kind):
    """Fail closed if the reviewed coarse statement boundaries change."""
    matches = Counter()

    class Ranges(ast.NodeTransformer):
        def visit(self, node):
            phase = None
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                fn = node.value.func
                if isinstance(fn, ast.Attribute) and fn.attr == 'synchronize':
                    phase = 'sync.' + kind
            if (kind == 'read_submit' and isinstance(node, ast.For)
                    and isinstance(node.iter, ast.Name) and node.iter.id == 'ranges'):
                phase = 'read.pread_loop'
            if (kind in ('pool_flush', 'consumer_release') and isinstance(node, ast.For)
                    and isinstance(node.iter, ast.Attribute) and node.iter.attr == 'leases'):
                phase = 'retire.consumer_join'
            if phase is None:
                return super().visit(node)
            assert not any(isinstance(child, (ast.Yield, ast.YieldFrom)) for child in ast.walk(node))
            matches[phase] += 1
            context = ast.Call(func=ast.Attribute(value=ast.Name(id='_bulk_timer', ctx=ast.Load()),
                                                 attr='phase', ctx=ast.Load()),
                               args=[ast.Constant(phase)], keywords=[])
            return ast.copy_location(ast.With(items=[ast.withitem(context_expr=context)], body=[node]), node)

    tree = ast.parse(textwrap.dedent(source))
    tree.body[0].decorator_list = []
    tree = Ranges().visit(tree)
    expected = {
        'read_submit': {'read.pread_loop': 1},
        'pool_submit': {'sync.pool_submit': 2},  # null stream plus exception drain
        'pool_flush': {'sync.pool_flush': 1, 'retire.consumer_join': 1},
        'producer_retire': {'sync.producer_retire': 1},
        'consumer_release': {'retire.consumer_join': 1},
    }[kind]
    if dict(matches) != expected:
        raise RuntimeError(f'{kind}: matches {dict(matches)} != {expected}')
    return ast.fix_missing_locations(tree), dict(matches)


def install(variant, nvtx=None):
    if variant not in ('Integrated-off', 'Integrated-on'):
        raise ValueError('this timer supports the integrated runtime only')
    from psana.gpu import gpu_allocation as allocation
    from psana.gpu import gpu_detector as detector
    from psana.gpu.gpu_events import GpuEventManager as Manager
    from psana.gpu.gpu_kvikio_read import KvikioGpuReader as Reader
    from psana.gpu.gpu_stream import EventPool as Pool
    from psana.gpu.gpu_input import GpuEventDgrams
    from psana.gpu.gpu_file_epochs import GpuFileEpochs
    from psana.gpu.gpudgram.batch import GpuXtcBatchPool, _GpuXtcSlotBuffers
    from psana.gpu.gpudgram.parser import GpuEventBatch
    recorder = Recorder(variant, nvtx)

    def remember(fn, ranges):
        source = inspect.getsource(fn)
        recorder.inventory.append(dict(function=fn.__qualname__, path=inspect.getfile(fn),
                                       source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                                       ranges=ranges))

    for owner, method, kind in (
        (Reader, 'issue_batch', 'read_submit'),
        (Pool, 'submit', 'pool_submit'),
        (Pool, 'flush', 'pool_flush'),
        (Pool, 'begin_retire_next', 'producer_retire'),
        (Pool, 'finish_retire_next', 'consumer_release'),
    ):
        fn = getattr(owner, method)
        assert not fn.__code__.co_freevars
        tree, matches = timed_statements(inspect.getsource(fn), kind)
        remember(fn, matches)
        fn.__globals__['_bulk_timer'] = recorder
        namespace = {}
        exec(compile(tree, inspect.getfile(fn) + ':bulk-timing', 'exec'), fn.__globals__, namespace)
        setattr(owner, method, wraps(fn)(namespace[fn.__name__]))

    def wrap(owner, method, phase):
        descriptor = inspect.getattr_static(owner, method)
        fn = descriptor.__func__ if isinstance(descriptor, (classmethod, staticmethod)) else descriptor
        remember(fn, [phase])
        measured = measured_function(fn, recorder, phase)
        if isinstance(descriptor, classmethod):
            measured = classmethod(measured)
        elif isinstance(descriptor, staticmethod):
            measured = staticmethod(measured)
        setattr(owner, method, measured)

    targets = [
        (Manager, '_setup_gpu_pipeline', 'setup.pipeline'),
        (Manager, '_next_batch', 'upstream.next_batch'),
        (Manager, '_process_batch', 'manager.process'),
        (Manager, '_handle_steps', 'transition.handle'),
        (Manager, '_split_subbatches', 'admission.plan'),
        (Manager, '_reserve_gpu_subbatch', 'admission.reserve_execution'),
        (Manager, '_start_resident_input', 'resident.start'),
        (Manager, '_close_resident_input', 'resident.close'),
        (Manager, '_trim_gpu_caches', 'cache.trim'),
        (Manager, '_flush_event_pool', 'manager.flush'),
        (Manager, '_issue_gpu_read', 'manager.issue_read'),
        (Manager, '_submit_gpu', 'manager.submit'),
        (Manager, '_yield_ready', 'delivery.events'),
        (GpuFileEpochs, 'resolve', 'read.file_epochs'),
        (Reader, 'issue_batch', 'read.submit'),
        (Reader, '_ensure_slot_buffer', 'read.buffer'),
        (Reader, '_coalesced_plan', 'read.coalesce'),
        (Reader, 'wait_batch', 'read.wait'),
        (Pool, 'submit', 'pool.submit'),
        (Pool, 'flush', 'pool.flush'),
        (Pool, 'begin_retire_next', 'retire.begin'),
        (Pool, 'finish_retire_next', 'retire.finish'),
        (GpuXtcBatchPool, 'parse_window', 'parser.window'),
        (GpuXtcBatchPool, 'parse', 'parser.parse'),
        (_GpuXtcSlotBuffers, 'prepare', 'parser.metadata'),
        (_GpuXtcSlotBuffers, 'batched_locator_rows', 'parser.locator_buffer'),
        (GpuEventBatch, '__init__', 'parser.walk_submit'),
        (GpuEventBatch, '_locate_configured', 'parser.locate_submit'),
        (GpuEventDgrams, 'from_windows', 'event.views'),
        (detector.GPUDetector, 'process_batch', 'detector.process'),
        (detector.GPUDetector, '_slot_buffer', 'detector.buffer'),
        (detector._GatherMap, 'prepare', 'detector.owner_map'),
        (detector._CanonicalGatherPlan, 'gather', 'detector.gather_submit'),
        (detector, 'fused_calib_gpu', 'detector.calib_submit'),
        (detector, '_zero_missing_rows_gpu', 'detector.cleanup_submit'),
    ]
    for target in targets:
        wrap(*target)

    wait = Reader.wait_batch

    @wraps(wait)
    def observed_wait(self, *args, **kwargs):
        pending = args[0] if args else kwargs['pending']
        collect = recorder.active and not pending.completed
        before = self.io_stats() if collect else None
        result = wait(self, *args, **kwargs)
        if collect:
            accumulate_read(recorder.counters, before, self.io_stats(), len(pending.futures))
        return result

    Reader.wait_batch = observed_wait

    # Patch existing imported aliases, so calls count actual owned_empty invocations
    # rather than reserve/commit bookkeeping. These are not cudaMalloc counts.
    original = allocation.owned_empty
    remember(original, ['allocation.owned_empty'])
    timed_allocation = measured_function(original, recorder, 'allocation.owned_empty')

    @wraps(original)
    def measured(*args, **kwargs):
        if recorder.active:
            category = kwargs.get('category', args[4] if len(args) > 4 else 'unknown')
            counts = recorder.counters.setdefault('owned_allocations', {})
            counts[category] = counts.get(category, 0) + 1
        return timed_allocation(*args, **kwargs)
    aliases = []
    for name, module in list(sys.modules.items()):
        if name.startswith('psana.gpu.') and module is not None:
            for attr, value in list(vars(module).items()):
                if value is original:
                    setattr(module, attr, measured)
                    aliases.append(name + '.' + attr)
    recorder.inventory[-1]['aliases'] = aliases
    return recorder
