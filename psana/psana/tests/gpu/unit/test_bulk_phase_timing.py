"""CPU tests for benchmark timer accounting and generator safety."""
import ast
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location('bulk_phase_timing', ROOT / 'gpu/scripts/bulk_phase_timing.py')
timing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(timing)


def test_nested_times_partition_wall_time():
    ticks = iter([0, 10, 30, 50])
    recorder = timing.Recorder('test', clock=lambda: next(ticks))
    recorder.active = True
    with recorder.phase('parent'):
        with recorder.phase('child'):
            pass
    phases = recorder.snapshot()['phases']
    assert phases['startup/parent']['total_ns'] == 50
    assert phases['startup/parent']['self_ns'] == 30
    assert sum(s['self_ns'] for s in phases.values()) == 50


def test_generator_send_return_throw_and_close_leave_no_open_timer():
    recorder = timing.Recorder('test')
    recorder.active = True
    closed = []

    def source():
        try:
            value = yield 1
            try:
                yield value
            except ValueError:
                yield 3
            return 42
        finally:
            closed.append(True)

    wrapped = timing.measured_function(source, recorder, 'generator')
    gen = wrapped()
    assert next(gen) == 1
    assert not recorder.stack
    assert gen.send(2) == 2
    assert gen.throw(ValueError) == 3
    with pytest.raises(StopIteration) as end:
        next(gen)
    assert end.value.value == 42
    gen = wrapped()
    assert next(gen) == 1
    gen.close()
    assert closed == [True, True]
    assert not recorder.stack


def test_caller_time_at_yield_is_excluded():
    now = [0]
    recorder = timing.Recorder('test', clock=lambda: now[0])
    recorder.active = True

    def source():
        now[0] += 5
        yield 1
        now[0] += 7

    gen = timing.measured_function(source, recorder, 'generator')()
    next(gen)
    now[0] += 1000
    list(gen)
    assert recorder.snapshot()['phases']['startup/generator']['total_ns'] == 12


@pytest.mark.parametrize('filename,cls,method,kind', [
    ('gpu_kvikio_read.py', 'KvikioGpuReader', '_submit_read', 'read_submit'),
    ('gpu_stream.py', 'EventPool', 'submit', 'pool_submit'),
    ('gpu_stream.py', 'EventPool', 'flush', 'pool_flush'),
    ('gpu_stream.py', 'EventPool', 'begin_retire_next', 'producer_retire'),
    ('gpu_stream.py', 'EventPool', 'finish_retire_next', 'consumer_release'),
])
def test_statement_boundaries_match_reviewed_runtime(filename, cls, method, kind):
    source = (ROOT / 'gpu' / filename).read_text()
    tree = ast.parse(source)
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    fn = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == method)
    instrumented, _ = timing.timed_statements(ast.get_source_segment(source, fn), kind)
    compile(instrumented, filename, 'exec')


def test_statement_instrumentation_rejects_changed_source():
    with pytest.raises(RuntimeError, match='matches'):
        timing.timed_statements('def changed():\n    pass\n', 'read_submit')


def test_read_counters_survive_batch_resets_and_multiple_reads():
    counters = {}
    zero = dict(io_path='CPU-fallback', compat_mode=True, bulk_read=True,
                total_bytes=0, useful_bytes=0, total_ns=0, issue_to_complete_ns=0)
    first = dict(zero, total_bytes=100, useful_bytes=100, total_ns=10, issue_to_complete_ns=12)
    second = dict(first, total_bytes=160, useful_bytes=160, total_ns=15, issue_to_complete_ns=20)
    timing.accumulate_read(counters, zero, first, 3)
    timing.accumulate_read(counters, first, second, 2)
    # A new EB batch has reset all reader statistics to zero.
    timing.accumulate_read(counters, zero, first, 3)
    assert counters['reader']['total_requests'] == 8
    assert counters['reader']['useful_bytes'] == 260
    assert counters['reader']['total_ns'] == 25
