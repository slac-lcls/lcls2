"""One gather across independent input bases, capacities, rows, and lifetimes."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from test_batched_gather import _setup, _input, _gpu_available
from psana.gpu import gpu_detector as gd
from psana.gpu.gpu_budget import _GpuBudget
from psana.gpu.gpu_stream import EventPool
from psana.gpu.gpu_kvikio_read import (
    DESC_STREAM_ID, DESC_EVENT_INDEX, DESC_DEVICE_OFFSET, DESC_READ_SIZE,
)
from psana.gpu.gpudgram import parser as p

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not _gpu_available(), reason='no CUDA device available')]


def fixture_inputs(cp, passthrough=False):
    budget = _GpuBudget(16 * 1024**2)
    parser, detector, dtype = _setup(cp, passthrough, budget, parser_slots=3)
    producer = cp.cuda.Stream(non_blocking=True)
    selections = [(1, 0), (0,), (1, 0), (1,)]
    batch, events, expected = _input(cp, parser, producer, selections, dtype)
    producer.synchronize()
    data, table = batch.data_gpu.get(), batch._test_descriptors.copy()
    specs = tuple(e.event for e in events)
    batch.retire()
    # A retained capacity larger than the row count must not become the stride
    # for another input owner. Every raw buffer also has an independent base.
    parser._slots[0].batched_locator_rows(len(parser.field_handles), 17)

    def owner(stream_ids, indices, stream):
        selected = [r.copy() for r in table if int(r[DESC_STREAM_ID]) in stream_ids
                    and int(r[DESC_EVENT_INDEX]) in {specs[i].batch_event_index for i in indices}]
        selected.reverse()  # owner-local rows differ from execution order
        packed = bytearray(32)
        for row in selected:
            start, size = int(row[DESC_DEVICE_OFFSET]), int(row[DESC_READ_SIZE])
            row[DESC_DEVICE_OFFSET] = len(packed)
            packed.extend(data[start:start + size].tobytes())
        raw_host = np.frombuffer(packed, dtype=np.uint8)
        with stream:
            raw = cp.asarray(raw_host)
        stream.synchronize()
        read = NS(data_gpu=raw, desc_table=np.asarray(selected, np.uint64),
                  retain_input=lambda: lambda: None)
        return parser.parse_window(read, stream, batch_id=7)

    return parser, detector, budget, specs, expected, owner


@pytest.mark.parametrize('passthrough', [False, True])
@pytest.mark.parametrize('resident', ['none', 'partial', 'all'])
def test_multiowner_pixels_reuse_and_constant_launch_count(monkeypatch, passthrough, resident):
    import cupy as cp
    parser, detector, budget, specs, expected, owner = fixture_inputs(cp, passthrough)
    executions = EventPool(n=1)
    producers = [cp.cuda.Stream(non_blocking=True) for _ in range(2)]
    counts = dict(walk=0, init=0, locate=0, gather=0)

    def count(module, name, label):
        original = getattr(module, name)
        def factory(*args):
            kernel = original(*args)
            def launch(*args, **kwargs):
                counts[label] += 1
                return kernel(*args, **kwargs)
            return launch
        monkeypatch.setattr(module, name, factory)
    for name, label in (('_walk_kernel', 'walk'), ('_init_locators_kernel', 'init'),
                        ('_locate_fields_kernel', 'locate')):
        count(p, name, label)
    count(gd, '_batched_gather_kernel', 'gather')
    fast = owner((0,), range(4), producers[0]) if resident != 'none' else None
    slow = owner((1,), range(4), producers[1]) if resident == 'all' else None
    retained = tuple(w for w in (fast, slow) if w is not None)
    planned = [w.acquire() for w in retained]
    for indices in ((0, 1), (2, 3)):
        left = fast if fast else owner((0,), indices, producers[0])
        right = slow if slow else owner((1,), indices, producers[1])
        assert left.batch.configured_locations().capacity != right.batch.configured_locations().capacity
        windows = (left, right)
        gv = NS(iter_events=lambda: (specs[i] for i in indices))
        record = executions.submit(gv, None, [], {'camera': (None, detector)},
                                   input_windows=windows, batch_id=7)
        for window in windows:
            assert window.batch._locators == {}  # no per-field wrapper/launch loop
            if window not in retained:
                window.close()
        assert executions.begin_retire_next() is record
        for i in indices:
            results = record.gpu_results_by_ts[specs[i].timestamp]
            if not passthrough:
                np.testing.assert_array_equal(results['camera.raw'].get(), expected[i])
            calib = expected[i].astype(np.float32)
            if not passthrough:
                present = np.any(expected[i], axis=(1, 2))
                calib[present] = (calib[present] - 7) * 2
            np.testing.assert_array_equal(results['camera.calib'].get(), calib)
        del results
        executions.finish_retire_next()
        for window in windows:
            assert window.released == (window not in retained)
    for window, use in zip(retained, planned):
        window.close()
        use.wait_until_safe_to_reuse()
        assert window.released
    ninputs = {'none': 4, 'partial': 3, 'all': 2}[resident]
    assert counts == dict(walk=ninputs, init=ninputs, locate=ninputs, gather=2)
    parser.trim_free_buffers()
    detector.trim_slot_buffers()
    assert budget.committed() == parser.memory_bytes()['config'] + detector.memory_bytes()['routing']


def test_partial_map_upload_failure_retains_owners_until_retry(monkeypatch):
    import cupy as cp
    parser, detector, budget, specs, _, owner = fixture_inputs(cp)
    producers = [cp.cuda.Stream(non_blocking=True) for _ in range(2)]
    windows = tuple(owner((i,), range(4), producers[i]) for i in range(2))
    executions = EventPool(n=1)
    real_stream = executions._streams[0]

    class RetryStream:
        fail = True
        def __getattr__(self, name):
            return getattr(real_stream, name)
        def __enter__(self):
            return real_stream.__enter__()
        def __exit__(self, *args):
            return real_stream.__exit__(*args)
        def synchronize(self):
            if self.fail:
                raise RuntimeError('unproven completion')
            real_stream.synchronize()
    stream = RetryStream()
    executions._streams[0] = stream
    prepare = gd._GatherMap.prepare
    def fail_after_upload(self, events, streams, stream, budget):
        result = prepare(self, events, streams, real_stream, budget)
        raise RuntimeError('after asynchronous map upload')
    monkeypatch.setattr(gd._GatherMap, 'prepare', fail_after_upload)
    gv = NS(iter_events=lambda: iter(specs))
    with pytest.raises(RuntimeError, match='unproven completion'):
        executions.submit(gv, None, [], {'camera': (None, detector)},
                          input_windows=windows, batch_id=7)
    assert executions.active_count == 1
    for window in windows:
        assert not window.close() and not window.released
    mapping = detector._gather_maps[0]
    assert mapping.device is not None and mapping.host is not None
    charged = budget.committed()
    stream.fail = False
    list(executions.flush())
    assert all(window.released for window in windows)
    parser.trim_free_buffers()
    detector.trim_slot_buffers()
    del mapping
    assert budget.committed() < charged
    assert budget.committed() == parser.memory_bytes()['config'] + detector.memory_bytes()['routing']
