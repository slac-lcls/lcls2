"""Device-side owner/row bounds and terminal consumers for pointer-table gather."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from test_batched_gather import _gpu_available
from test_multiowner_gather import fixture_inputs
from psana.gpu import gpu_detector as gd
from psana.gpu.gpu_input import GpuEventDgrams
from psana.gpu.gpu_stream import EventPool

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not _gpu_available(), reason='no CUDA device available')]


@pytest.mark.parametrize('damage', ['owner', 'row', 'capacity', 'raw_bytes'])
def test_invalid_owner_or_local_bounds_zero_only_that_segment(damage):
    import cupy as cp
    parser, detector, _, specs, expected, owner = fixture_inputs(cp)
    stream = cp.cuda.Stream(non_blocking=True)
    windows = tuple(owner((i,), range(4), stream) for i in range(2))
    events = GpuEventDgrams.from_windows(NS(iter_events=lambda: iter(specs)), windows, batch_id=7)
    mapping = gd._GatherMap()
    with stream:
        inputs = mapping.prepare(events, detector._gather_plan.streams, stream, None)
        raw = cp.empty((len(specs) * 3, 3, 100), cp.uint16)
        present = cp.empty((len(specs), 3), cp.uint8)
    stream.synchronize()
    stream_column = detector._gather_plan.streams.index(0)
    rows = inputs.rows.get().reshape(len(specs), 2, 2)
    index = int(rows[0, stream_column, 0])
    with stream:
        if damage == 'owner':
            inputs.rows[2 * stream_column] = len(inputs.locations)
        elif damage == 'row':
            inputs.rows[2 * stream_column + 1] = inputs.locations[index].owner.n_dgrams
        else:
            column = 3 if damage == 'capacity' else 1
            inputs.owners[index * gd._GATHER_OWNER_WORDS + column] = 0
        detector._gather_plan.gather(inputs, raw, present, 300, stream)
    stream.synchronize()
    observed = raw.get().reshape(len(specs), 3, 3, 100)
    reference = np.stack(expected)
    reference[0, 1] = 0
    if damage in ('capacity', 'raw_bytes'):
        reference[:, 1] = 0
    np.testing.assert_array_equal(observed, reference)
    for window in windows:
        assert window.close()


def test_two_delayed_input_consumers_keep_both_gather_owners():
    import cupy as cp
    _, detector, _, specs, _, owner = fixture_inputs(cp)
    producer = cp.cuda.Stream(non_blocking=True)
    windows = tuple(owner((i,), range(4), producer) for i in range(2))
    pool = EventPool(n=1)
    record = pool.submit(NS(iter_events=lambda: iter(specs)), None, [],
                         {'camera': (None, detector)}, input_windows=windows, batch_id=7)
    assert pool.begin_retire_next() is record
    delay = cp.RawKernel('''extern "C" __global__ void delay(unsigned long long ticks) {
        unsigned long long start = clock64();
        while (clock64() - start < ticks) {}
    }''', 'delay')
    delay.compile()
    consumers = [cp.cuda.Stream(non_blocking=True) for _ in range(2)]
    copies, references, completions = [], [], []
    lease = record.input_leases_by_ts[specs[0].timestamp]
    for i, (window, stream) in enumerate(zip(windows, consumers)):
        references.append(window.batch.data_gpu.get())
        with stream:
            stream.wait_event(lease.result_ready)
            delay((1,), (1,), (np.uint64(60000000 if i == 0 else 1000000),))
            copies.append(window.batch.data_gpu.copy())
            done = cp.cuda.Event(disable_timing=True)
            done.record(stream)
        lease.register_consumer_done(done)
        completions.append(done)
        assert not window.close()
    pool.finish_retire_next()
    assert all(done.done for done in completions)
    assert all(window.released for window in windows)
    for actual, expected in zip(copies, references):
        np.testing.assert_array_equal(actual.get(), expected)
