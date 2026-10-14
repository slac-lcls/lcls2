"""Admission and retained allocation generations across multi-owner executions."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from test_batched_gather import _gpu_available
from test_multiowner_gather import fixture_inputs
from psana.gpu.gpu_allocation import backing_capacity
from psana.gpu.gpu_budget import GpuMemoryPressureError, allocation_growth_bytes
from psana.gpu.gpu_stream import EventPool

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not _gpu_available(), reason='no CUDA device available')]


@pytest.mark.parametrize('passthrough', [False, True])
def test_multiowner_growth_keeps_old_map_and_output_charged(passthrough):
    import cupy as cp
    parser, detector, budget, specs, expected, owner = fixture_inputs(cp, passthrough)
    producer = cp.cuda.Stream(non_blocking=True)
    windows = tuple(owner((i,), range(4), producer) for i in range(2))
    planned = [w.acquire() for w in windows]
    pool = EventPool(n=1)
    aliases = []
    old_pixels = None
    try:
        for count in (1, 4, 2):
            growth = allocation_growth_bytes(detector.allocation_requirements(count, 0))
            committed = budget.committed()
            if growth:
                budget._limit = committed + growth - 1
                with pytest.raises(GpuMemoryPressureError):
                    budget.hold(growth)
                assert budget.committed() == committed and budget._held == 0
            budget._limit = committed + growth
            hold = budget.hold(growth)
            try:
                with hold:
                    record = pool.submit(NS(iter_events=lambda: iter(specs[:count])),
                                         None, [], {'camera': (None, detector)},
                                         input_windows=windows, batch_id=7)
            finally:
                hold.close()
            assert budget._held == 0 and budget.committed() <= budget.limit()
            assert pool.begin_retire_next() is record
            for i in range(count):
                actual = record.gpu_results_by_ts[specs[i].timestamp]['camera.calib']
                reference = expected[i].astype(np.float32)
                if not passthrough:
                    present = np.any(expected[i], axis=(1, 2))
                    reference[present] = (reference[present] - 7) * 2
                np.testing.assert_array_equal(actual.get(), reference)
            del actual
            if count == 1:
                aliases = [detector._gather_maps[0].device[:], detector._calib_slot_bufs[0][:]]
                old_pixels = aliases[1].get()
            else:
                assert detector._gather_maps[0].device.data.ptr != aliases[0].data.ptr
                assert detector._calib_slot_bufs[0].data.ptr != aliases[1].data.ptr
                np.testing.assert_array_equal(aliases[1].get(), old_pixels)
            pool.finish_retire_next()
        for window, use in zip(windows, planned):
            window.close()
            use.wait_until_safe_to_reuse()
        parser.trim_free_buffers()
        detector.trim_slot_buffers()
        fixed = parser.memory_bytes()['config'] + detector.memory_bytes()['routing']
        assert budget.committed() == fixed + sum(backing_capacity(a) for a in aliases)
        aliases.clear()
        assert budget.committed() == fixed
    finally:
        for _ in pool.flush():
            pass
        for window, use in zip(windows, planned):
            window.close()
            use.wait_until_safe_to_reuse()
