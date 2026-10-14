"""Stage 2 exact constant uploads and serial task setup on a real device."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from test_gpu_allocation_device import available
from psana.gpu import GpuTask
from psana.gpu.gpu_task import RequestedConstants
from psana.gpu.gpu_budget import _GpuBudget, GpuMemoryPressureError

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not available(), reason='no CUDA device')]


def unused(evt, stream):
    raise AssertionError('callback dispatch belongs to Stage 3')


@pytest.mark.parametrize('value', [np.array(3, np.uint32), np.empty((0, 3), np.float32),
    np.arange(60).reshape(6, 10)[::2, ::2],
    np.arange(60, dtype=np.float64).reshape(3, 10, 2), np.array([1, 2], np.float16),
    np.array([True, False]), np.array([1+2j], np.complex128)])
def test_exact_gain_only_upload_and_retained_alias(value):
    import cupy as cp
    budget = _GpuBudget(4096)
    constants = RequestedConstants([('jf', 'pixel_gain')], budget)
    source = {'jf': {'pixel_gain': (value, {})}}
    assert constants.refresh(source)
    array = constants.get('jf', 'pixel_gain')
    assert array.shape == value.shape and array.dtype == value.dtype
    np.testing.assert_array_equal(cp.asnumpy(array), value)
    assert not constants.refresh(source)
    charged = budget.committed()
    constants.close()
    assert budget.committed() == charged
    del array
    assert budget.committed() == 0


def test_failed_replacement_keeps_previous_device_generation():
    import cupy as cp
    budget = _GpuBudget(512)
    constants = RequestedConstants([('jf', 'gain')], budget)
    constants.refresh({'jf': {'gain': np.array(1.)}})
    with pytest.raises(GpuMemoryPressureError):
        constants.refresh({'jf': {'gain': np.array(2.)}})
    assert cp.asnumpy(constants.get('jf', 'gain')) == 1
    assert budget.committed() == 512 and budget._held == 0
    constants.close()
    assert budget.committed() == 0


def test_serial_datasource_stages_dense_inputs_and_exact_constants(monkeypatch):
    import cupy as cp
    from psana import DataSource
    from psana.psexp.run import Run
    gain = np.arange(60, dtype=np.float64).reshape(3, 10, 2)
    def setup(run):
        run.dsparms.calibconst = {'jungfrau': {'pixel_gain': (gain, {})}}
    monkeypatch.setattr(Run, '_setup_run_calibconst', setup)
    task = GpuTask(unused, ['jungfrau.raw'], [('jungfrau', 'pixel_gain')])
    ds = DataSource(exp='mfx100848724', run=51, dir='/sdf/data/lcls/ds/prj/public01/xtc',
                    detectors=['jungfrau'], max_events=1, gpu_det='jungfrau', gpu_fn=task)
    run = next(ds.runs())
    manager = run._evt_iter
    try:
        assert manager._gpu_task is task and ds.dsparms.batch_size == 1
        assert set(manager.input_preparers) == {'jungfrau.raw'}
        np.testing.assert_array_equal(cp.asnumpy(manager._task_constants.get('jungfrau', 'pixel_gain')), gain)
    finally:
        manager.close()
