"""Raw adapter setup requires Configure, not calibration or a CUDA context."""
import sys

import numpy as np
import pytest

from psana.gpu.gpu_detector import DenseInputPreparer, _GATHER_MAP_BYTES_PER_ENTRY
from psana.gpu.gpu_input import GpuDetectorBinding
from psana.gpu.gpudgram.config import GpuStreamConfigTable


def raw_binding(rank=3, dtype=1, det_type='jungfrau', alg='raw', field='raw'):
    configs = GpuStreamConfigTable({i: [dict(
        det_name='jf', det_type=det_type, det_id='jf', segment=segment,
        alg_name=alg, alg_version=(0, 2, 0), names_id_value=10,
        fields=[dict(name=field, type=dtype, element_size=2, rank=rank,
                     field_index=0, shape_index=0)],
    )] for i, segment in enumerate((4, 9))})
    binding = GpuDetectorBinding('jf', canonical_segment_ids=(9, 4),
                                field_handles_by_segment={
                                    s: configs.resolve('jf', s, alg, field) for s in (9, 4)})
    return configs, binding


@pytest.mark.parametrize('rank', [2, 3])
def test_jungfrau_raw_setup_without_cuda_or_calibration(monkeypatch, rank):
    monkeypatch.setitem(sys.modules, 'cupy', None)
    configs, binding = raw_binding(rank=rank)
    raw = DenseInputPreparer.jungfrau_raw(configs, binding, n_slots=2)
    assert raw.det_shape == (2, 512, 1024)
    assert raw.canonical_segment_ids == (9, 4)
    assert raw.memory_bytes()['total'] == 0
    assert raw.prepare_batch([]) is None
    assert not any(hasattr(raw, attr) for attr in (
        'peds_gpu', 'gmask_gpu', '_calib_slot_bufs', '_scatter_ix', 'beginstep'))
    assert raw.estimate_subbatch_bytes(3) == 3 * (
        2 * 512 * 1024 * 2 + 2 + 2 * _GATHER_MAP_BYTES_PER_ENTRY)


@pytest.mark.parametrize('options', [dict(rank=1), dict(dtype=3),
                                    dict(det_type='epix'), dict(alg='fex'),
                                    dict(field='calibrated')])
def test_jungfrau_raw_rejects_unsupported_configure(options):
    configs, binding = raw_binding(**options)
    with pytest.raises(ValueError, match='unsupported'):
        DenseInputPreparer.jungfrau_raw(configs, binding)


@pytest.mark.parametrize('shape,slots', [((2, 0, 5), 1), ((2, 5), 1),
                                       ((2, 5, 5), 0), ((2, 5, 5), 1.5)])
def test_dense_preparation_rejects_invalid_storage_contract(shape, slots):
    _, binding = raw_binding()
    with pytest.raises(ValueError):
        DenseInputPreparer(shape, binding, n_slots=slots)
