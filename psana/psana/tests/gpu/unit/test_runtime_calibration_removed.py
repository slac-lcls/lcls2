"""GPU input setup must not restore built-in calibration or MPI derived work."""
from types import SimpleNamespace as NS

import pytest

from psana.psexp.ds_base import DataSourceBase, DsParms
from psana.psexp.mpi_ds import RunParallel


@pytest.mark.parametrize('value', [1, 8, -1])
def test_image_d2h_option_is_retired_before_run_setup(value):
    with pytest.raises(ValueError, match='retired'):
        DataSourceBase.__init__(NS(), gpu_d2h_chunk_size=value)


def test_bare_callback_requires_a_task_declaration():
    with pytest.raises(TypeError, match='GpuTask'):
        DataSourceBase.__init__(NS(), gpu_fn=lambda *args: None)


@pytest.mark.parametrize('exclusive,expected', [
    (None, ['cpu', 'hybrid', 'only_gpu']),
    ('only_gpu', ['cpu', 'hybrid']),
    (['only_gpu', 'cpu'], ['hybrid']),
])
def test_shared_cache_targets_filter_exclusive_detectors_on_every_rank(exclusive, expected):
    detector = type('jungfrau_raw', (), {'__module__': 'psana.detector.jungfrau'})
    parameters = NS(
        gpu_det=exclusive, hybrid_det=['hybrid'],
        _detector_names=DsParms._detector_names,
        det_classes={'normal': {(name, 'raw'): detector
                               for name in ('only_gpu', 'cpu', 'hybrid')}},
        configinfo_dict=dict.fromkeys(('only_gpu', 'cpu', 'hybrid'), object()),
        calibconst={name: {'pedestals': object()} for name in ('only_gpu', 'cpu', 'hybrid')},
    )
    # Filtering depends on the shared detector selection, never the local role.
    targets = RunParallel._iter_jungfrau_raw(NS(dsparms=parameters))
    assert [t[0] for t in targets] == expected
