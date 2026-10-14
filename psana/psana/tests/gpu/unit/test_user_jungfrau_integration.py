"""CPU bin preparation, configuration and standalone import checks."""
from pathlib import Path
import inspect
import subprocess
import sys
import numpy as np
import pytest
from psana.gpu.examples.jungfrau_azimuthal_integration import JungfrauAzimuthalIntegration, radial_bin_ids


def test_radial_edges_center_nonfinite_and_exclusion():
    x = np.array([[[1, 2, 3, 4, np.nan, np.inf]]])
    bins = radial_bin_ids(x, np.zeros_like(x), [0, 1, 2, 3], center_mm=(1, 0))
    np.testing.assert_array_equal(bins, [[[0, 1, 2, -1, -1, -1]]])


@pytest.mark.parametrize('edges,center', [([0,0], (0,0)), ([1,0], (0,0)),
    ([0,np.inf], (0,0)), ([0,1], (np.nan,0)), ([0,1], (0,))])
def test_reject_invalid_radial_policy(edges, center):
    with pytest.raises(ValueError):
        radial_bin_ids(np.zeros((1,2,3)), np.zeros((1,2,3)), edges, center_mm=center)


@pytest.mark.parametrize('bins,n', [(np.zeros((1,2,3)), 2), (np.zeros((1,2,3),np.int32),0),
    (np.zeros((1,2,3),np.int32),True), (np.full((1,2,3),2),2),
    (np.full((1,2,3),-2),2), (np.zeros((1,2),np.int32),2)])
def test_reject_invalid_tables(bins,n):
    with pytest.raises(ValueError): JungfrauAzimuthalIntegration(bins,n)


def test_sparse_reordered_tables_and_immutable_copy():
    bins = np.array([[[0,1]], [[1,-1]], [[0,0]], [[1,0]]], np.int32)
    user = JungfrauAzimuthalIntegration(bins, 3)
    bins.fill(-1)
    order, offsets = user._host_table((3,1), (1,2))
    np.testing.assert_array_equal(order, [1,0,2])
    np.testing.assert_array_equal(offsets, [0,1,3,3])
    assert not user.bin_ids.flags.writeable
    with pytest.raises(ValueError): user._host_table((4,), (1,2))
    with pytest.raises(ValueError): user._host_table((1,1), (1,2))
    with pytest.raises(ValueError): user._host_table((1,), (2,1))


def test_external_declaration_without_psana_or_cuda(tmp_path):
    folder = Path(inspect.getfile(JungfrauAzimuthalIntegration)).parent
    for name in ('jungfrau_calibration.py','jungfrau_azimuthal_integration.py'):
        (tmp_path/name).write_bytes((folder/name).read_bytes())
    subprocess.run([sys.executable, '-c', '''
import sys
sys.modules['psana'] = None
sys.modules['cupy'] = None
import numpy as np
from jungfrau_azimuthal_integration import JungfrauAzimuthalIntegration
u = JungfrauAzimuthalIntegration(np.zeros((1,2,3),np.int32), 2)
assert u.inputs == ('jungfrau.raw',) and u.calls == 0
'''], cwd=tmp_path, check=True)
