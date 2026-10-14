"""Metadata rejection and CUDA-free declaration for the external user example."""
import ast
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from psana.gpu.examples.jungfrau_calibration import JungfrauCalibration


def test_copy_import_and_declaration_need_neither_psana_nor_cupy(tmp_path):
    import inspect
    source = Path(inspect.getfile(JungfrauCalibration))
    (tmp_path/'jungfrau_calibration.py').write_text(source.read_text())
    subprocess.run([sys.executable, '-c', '''
import sys
sys.modules['psana'] = None
sys.modules['cupy'] = None
from jungfrau_calibration import JungfrauCalibration
user = JungfrauCalibration('jf', use_offset=True, status_bits=0xffff, stextra_bits=1 << 40)
assert user.inputs == ('jf.raw',)
assert user.calibconst == tuple(('jf', k) for k in
    ('pedestals', 'pixel_gain', 'pixel_offset', 'pixel_status', 'status_extra'))
assert user.calls == user.events == 0
'''], cwd=tmp_path, check=True)


@pytest.mark.parametrize('options,error', [
    ({'status_bits': True}, TypeError), ({'status_bits': 1.5}, TypeError),
    ({'status_bits': -1}, ValueError), ({'status_bits': 1 << 64}, ValueError),
    ({'stextra_bits': True}, TypeError), ({'stextra_bits': -1}, ValueError),
    ({'stextra_bits': 1 << 64}, ValueError), ({'use_offset': 1}, TypeError), ({'detector': ''}, ValueError),
    ({'output': 'jungfrau.raw'}, ValueError),
])
def test_reject_bad_user_configuration(options, error):
    with pytest.raises(error):
        JungfrauCalibration(**options)


@pytest.mark.parametrize('bad', ['duplicate', 'negative', 'bounds', 'shape', 'dtype', 'presence', 'strides'])
def test_reject_unsafe_layout_before_launch(bad):
    user = JungfrauCalibration()
    raw = np.zeros((2, 2, 3, 4), np.uint16)
    present = np.ones((2, 2), np.uint8)
    constants = {k: np.ones((3, 4, 3, 4), np.float32) for _, k in user.calibconst}
    segments = (3, 1)
    if bad == 'duplicate': segments = (1, 1)
    elif bad == 'negative': segments = (-1, 1)
    elif bad == 'bounds': segments = (4, 1)
    elif bad == 'shape': constants['pixel_gain'] = np.ones((3, 2, 3, 4), np.float32)
    elif bad == 'dtype': constants['pixel_gain'] = constants['pixel_gain'].astype(np.int32)
    elif bad == 'presence': present = np.ones((2, 2), bool)
    else: raw = raw[..., ::-1]
    with pytest.raises((TypeError, ValueError)):
        user._layout(raw, present, segments, constants, 2)


def test_user_module_imports_no_psana_internals():
    import inspect
    tree = ast.parse(Path(inspect.getfile(JungfrauCalibration)).read_text())
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not any(n == 'psana' or n.startswith('psana.') for n in imports)
