import pytest
from psana import DataSource
from setup_input_files import setup_input_files

@pytest.mark.parametrize("case, setup_dsparms", [
    ("no calibconst attr", lambda ds: setattr(ds, 'dsparms', type('D', (), {})())),
    ("calibconst is None", lambda ds: setattr(ds, 'dsparms', type('D', (), {'calibconst': None})())),
    ("missing det_name", lambda ds: setattr(ds, 'dsparms', type('D', (), {'calibconst': {}})())),
])
def test_check_empty_calibconst_with_datasource(tmp_path, case, setup_dsparms):
    # Prepare input files
    setup_input_files(tmp_path)
    xtc_dir = tmp_path / ".tmp"

    # Load the real DataSource and create a Run object
    ds = DataSource(exp='xpptut15', run=14, dir=str(xtc_dir), batch_size=1)
    run = next(ds.runs())

    # Apply test-specific manipulation to dsparms
    setup_dsparms(run)

    det_name = "fake_detector"
    run._check_empty_calibconst(det_name)

    # Assert expectations
    assert hasattr(run.dsparms, "calibconst"), f"{case}: calibconst attribute missing"
    assert det_name in run.dsparms.calibconst, f"{case}: detector not in calibconst"
    assert run.dsparms.calibconst[det_name] is None, f"{case}: detector value is not None"
