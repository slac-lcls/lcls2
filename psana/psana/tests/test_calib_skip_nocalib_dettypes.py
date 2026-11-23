"""
Detectors whose types never use calibration constants (scalar PVA PVs, BLD,
timing, ...) must not be looked up in the calib DB at BeginRun.
Uses a mocked calib_constants_all_types (no network).
"""

from types import SimpleNamespace

import pytest

import psana.pscalib.calib.MDBWebUtils as wu
from psana.psexp.run import Run


def make_run(dettypes, skip_calib_load=None):
    configinfo_dict = {name: SimpleNamespace(dettype=dt, uniqueid=f"{dt}_") for name, dt in dettypes.items()}
    dsparms = SimpleNamespace(
        use_calib_cache=False,
        configinfo_dict=configinfo_dict,
        dbsuffix="",
        calibconst={},
    )
    if skip_calib_load is not None:
        dsparms.skip_calib_load = skip_calib_load
    run = Run.__new__(Run)  # skip __init__: no xtc needed
    run.expt, run.runnum = "txix00000", 1
    run.dsparms = dsparms
    run._calib_const = None
    run.logger = SimpleNamespace(debug=lambda *a, **k: None, info=lambda *a, **k: None)
    return run


@pytest.fixture
def fetched(monkeypatch):
    calls = []

    def fake(det_uniqueid, **kwargs):
        calls.append(det_uniqueid)
        return {"pedestals": (None, {})}

    monkeypatch.setattr(wu, "calib_constants_all_types", fake)
    return calls


def test_nocalib_dettypes_are_not_fetched(fetched):
    run = make_run(
        {
            "bpms_unds_2390_x": "pv",
            "pvdetinfo_pvs": "pvdetinfo",
            "epics": "epics",
            "timing": "ts",
            "ebeam": "ebeam",
            "gmd": "gmd",
            "rix_fim0": "wave8",
            "jungfrau": "jungfrau",
            "alvium3rix1": "epics2d",
        }
    )
    run._setup_run_calibconst()

    assert sorted(fetched) == ["epics2d_", "jungfrau_"]
    assert run.dsparms.calibconst["bpms_unds_2390_x"] == {}
    assert run.dsparms.calibconst["timing"] == {}
    assert run.dsparms.calibconst["jungfrau"] == {"pedestals": (None, {})}
    assert run.dsparms.calibconst["alvium3rix1"] == {"pedestals": (None, {})}


def test_skip_calib_load_still_applies(fetched):
    run = make_run({"jungfrau": "jungfrau", "pv1": "pv"}, skip_calib_load=["jungfrau"])
    run._setup_run_calibconst()
    assert fetched == []
    assert run.dsparms.calibconst == {"jungfrau": {}, "pv1": {}}


@pytest.mark.parametrize("dettype", ["jungfrau", "epix100", "epixm320", "archon", "axis", "epics2d", "opal", "piranha4", "ttdet", "xtcav", ""])
def test_calib_dettypes_are_fetched(dettype):
    assert wu.dettype_uses_calib(dettype)
