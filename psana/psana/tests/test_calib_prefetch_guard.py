"""
Tests for the calib prefetch writer (calib_prefetch/calib_utils.py): a fetch
that returns no constants for a detector must not replace a good cache, and
the pickle records provenance. calib_constants_all_types is mocked (no network).
"""

import logging
import os
import pickle

import pytest

from psana.pscalib.app.calib_prefetch import calib_utils

DET_UID = "jungfrau_000001"
EXP = "xpptst00123"
INFO = {"jungfrau": DET_UID}
GOOD = {"pedestals": ("peds_data", {"_id": "doc_good", "ctype": "pedestals"})}


class FakeRun:
    def __init__(self, runnum, expt=EXP):
        self.runnum, self.expt = runnum, expt


@pytest.fixture
def log():
    return logging.getLogger("test_calib_prefetch_guard")


@pytest.fixture
def db(monkeypatch):
    """What the mocked calib_constants_all_types returns for each detector uid."""
    state = {DET_UID: GOOD}
    monkeypatch.setattr(calib_utils, "calib_constants_all_types", lambda det_uid, **kwa: state.get(det_uid))
    return state


def path(tmp_path):
    return os.path.join(tmp_path, calib_utils.CALIB_PICKLE_FILENAME)


def load(tmp_path):
    with open(path(tmp_path), "rb") as f:
        return pickle.load(f)


@pytest.mark.parametrize("runnum, selection", [(137, "run_range"), (0, "latest")])
def test_writes_provenance(db, tmp_path, log, runnum, selection):
    calib_utils.update_calib(FakeRun(runnum), INFO, log, str(tmp_path))
    data = load(tmp_path)
    assert data["calib_const"]["jungfrau"] == GOOD
    assert data["det_info"] == INFO
    assert (data["expt"], data["runnum"], data["selection"]) == (EXP, runnum, selection)
    assert data["doc_ids"] == {"jungfrau": {"pedestals": "doc_good"}}


@pytest.mark.parametrize("empty", [None, {}])
def test_empty_fetch_does_not_replace_good_cache(db, tmp_path, log, empty):
    calib_utils.update_calib(FakeRun(137), INFO, log, str(tmp_path))
    db[DET_UID] = empty  # e.g. DB outage or auth failure
    calib_utils.update_calib(FakeRun(0), INFO, log, str(tmp_path))
    data = load(tmp_path)
    assert data["runnum"] == 137
    assert data["calib_const"]["jungfrau"] == GOOD


def test_empty_fetch_replaces_cache_for_other_experiment(db, tmp_path, log):
    calib_utils.update_calib(FakeRun(137, expt="xpptst00999"), INFO, log, str(tmp_path))
    db[DET_UID] = None
    calib_utils.update_calib(FakeRun(0), INFO, log, str(tmp_path))
    assert load(tmp_path)["expt"] == EXP


def test_empty_fetch_replaces_cache_for_other_detector_uid(db, tmp_path, log):
    calib_utils.update_calib(FakeRun(137), INFO, log, str(tmp_path))
    db["jungfrau_000002"] = None
    calib_utils.update_calib(FakeRun(138), {"jungfrau": "jungfrau_000002"}, log, str(tmp_path))
    assert load(tmp_path)["runnum"] == 138


def test_legacy_cache_without_provenance_is_replaced(db, tmp_path, log):
    with open(path(tmp_path), "wb") as f:
        pickle.dump({"det_info": INFO, "calib_const": {"jungfrau": GOOD}}, f)
    db[DET_UID] = None
    calib_utils.update_calib(FakeRun(0), INFO, log, str(tmp_path))
    data = load(tmp_path)
    assert data["expt"] == EXP and not data["calib_const"]["jungfrau"]
