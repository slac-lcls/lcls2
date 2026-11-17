"""
Regression tests for github issue #316: non-recording (run 0) calibration
lookup must select the latest constants, not stale constants whose validity
range happens to include run 0. Uses mocked DB documents (no network).
"""

import datetime
import logging
import os
import pickle

import pytest
from bson import ObjectId

import psana.pscalib.calib.MDBWebUtils as wu
from psana.pscalib.app.calib_prefetch import calib_utils

DET = "jungfrau_000001"
EXP = "xpptst00123"


def mkdoc(gen_time, run, run_end, ctype="pedestals", tag=None):
    """Document with an _id generated at gen_time (sec), so insertion order is controlled."""
    return {
        "_id": str(ObjectId.from_datetime(datetime.datetime.fromtimestamp(gen_time, datetime.timezone.utc))),
        "ctype": ctype,
        "run": run,
        "run_end": run_end,
        "detector": DET,
        "id_data": tag or f"{ctype}-{run}-{run_end}",
    }


OLD = mkdoc(1_700_000_000, 0, "end", tag="old")
NEW_OPEN = mkdoc(1_750_000_000, 100, "end", tag="new_open")
NEW_FINITE = mkdoc(1_750_000_000, 100, 200, tag="new_finite")
DETDB_PEDS = mkdoc(1_760_000_000, 0, "end", tag="detdb_peds")
DETDB_GAIN = mkdoc(1_760_000_000, 0, "end", ctype="pixel_gain", tag="detdb_gain")


@pytest.fixture
def mock_db(monkeypatch):
    """Patch find_docs/get_data_for_doc: experiment DB and detector DB contents are configurable."""
    db = {"exp": [], "det": []}

    def find_docs(dbname, colname, query={}, **kwa):
        docs = db["exp"] if dbname == f"cdb_{EXP}" else db["det"]
        runq = query["run"]["$lte"]
        # mimic the server-side query: copy so selectors' annotations do not leak
        return [dict(d) for d in docs if d["run"] <= runq] or None

    monkeypatch.setattr(wu, "pro_detector_name", lambda detname, **kwa: detname)  # avoids DB lookup
    monkeypatch.setattr(wu, "find_docs", find_docs)
    monkeypatch.setattr(wu, "get_data_for_doc", lambda dbname, doc, **kwa: doc["id_data"])
    return db


def selected(resp, ctype="pedestals"):
    return resp[ctype][0]


@pytest.mark.parametrize("newer", [NEW_OPEN, NEW_FINITE], ids=["open_ended", "finite_ended"])
def test_run0_selects_latest(mock_db, newer):
    mock_db["exp"] = [OLD, newer]
    resp = wu.calib_constants_all_types(DET, exp=EXP, run=0)
    assert selected(resp) == newer["id_data"]


@pytest.mark.parametrize("newer", [NEW_OPEN, NEW_FINITE], ids=["open_ended", "finite_ended"])
def test_run_none_selects_latest(newer):
    # calib_constants_all_types(run=None) without time_sec/vers is rejected by
    # dbnames_collection_query, so exercise the selection contract directly.
    docs = [dict(OLD), dict(newer)]
    assert wu.select_doc_for_run(docs, None, {})["id_data"] == newer["id_data"]


def test_positive_run_range_selection_unchanged(mock_db):
    mock_db["exp"] = [OLD, NEW_FINITE]
    assert selected(wu.calib_constants_all_types(DET, exp=EXP, run=137)) == "new_finite"
    assert selected(wu.calib_constants_all_types(DET, exp=EXP, run=50)) == "old"
    # run 300 is outside new_finite's range -> falls back to the old open-ended doc
    assert selected(wu.calib_constants_all_types(DET, exp=EXP, run=300)) == "old"


def test_recording_nonrecording_recording_consistent(mock_db):
    mock_db["exp"] = [OLD, NEW_OPEN]
    picks = [selected(wu.calib_constants_all_types(DET, exp=EXP, run=r)) for r in (137, 0, 138)]
    assert picks == ["new_open"] * 3


def test_run0_detdb_fallback_fills_missing_types(mock_db):
    mock_db["exp"] = []
    mock_db["det"] = [DETDB_PEDS, DETDB_GAIN]
    resp = wu.calib_constants_all_types(DET, exp=EXP, run=0)
    assert selected(resp) == "detdb_peds"
    assert selected(resp, "pixel_gain") == "detdb_gain"


def test_exp_constants_kept_when_detdb_empty(mock_db):
    mock_db["exp"] = [NEW_OPEN]
    mock_db["det"] = []
    resp = wu.calib_constants_all_types(DET, exp=EXP, run=0)
    assert resp is not None and selected(resp) == "new_open"


# ---- prefetch writer ---------------------------------------------------------


class FakeRun:
    def __init__(self, runnum, expt=EXP):
        self.runnum, self.expt = runnum, expt


@pytest.fixture
def log():
    return logging.getLogger("test_calib_run0_selection")


def load(tmp_path):
    with open(os.path.join(tmp_path, calib_utils.CALIB_PICKLE_FILENAME), "rb") as f:
        return pickle.load(f)


def test_prefetch_run0_writes_latest_with_provenance(mock_db, tmp_path, log, monkeypatch):
    monkeypatch.setattr(calib_utils, "calib_constants_all_types", wu.calib_constants_all_types)
    mock_db["exp"] = [OLD, NEW_OPEN]
    info = {"jungfrau": DET}
    for runnum in (137, 0):
        calib_utils.update_calib(FakeRun(runnum), info, log, str(tmp_path))
        data = load(tmp_path)
        assert data["calib_const"]["jungfrau"]["pedestals"][0] == "new_open"
        assert data["runnum"] == runnum and data["expt"] == EXP
        assert data["doc_ids"]["jungfrau"]["pedestals"] == NEW_OPEN["_id"]
    assert data["selection"] == "latest"


def test_prefetch_does_not_replace_good_cache_with_empty(mock_db, tmp_path, log, monkeypatch):
    monkeypatch.setattr(calib_utils, "calib_constants_all_types", wu.calib_constants_all_types)
    mock_db["exp"] = [NEW_OPEN]
    info = {"jungfrau": DET}
    calib_utils.update_calib(FakeRun(137), info, log, str(tmp_path))
    # DB now returns nothing (e.g. outage): keep the good cache
    mock_db["exp"] = []
    calib_utils.update_calib(FakeRun(0), info, log, str(tmp_path))
    data = load(tmp_path)
    assert data["runnum"] == 137
    assert data["calib_const"]["jungfrau"]["pedestals"][0] == "new_open"


def test_prefetch_replaces_legacy_cache(mock_db, tmp_path, log, monkeypatch):
    """Legacy (pre-provenance) caches may be poisoned and are not protected."""
    monkeypatch.setattr(calib_utils, "calib_constants_all_types", wu.calib_constants_all_types)
    info = {"jungfrau": DET}
    with open(os.path.join(tmp_path, calib_utils.CALIB_PICKLE_FILENAME), "wb") as f:
        pickle.dump({"det_info": info, "calib_const": {"jungfrau": {"pedestals": ("old", OLD)}}}, f)
    mock_db["exp"] = [OLD, NEW_OPEN]
    calib_utils.update_calib(FakeRun(0), info, log, str(tmp_path))
    assert load(tmp_path)["calib_const"]["jungfrau"]["pedestals"][0] == "new_open"
