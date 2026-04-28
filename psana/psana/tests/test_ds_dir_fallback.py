import os
from pathlib import Path

import pytest

from psana.psexp.ds_base import DataSourceBase, InvalidDataSourceArgument


class DummyDataSource(DataSourceBase):
    def runs(self):
        return iter(())

    def is_mpi(self):
        return False


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_gpu_detectors_plural_kwarg_sets_dsparms():
    ds = DummyDataSource(gpu_detectors=["jungfrau"])

    assert ds.gpu_detectors == ("jungfrau",)
    assert ds.gpu_queue_depth == 2
    assert ds.dsparms.gpu_detectors == ("jungfrau",)
    assert ds.dsparms.gpu_queue_depth == 2


def test_gpu_detectors_accepts_string_value():
    ds = DummyDataSource(gpu_detectors=" Jungfrau ", gpu_queue_depth=4)

    assert ds.gpu_detectors == ("jungfrau",)
    assert ds.dsparms.gpu_detectors == ("jungfrau",)
    assert ds.dsparms.gpu_queue_depth == 4


def test_gpu_detectors_rejects_multiple_detectors_for_phase1():
    with pytest.raises(InvalidDataSourceArgument, match="supports one GPU detector"):
        DummyDataSource(gpu_detectors=["jungfrau", "epixuhr"])


def test_gpu_detectors_rejects_unsupported_detector():
    with pytest.raises(InvalidDataSourceArgument, match="only supports 'jungfrau'"):
        DummyDataSource(gpu_detectors=["epixuhr"])


def test_explicit_dir_falls_back_to_on_disk_streams_when_db_list_mismatches(tmp_path, monkeypatch):
    xtc_dir = Path(tmp_path)
    smd_dir = xtc_dir / "smalldata"

    # Only a subset of streams exists on disk, similar to extracted regression data.
    touch(xtc_dir / "mfx100848724-r0054-s003-c000.xtc2")
    touch(xtc_dir / "mfx100848724-r0054-s005-c000.xtc2")
    touch(smd_dir / "mfx100848724-r0054-s003-c000.smd.xtc2")
    touch(smd_dir / "mfx100848724-r0054-s005-c000.smd.xtc2")

    ds = DummyDataSource(exp="mfx100848724", run=54, dir=str(xtc_dir))
    ds._setup_runnum_list()

    def fake_db_file_info(runnum):
        assert runnum == 54
        return {
            "xtc_files": [
                "mfx100848724-r0054-s000-c000.xtc2",
                "mfx100848724-r0054-s003-c000.xtc2",
                "mfx100848724-r0054-s005-c000.xtc2",
            ],
            "dirname": "/mfx/mfx100848724/xtc",
        }

    monkeypatch.setattr(ds, "_get_file_info_from_db", fake_db_file_info)

    ds._setup_run_files(54)

    assert [os.path.basename(p) for p in ds.smd_files] == [
        "mfx100848724-r0054-s003-c000.smd.xtc2",
        "mfx100848724-r0054-s005-c000.smd.xtc2",
    ]
    assert [os.path.basename(p) for p in ds.xtc_files] == [
        "mfx100848724-r0054-s003-c000.xtc2",
        "mfx100848724-r0054-s005-c000.xtc2",
    ]


def test_missing_xtc_files_error_non_live(tmp_path):
    ds = DummyDataSource(exp="mfx100848724", run=54, dir=str(tmp_path))
    ds._setup_runnum_list()

    err = ds._missing_xtc_files_error()

    assert str(err) == (
        "No XTC files found for exp=mfx100848724 run=54 in dir=%s. "
        "Checked for both final and .inprogress filenames."
    ) % str(tmp_path)


def test_missing_xtc_files_error_live(tmp_path, monkeypatch):
    monkeypatch.setenv("PS_R_MAX_RETRIES", "60")
    ds = DummyDataSource(exp="mfx100848724", run=54, dir=str(tmp_path), live=True)
    ds._setup_runnum_list()

    err = ds._missing_xtc_files_error()

    assert str(err) == (
        "Timed out waiting for XTC files for exp=mfx100848724 run=54 in dir=%s (timeout=60s). "
        "Checked for both final and .inprogress filenames."
    ) % str(tmp_path)
