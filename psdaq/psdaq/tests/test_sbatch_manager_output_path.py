from datetime import datetime
from pathlib import Path

import psdaq.slurm.utils as slurm_utils
from psdaq.slurm.utils import SbatchManager


class FixedDatetime(datetime):
    @classmethod
    def now(cls):
        return cls(2026, 2, 26, 8, 0, 0)


def create_manager(tmp_path, monkeypatch, user, output=None):
    monkeypatch.setattr(slurm_utils, "datetime", FixedDatetime)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USER", user)
    return SbatchManager(
        configfilename="dummy.py",
        xpm_id=99,
        platform=0,
        station=0,
        as_step=False,
        verbose=False,
        output=output,
    )


def test_xpp_default_output_root_uses_daq_logs(tmp_path, monkeypatch):
    manager = create_manager(tmp_path, monkeypatch, user="xppopr")
    expected = Path(tmp_path) / "daq" / "logs" / "2026" / "02"

    assert Path(manager.output_path) == expected
    assert expected.exists()


def test_non_xpp_default_output_root_uses_home(tmp_path, monkeypatch):
    manager = create_manager(tmp_path, monkeypatch, user="mfxopr")
    expected = Path(tmp_path) / "2026" / "02"

    assert Path(manager.output_path) == expected
    assert expected.exists()


def test_output_arg_overrides_default_root(tmp_path, monkeypatch):
    custom_root = tmp_path / "my_output_root"
    manager = create_manager(
        tmp_path, monkeypatch, user="xppopr", output=str(custom_root)
    )
    expected = custom_root / "2026" / "02"

    assert Path(manager.output_path) == expected
    assert expected.exists()
