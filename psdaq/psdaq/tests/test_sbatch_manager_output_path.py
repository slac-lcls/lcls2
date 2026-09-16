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


def test_default_log_root_uses_daq_logs_for_any_user(tmp_path, monkeypatch):
    for user in ("xppopr", "tmoopr", "mfxopr", "cpo"):
        home = tmp_path / user
        home.mkdir()
        manager = create_manager(home, monkeypatch, user=user)

        assert Path(manager.output_path) == home / "daq" / "logs" / "2026" / "02"
        assert Path(manager.output_path).is_dir()


def test_sbatch_script_uses_default_log_root(tmp_path, monkeypatch):
    manager = create_manager(tmp_path, monkeypatch, user="mfxopr")
    manager.generate(
        node="mfx-daq",
        job_name="control",
        details={"cmd": "echo ready", "comment": "test"},
        node_features=None,
    )
    expected_log = (
        tmp_path / "daq" / "logs" / "2026" / "02" / "26_08:00:00_mfx-daq:control.log"
    )

    assert f"#SBATCH --output={expected_log}\n" in manager.sb_script


def test_output_arg_overrides_default_root(tmp_path, monkeypatch):
    custom_root = tmp_path / "my_output_root"
    manager = create_manager(
        tmp_path, monkeypatch, user="xppopr", output=str(custom_root)
    )
    expected = custom_root / "2026" / "02"

    assert Path(manager.output_path) == expected
    assert expected.exists()
    assert not (tmp_path / "daq" / "logs").exists()
