import pytest
from subprocess import CalledProcessError

from psdaq.slurm.utils import SbatchManager


def create_manager(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USER", "tstopr")
    return SbatchManager(
        configfilename="dummy.py",
        xpm_id=0,
        platform=4,
        station=0,
        as_step=False,
        verbose=False,
        output=str(tmp_path),
    )


def test_get_job_info_returns_empty_on_failure_by_default(tmp_path, monkeypatch):
    manager = create_manager(tmp_path, monkeypatch)

    def fail_slurm(*args, **kwargs):
        raise RuntimeError("squeue unavailable")

    monkeypatch.setattr("psdaq.slurm.utils.run_slurm_with_retries", fail_slurm)

    assert manager.get_job_info() == {}


def test_get_job_info_reraises_on_failure_when_strict(tmp_path, monkeypatch):
    manager = create_manager(tmp_path, monkeypatch)

    def fail_slurm(*args, **kwargs):
        raise RuntimeError("squeue unavailable")

    monkeypatch.setattr("psdaq.slurm.utils.run_slurm_with_retries", fail_slurm)

    with pytest.raises(RuntimeError, match="squeue unavailable"):
        manager.get_job_info(strict=True)


def test_get_job_info_strict_error_includes_slurm_stderr(tmp_path, monkeypatch):
    manager = create_manager(tmp_path, monkeypatch)

    def fail_slurm(*args, **kwargs):
        raise CalledProcessError(
            returncode=1,
            cmd=args,
            stderr=b"slurm controller unavailable",
        )

    monkeypatch.setattr("psdaq.slurm.utils.run_slurm_with_retries", fail_slurm)

    with pytest.raises(RuntimeError, match="slurm controller unavailable"):
        manager.get_job_info(strict=True)


class FailingSbatchManager:
    as_step = False

    def get_job_info(self, strict=False):
        if strict:
            raise RuntimeError("job lookup failed")
        return {}

    def get_comment(self, config_id):
        return f"x0_p4_s0_{config_id}"

    def generate(self, *args, **kwargs):
        raise AssertionError("generate should not run after strict lookup failure")


def test_runner_exists_propagates_strict_lookup_failure():
    pytest.importorskip("typer")
    from psdaq.slurm.main import Runner

    runner = object.__new__(Runner)
    runner.config = {"bld_0": {"cmd": "drp"}}
    runner.sbman = FailingSbatchManager()

    with pytest.raises(RuntimeError, match="job lookup failed"):
        runner._exists(strict=True)


def test_runner_start_does_not_submit_when_strict_lookup_fails():
    pytest.importorskip("typer")
    from psdaq.slurm.main import Runner

    runner = object.__new__(Runner)
    runner.config = {"bld_0": {"cmd": "drp"}}
    runner.sbjob = {"drp-srcf-cmp001": {"bld_0": {"cmd": "drp"}}}
    runner.node_features = None
    runner.sbman = FailingSbatchManager()

    def fail_submit():
        raise AssertionError("submit should not run after strict lookup failure")

    runner.submit = fail_submit

    with pytest.raises(RuntimeError, match="job lookup failed"):
        runner.start()
