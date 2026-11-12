"""restart must not let a new job start while the old one is still exiting.

After scancel, Slurm gives a step up to KillWait (32 s) to exit, and the job stays in
squeue (COMPLETING) until it has.  restart used to skip both stop()'s wait and start()'s
"still running" check, so the new job could open the same device (e.g. a KCU's
/dev/datadev_N, resetting its timing link) while the old process still had it open.
"""

import types

import pytest


def make_runner(main_mod):
    runner = object.__new__(main_mod.Runner)
    runner.config = {"epix100_1_0": {"cmd": "drp"}}
    runner.sbjob = {"drp-srcf-cmp044": {"epix100_1_0": {"cmd": "drp"}}}
    runner.node_features = None
    return runner


class FakeSbman:
    """Job table keyed by comment, shared between squeue lookups and scancel."""

    as_step = False

    def __init__(self, jobs):
        self.jobs = jobs  # comment -> {"job_id", "job_name", "state"}

    def get_job_info(self, strict=False):
        return dict(self.jobs)

    def get_comment(self, config_id):
        return f"x0_p0_s0_{config_id}"

    def generate(self, *args, **kwargs):
        pass


def install_slurm(monkeypatch, main_mod, sbman, polls_until_gone):
    """Fake scancel/squeue: cancelled jobs stay COMPLETING for N squeue polls."""
    state = {"polls": 0, "cancelled": set()}

    def fake_run_slurm(cmd, *args, **kwargs):
        if cmd == "scancel":
            state["cancelled"].update(args)
            return types.SimpleNamespace(returncode=0, stdout="")
        if cmd == "squeue":
            state["polls"] += 1
            if state["polls"] > polls_until_gone:
                sbman.jobs.clear()
                return types.SimpleNamespace(returncode=0, stdout="")
            lines = [f"{j['job_id']}|COMPLETING" for j in sbman.jobs.values()]
            return types.SimpleNamespace(returncode=0, stdout="\n".join(lines))
        raise AssertionError(f"unexpected slurm command {cmd}")

    monkeypatch.setattr(main_mod, "run_slurm", fake_run_slurm)
    monkeypatch.setattr(main_mod.time, "sleep", lambda s: None)
    return state


def test_restart_waits_for_old_job_before_submitting(monkeypatch):
    pytest.importorskip("typer")
    import psdaq.slurm.main as main_mod

    sbman = FakeSbman(
        {"x0_p0_s0_epix100_1_0": {"job_id": 111, "job_name": "epix100_1_0"}}
    )
    runner = make_runner(main_mod)
    runner.sbman = sbman
    runner.get_unique_prefix = lambda: "x0_p0_s0"
    state = install_slurm(monkeypatch, main_mod, sbman, polls_until_gone=3)

    submitted = []

    def fake_submit(verbose=False):
        # The old job must already be gone from squeue when the new one is submitted.
        assert sbman.jobs == {}, "submitted while the old job was still in squeue"
        submitted.append(True)
        return 0

    runner.submit = fake_submit

    runner.restart(unique_ids="epix100_1_0")

    assert state["cancelled"] == {"111"}
    assert state["polls"] == 4  # 3 COMPLETING polls, then gone
    assert submitted == [True]


def test_restart_refuses_to_start_if_old_job_outlives_the_wait(monkeypatch, capsys):
    pytest.importorskip("typer")
    import psdaq.slurm.main as main_mod

    sbman = FakeSbman(
        {"x0_p0_s0_epix100_1_0": {"job_id": 111, "job_name": "epix100_1_0"}}
    )
    runner = make_runner(main_mod)
    runner.sbman = sbman
    runner.get_unique_prefix = lambda: "x0_p0_s0"
    install_slurm(monkeypatch, main_mod, sbman, polls_until_gone=10**6)

    # Expire the wait immediately instead of sleeping for STOP_WAIT_SECONDS.
    clock = iter([0.0] + [main_mod.STOP_WAIT_SECONDS + 1.0] * 10)
    monkeypatch.setattr(main_mod.time, "monotonic", lambda: next(clock))

    def fail_submit(verbose=False):
        raise AssertionError("submit must not run while the old job is still active")

    runner.submit = fail_submit

    with pytest.raises(RuntimeError, match="running jobs using the same resources"):
        runner.restart(unique_ids="epix100_1_0")

    err = capsys.readouterr().err
    assert "still active" in err and "111 (COMPLETING)" in err


def test_stop_wait_outlasts_slurm_killwait():
    pytest.importorskip("typer")
    import psdaq.slurm.main as main_mod

    # Slurm's KillWait on the DAQ nodes is 32 s; the wait has to be longer.
    assert main_mod.STOP_WAIT_SECONDS > 32
