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


def test_jobstep_command_dumps_batch_and_step_env(tmp_path, monkeypatch):
    manager = create_manager(tmp_path, monkeypatch)

    cmd = manager.get_jobstep_cmd(
        node="drp-srcf-cmp001",
        job_name="timing_0",
        details={"cmd": "drp -P tst"},
    )

    assert 'echo "===== BATCH ENV BEFORE SRUN (timing_0) ====="' in cmd
    assert "env | sort\n" in cmd
    assert cmd.index("===== BATCH ENV BEFORE SRUN") < cmd.index("srun -n1")
    assert 'echo "===== STEP ENV AFTER SRUN (timing_0) ====="' in cmd
    assert "env | sort; " in cmd
    assert cmd.index("===== STEP ENV AFTER SRUN") < cmd.index("daqlog_header")
