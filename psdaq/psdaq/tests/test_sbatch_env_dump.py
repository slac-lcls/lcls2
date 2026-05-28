from psdaq.slurm.utils import (
    DAQMGR_DEBUG_ENV,
    SbatchManager,
    build_sbatch_env,
    daqmgr_debug_env_enabled,
)


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


def get_jobstep_cmd(tmp_path, monkeypatch):
    manager = create_manager(tmp_path, monkeypatch)
    return manager.get_jobstep_cmd(
        node="drp-srcf-cmp001",
        job_name="timing_0",
        details={"cmd": "drp -P tst"},
    )


def test_jobstep_command_omits_env_dump_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv(DAQMGR_DEBUG_ENV, raising=False)

    cmd = get_jobstep_cmd(tmp_path, monkeypatch)

    assert "===== BATCH ENV BEFORE SRUN" not in cmd
    assert "===== STEP ENV AFTER SRUN" not in cmd
    assert "daqlog_header timing_0" in cmd


def test_jobstep_command_dumps_batch_and_step_env_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv(DAQMGR_DEBUG_ENV, "1")

    cmd = get_jobstep_cmd(tmp_path, monkeypatch)

    assert 'echo "===== BATCH ENV BEFORE SRUN (timing_0) ====="' in cmd
    assert "env | sort\n" in cmd
    assert cmd.index("===== BATCH ENV BEFORE SRUN") < cmd.index("srun -n1")
    assert 'echo "===== STEP ENV AFTER SRUN (timing_0) ====="' in cmd
    assert "env | sort; " in cmd
    assert cmd.index("===== STEP ENV AFTER SRUN") < cmd.index("daqlog_header")


def test_daqmgr_debug_env_accepts_explicit_true_values():
    for value in ("1", "true", "yes", "on", "TRUE"):
        assert daqmgr_debug_env_enabled({DAQMGR_DEBUG_ENV: value})


def test_daqmgr_debug_env_is_not_in_sbatch_env_allowlist():
    env = build_sbatch_env(
        {
            "HOME": "/tmp/home",
            "USER": "tstopr",
            DAQMGR_DEBUG_ENV: "1",
            "SLURM_CPU_BIND": "quiet,mask_cpu:0x1",
        }
    )

    assert env == {"HOME": "/tmp/home", "USER": "tstopr"}
