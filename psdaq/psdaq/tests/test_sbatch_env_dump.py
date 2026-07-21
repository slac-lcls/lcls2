import shlex

from psdaq.slurm.utils import (
    DAQMGR_DEBUG_ENV,
    DAQMGR_DEBUG_ENV_DUMP_CMD,
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


def get_bash_payload(jobstep_cmd):
    args = shlex.split(jobstep_cmd)
    bash_index = args.index("bash")
    assert args[bash_index + 1] == "-c"
    return args[bash_index + 2]


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
    assert f"{DAQMGR_DEBUG_ENV_DUMP_CMD}\n" in cmd
    assert cmd.index("===== BATCH ENV BEFORE SRUN") < cmd.index("srun -n1")
    assert 'echo "===== STEP ENV AFTER SRUN (timing_0) ====="' in cmd
    assert f"{DAQMGR_DEBUG_ENV_DUMP_CMD}; " in cmd
    assert cmd.index("===== STEP ENV AFTER SRUN") < cmd.index("daqlog_header")


def test_jobstep_command_redacts_configdb_auth_when_dumping_env(tmp_path, monkeypatch):
    monkeypatch.setenv(DAQMGR_DEBUG_ENV, "1")

    cmd = get_jobstep_cmd(tmp_path, monkeypatch)

    assert "CONFIGDB_AUTH=*****" in cmd
    assert "env | sort" not in cmd


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


def test_sbatch_env_preserves_daqmgr_export_backdoor():
    env = build_sbatch_env(
        {
            "HOME": "/tmp/home",
            "USER": "tstopr",
            "DAQMGR_EXPORT": "RDMAV_FORK_SAFE=1,OPENBLAS_NUM_THREADS=1",
            "SLURM_CPU_BIND": "quiet,mask_cpu:0x1",
        }
    )

    assert env == {
        "HOME": "/tmp/home",
        "USER": "tstopr",
        "DAQMGR_EXPORT": "RDMAV_FORK_SAFE=1,OPENBLAS_NUM_THREADS=1",
    }


def test_jobstep_command_exports_daqmgr_export_for_restart_reuse(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "DAQMGR_EXPORT",
        "RDMAV_FORK_SAFE=1,RDMAV_HUGEPAGES_SAFE=1,OPENBLAS_NUM_THREADS=1",
    )

    cmd = get_jobstep_cmd(tmp_path, monkeypatch)

    assert ",DAQMGR_EXPORT,$DAQMGR_EXPORT" in cmd


def test_jobstep_command_quotes_bash_payload_with_single_quotes(tmp_path, monkeypatch):
    monkeypatch.delenv(DAQMGR_DEBUG_ENV, raising=False)
    manager = create_manager(tmp_path, monkeypatch)
    daq_cmd = "python -c 'print(1)'"

    cmd = manager.get_jobstep_cmd(
        node="drp-srcf-cmp001",
        job_name="quoted_0",
        details={"cmd": daq_cmd},
    )

    payload = get_bash_payload(cmd)
    assert daq_cmd in payload
    assert shlex.split(payload)[:5] == [
        "daqlog_header",
        "quoted_0",
        "4",
        "drp-srcf-cmp001",
        daq_cmd,
    ]


def test_jobstep_command_preserves_shell_sensitive_bash_payload(tmp_path, monkeypatch):
    monkeypatch.delenv(DAQMGR_DEBUG_ENV, raising=False)
    manager = create_manager(tmp_path, monkeypatch)
    daq_cmd = 'echo "$HOME"; printf "%s\\n" done'

    cmd = manager.get_jobstep_cmd(
        node="drp-srcf-cmp001",
        job_name="shell_0",
        details={"cmd": daq_cmd},
    )

    payload = get_bash_payload(cmd)
    assert daq_cmd in payload
    assert shlex.split(payload)[:5] == [
        "daqlog_header",
        "shell_0",
        "4",
        "drp-srcf-cmp001",
        daq_cmd,
    ]
