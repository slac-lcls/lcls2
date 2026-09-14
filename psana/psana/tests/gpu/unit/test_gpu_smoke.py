"""CPU checks for the manual MPI smoke report and its Slurm launcher."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[3] / "gpu" / "scripts"


@pytest.fixture
def smoke():
    spec = importlib.util.spec_from_file_location(
        "gpu_multi_rank_smoke", SCRIPTS / "gpu_multi_rank_smoke.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reports(first=(1, 2), second=(3, 4), second_gpu="0000:02:00.0"):
    return [
        dict(rank=0, role="smd0", host="node", gpu=None, timestamps=[]),
        dict(rank=1, role="eb", host="node", gpu=None, timestamps=[]),
        dict(rank=2, role="bd", host="node", gpu="0000:01:00.0",
             timestamps=list(first)),
        dict(rank=3, role="bd", host="node", gpu=second_gpu,
             timestamps=list(second)),
    ]


def test_smoke_reports_full_participation(smoke, capsys):
    assert smoke.summarize(_reports(), 4) == 0
    assert "Active GPU BDs: 2 / 2" in capsys.readouterr().out


def test_smoke_reports_idle_bd_as_incomplete(smoke, capsys):
    assert smoke.summarize(_reports(first=(1, 2, 3, 4), second=()), 4) == 2
    output = capsys.readouterr().out
    assert "INCOMPLETE" in output
    assert "cuda_pci_bus=0000:02:00.0 events=0" in output
    assert "PASS: all requested" not in output


@pytest.mark.parametrize("reports,expected,message", [
    (_reports(), 5, "total events"),
    (_reports(second=(2, 3)), 4, "duplicate timestamps"),
    (_reports(second_gpu="0000:01:00.0"), 4, "share a CUDA device"),
    (_reports(second_gpu=None), 4, "no measured CUDA device identity"),
])
def test_smoke_reports_transport_or_placement_failure(smoke, capsys, reports,
                                                    expected, message):
    assert smoke.summarize(reports, expected) == 1
    assert message in capsys.readouterr().out


@pytest.mark.parametrize("allocated", [False, True])
@pytest.mark.parametrize("child_status", [0, 2, 7])
def test_launcher_preserves_arguments_and_child_status(tmp_path, allocated,
                                                      child_status):
    # Exercise the real shell wrapper without acquiring GPUs. The fake Python
    # acknowledges the install preflight, then reports the forwarded task argv.
    prefix = tmp_path / "install with spaces"
    prefix.mkdir()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (prefix / "activate.sh").write_text(f'export PATH="{bindir}:$PATH"\n')
    python = bindir / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "if sys.argv[1] == '-': sys.exit(0)\n"
        "print('FORWARDED=' + json.dumps(sys.argv[1:]))\n"
        "sys.exit(int(os.environ['SMOKE_CHILD_STATUS']))\n"
    )
    python.chmod(0o755)
    srun = bindir / "srun"
    srun.write_text(
        f"#!{sys.executable}\n"
        "import os, subprocess, sys\n"
        "os.environ['SLURM_PROCID'] = '2'\n"
        "sys.exit(subprocess.call(sys.argv[sys.argv.index('bash'):]))\n"
    )
    srun.chmod(0o755)
    env = dict(os.environ, PSANA_GPU_TEST_PREFIX=str(prefix), N_GPUS_PER_NODE="2",
               PS_EB_NODES="1", PS_SRV_NODES="0",
               SMOKE_CHILD_STATUS=str(child_status))
    env.pop("SLURM_JOB_ID", None)
    if allocated:
        env["SLURM_JOB_ID"] = "123"
    args = ["--a value", "literal $HOME; * 'quoted'"]
    result = subprocess.run(
        ["bash", str(SCRIPTS / "run_multi_gpu_test.sh"), *args],
        env=env, text=True, capture_output=True, timeout=15,
    )
    assert result.returncode == child_status, result.stderr
    line = next(line for line in result.stdout.splitlines()
                if line.startswith("FORWARDED="))
    assert json.loads(line.removeprefix("FORWARDED=")) == [
        str(SCRIPTS / "gpu_multi_rank_smoke.py"), *args
    ]
