"""Ensure failed or timed-out MPI launchers cannot outlive input cleanup."""
import os
from pathlib import Path
import subprocess
import sys
import pytest
from campaign import run_owned


def test_success_and_failure(tmp_path):
    with (tmp_path/'log').open('w') as log:
        run_owned([sys.executable,'-c','print("complete")'],log,5)
    assert (tmp_path/'log').read_text().strip()=='complete'
    with (tmp_path/'error').open('w') as log:
        with pytest.raises(subprocess.CalledProcessError) as error:
            run_owned([sys.executable,'-c','raise SystemExit(7)'],log,5)
    assert error.value.returncode==7


def test_timeout_reaps_process(tmp_path):
    pid_path=tmp_path/'pid'
    code='import os,time,pathlib;pathlib.Path(%r).write_text(str(os.getpid()));time.sleep(60)'%str(pid_path)
    with (tmp_path/'log').open('w') as log:
        with pytest.raises(subprocess.TimeoutExpired):
            run_owned([sys.executable,'-c',code],log,1)
    pid=int(pid_path.read_text())
    with pytest.raises(ProcessLookupError):os.kill(pid,0)
