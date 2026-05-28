import asyncio
import sys

from psdaq.slurm.subproc import SubprocHelper


def run_python(code, echo_output=False):
    helper = SubprocHelper()
    return asyncio.run(
        helper.run_exec(
            [sys.executable, "-c", code],
            wait_output=True,
            echo_output=echo_output,
        )
    )


def test_run_exec_can_suppress_success_output(capsys):
    rc = run_python(
        "import sys; print('stdout ok'); print('stderr ok', file=sys.stderr)"
    )

    assert rc == 0
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == ""


def test_run_exec_prints_failure_output_when_suppressed(capsys):
    rc = run_python(
        "import sys; print('stdout failed'); "
        "print('stderr failed', file=sys.stderr); sys.exit(7)"
    )

    assert rc == 7
    output = capsys.readouterr()
    assert output.out == "stdout failed\n"
    assert output.err == "stderr failed\n"
