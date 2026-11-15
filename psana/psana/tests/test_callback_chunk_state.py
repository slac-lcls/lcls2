"""Regression contract: per-step selection must not depend on SMD chunk size.

Requires the locally built psana environment, xtcwriter/smdwriter and mpirun.
"""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest


DRIVER = Path(__file__).with_name("run_callback_chunk_state.py")
RESULT_PREFIX = "CALLBACK_CHUNK_RESULT="


class CallbackSelectionMismatch(AssertionError):
    """A completed callback run selected incorrect event timestamps."""


def run_process(command, env):
    # Kill the whole process group on timeout, including MPI workers.
    with subprocess.Popen(
        command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True,
    ) as process:
        try:
            output, _ = process.communicate(timeout=90)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
            pytest.fail(f"Timed out: {command}\n{output}")
        assert process.returncode == 0, f"Failed: {command}\n{output}"
    return output


def run_driver(xtc_dir, execution="serial", chunk_size=100, batch_size=1, extra=()):
    env = os.environ.copy()
    # Isolate psana settings from the caller's analysis/DAQ environment.
    for key in list(env):
        if key.startswith("PS_"):
            env.pop(key)
    env.update(
        PS_PARALLEL="mpi" if execution == "mpi" else "none",
        PS_EB_NODES="1", PS_SRV_NODES="0",
        PS_SMD_N_EVENTS=str(chunk_size), PS_SMD_CHUNKSIZE=str(4 * 1024 * 1024),
        OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
    )
    command = [sys.executable, str(DRIVER), str(xtc_dir), "--batch-size", str(batch_size), *extra]
    if execution == "mpi":
        # The four-rank topology must also run on CI hosts with fewer slots.
        command = ["mpirun", "--oversubscribe", "-n", "4", *command]
    output = run_process(command, env)
    records = [line[len(RESULT_PREFIX):] for line in output.splitlines() if line.startswith(RESULT_PREFIX)]
    assert len(records) == 1, f"Expected exactly one result record:\n{output}"
    return json.loads(records[0])


@pytest.fixture(scope="module")
def scan_data(tmp_path_factory):
    xtc_dir = tmp_path_factory.mktemp("callback_chunk_state")
    smd_dir = xtc_dir / "smalldata"
    smd_dir.mkdir()
    for stream in range(2):
        stem = f"xpptut15-r0014-s{stream:03d}-c000"
        xtc_file = xtc_dir / (stem + ".xtc2")
        run_process([
            "xtcwriter", "-f", str(xtc_file), "-t", "-n", "10",
            "-s", str(stream * 2), "-e", "4", "-m", "3",
        ], os.environ.copy())
        run_process([
            "smdwriter", "-f", str(xtc_file), "-o", str(smd_dir / (stem + ".smd.xtc2")),
        ], os.environ.copy())

    steps = run_driver(xtc_dir, extra=["--reference"])["steps"]
    assert [len(step) for step in steps] == [10, 10, 10]
    timestamps = [ts for step in steps for ts in step]
    assert timestamps == sorted(set(timestamps)), "Oracle needs unique, ordered event timestamps"
    return xtc_dir, steps


@pytest.mark.parametrize("execution", ["serial", "mpi"])
@pytest.mark.parametrize("batch_size", [1, 3], ids=["batch1", "batch3"])
@pytest.mark.parametrize("chunk_size", [
    pytest.param(100, id="whole_run"),
    pytest.param(4, id="chunk4"),
    pytest.param(7, id="chunk7"),
])
@pytest.mark.parametrize("stepnum", [None, 1], ids=["all_steps", "step1"])
def test_callback_step_selection(scan_data, execution, batch_size, chunk_size, stepnum):
    xtc_dir, steps = scan_data
    # Match Jungfrau's current upper-index bound: skip 2, then take indices 2:5.
    expected = [
        ts for index, step in enumerate(steps)
        if stepnum is None or index == stepnum
        for ts in step[2:5]
    ]
    extra = ["--evskip", "2", "--nrecs", "5"]
    if stepnum is not None:
        extra += ["--stepnum", str(stepnum)]
    result = run_driver(xtc_dir, execution, chunk_size, batch_size, extra)

    # These are ordinary failures, never hidden by the expected-failure mark.
    assert len(result["trace"]) == 30, result
    assert [row[3] for row in result["trace"]] == [ts for step in steps for ts in step], result
    assert result["invocations"] == 1, result
    assert result["chunks"] == (30 + chunk_size - 1) // chunk_size, result
    assert result["trace"] == [
        [1, istep, ievt, ts] for istep, step in enumerate(steps) for ievt, ts in enumerate(step)
    ]
    if result["selected"] != expected:
        raise CallbackSelectionMismatch(
            f"{execution=}, {chunk_size=}, {batch_size=}, {stepnum=}\n"
            f"expected timestamps: {expected}\nactual timestamps: {result['selected']}\n"
            f"callback trace [invocation, istep, ievt, timestamp]: {result['trace']}"
        )
