"""Production routing, transition draining and bounded callback output."""

import pytest
import os
from pathlib import Path
import sys

from test_callback_chunk_state import scan_data, run_driver, run_process


def test_backpressure_and_cleanup():
    env = dict(os.environ, PS_PARALLEL="none")
    output = run_process([sys.executable, str(Path(__file__).with_name("run_callback_stream.py"))], env)
    assert "CALLBACK_STREAM_STRESS_OK" in output


@pytest.mark.parametrize("execution", ["serial", "mpi"])
def test_fresh_callback_per_run(scan_data, execution):
    xtc_dir, steps = scan_data
    result = run_driver(xtc_dir, execution, 4, 3, ["--twice"])
    expected = [ts for step in steps for ts in step[2:5]] * 2
    assert result["selected"] == (sorted(expected) if execution == "mpi" else expected)
    assert result["invocations"] == 2
    assert result["chunks"] == 16
    for invocation in (1, 2):
        assert [row[1:] for row in result["trace"] if row[0] == invocation] == [
            [istep, ievt, ts] for istep, step in enumerate(steps) for ievt, ts in enumerate(step)
        ]


@pytest.mark.parametrize("execution", ["serial", "mpi"])
def test_callback_at_max_events(scan_data, execution):
    xtc_dir, steps = scan_data
    result = run_driver(xtc_dir, execution, 4, 3, ["--max-events", "11"])
    assert result["selected"] == steps[0][2:5]
    assert result["invocations"] == 1
    assert len(result["trace"]) == 11


@pytest.mark.parametrize("execution", ["serial", "mpi"])
@pytest.mark.parametrize("behavior", ["select", "reject", "return"])
@pytest.mark.parametrize("destinations", [False, True])
def test_stream_delivery(scan_data, execution, behavior, destinations):
    xtc_dir, steps = scan_data
    extra = ["--behavior", behavior, "--small-limits"]
    if destinations:
        extra.append("--destinations")
    result = run_driver(xtc_dir, execution, 4, 3, extra)
    reference = run_driver(xtc_dir, extra=["--reference"])["metadata"]
    expected = [ts for step in steps for ts in step[2:5]]
    if behavior == "reject":
        expected = []
    elif behavior == "return":
        expected = expected[:1]
    assert result["selected"] == expected
    assert result["invocations"] == 1
    assert result["chunks"] == 8  # Early return drains the full run's transitions.
    assert result["step_counts"] == ([3] if execution == "serial" else [3, 3])
    assert result["metadata"] == [row for row in reference if row[0] in expected]
    expected_callback = reference[:3] if behavior == "return" else reference
    assert result["callback_metadata"] == expected_callback
    if execution == "mpi":
        receivers = dict(result["received"])
        for ts, dest in result["destinations"]:
            if dest:
                assert receivers[ts] == dest
