"""The new input iterator is tested independently of production batch routing."""

import json
import os
from pathlib import Path
import sys

import pytest

from test_callback_chunk_state import run_process, scan_data


def run_iterator(xtc_dir, chunk_size, batch_size=1, stepnum=None, behavior="steps"):
    env = {key: value for key, value in os.environ.items() if not key.startswith("PS_")}
    env.update(PS_PARALLEL="none", PS_SMD_N_EVENTS=str(chunk_size),
               PS_SMD_CHUNKSIZE=str(4 * 1024 * 1024),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    command = [sys.executable, str(Path(__file__).with_name("run_callback_iterator.py")),
               str(xtc_dir), "--batch-size", str(batch_size), "--behavior", behavior]
    if stepnum is not None:
        command += ["--stepnum", str(stepnum)]
    output = run_process(command, env)
    prefix = "CALLBACK_ITERATOR_RESULT="
    records = [line[len(prefix):] for line in output.splitlines() if line.startswith(prefix)]
    assert len(records) == 1, output
    result = json.loads(records[0])
    assert result["callback_calls"] == (0 if behavior == "before_start" else 1), result
    if behavior != "before_start":
        assert result["source_closed"] and result["callback_closed"], result
    assert result["source_close_calls"] == 1, result
    assert result["chunks_read"] == result["source_chunks"], result
    return result


@pytest.mark.parametrize("chunk_size", [100, 4, 7])
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("stepnum", [None, 1])
def test_step_counters_survive_refill(scan_data, chunk_size, batch_size, stepnum):
    xtc_dir, steps = scan_data
    result = run_iterator(xtc_dir, chunk_size, batch_size, stepnum)
    assert result["trace"] == [
        [istep, ievt, ts] for istep, step in enumerate(steps) for ievt, ts in enumerate(step)
    ]
    assert result["selected"] == [
        ts for istep, step in enumerate(steps) if stepnum is None or istep == stepnum
        for ts in step[2:5]
    ]
    # Count input chunks rather than callback invocations: the latter must be 1.
    assert result["source_chunks"] == (30 + chunk_size - 1) // chunk_size
    assert result["transition_count"] > 0
    assert result["caught"] is None


@pytest.mark.parametrize("behavior", ["events", "reject", "return", "close", "before_start", "error"])
def test_callback_lifetime(scan_data, behavior):
    xtc_dir, steps = scan_data
    result = run_iterator(xtc_dir, 4, behavior=behavior)
    if behavior == "events":
        assert result["selected"] == steps[0][2:5]
        assert [row[1] for row in result["trace"]] == list(range(30))
    elif behavior == "reject":
        assert result["selected"] == []
        assert len(result["trace"]) == 30
        assert result["transition_count"] > 0
    elif behavior in ("return", "close"):
        assert result["selected"] == steps[0][2:3]
        assert len(result["trace"]) == 3
        assert result["source_chunks"] == 1
    elif behavior == "before_start":
        assert result["selected"] == []
        assert result["trace"] == []
        assert result["source_chunks"] == 0
    else:
        assert result["caught"] == "callback test error"
        assert len(result["trace"]) == 5
        assert result["source_chunks"] == 2
    if behavior != "error":
        assert result["caught"] is None
    if behavior in ("close", "before_start", "error"):
        assert result["transition_count"] == 0
