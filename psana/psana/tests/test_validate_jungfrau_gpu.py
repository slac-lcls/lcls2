from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from psana.gpu.tools.jungfrau_cpu_reference import load_reference

EXP = "mfx101344525"
RUN = 125
EVENTS = 10
MAX_EVENTS = 20


@pytest.mark.slow
def test_save_jungfrau_cpu_reference_mfx101344525_r125(tmp_path):
    reference_path, completed = _generate_reference(tmp_path)
    payload = load_reference(reference_path)

    assert completed.returncode == 0
    assert reference_path.exists()
    assert payload["events"] == EVENTS
    assert payload["detector"] == "jungfrau"
    assert int(payload["cversion"]) == 3
    assert payload["source"]["exp"] == EXP
    assert int(payload["source"]["run"]) == RUN
    assert payload["raw"].shape[0] == EVENTS
    assert payload["calib"].shape[0] == EVENTS
    assert payload["raw"].shape == payload["calib"].shape
    assert payload["raw"].dtype == np.uint16
    assert payload["calib"].dtype == np.float32
    assert payload["event_timestamps"].shape == (EVENTS,)


@pytest.mark.slow
def test_validate_jungfrau_gpu_mfx101344525_r125(tmp_path):
    _require_gpu_or_skip()
    reference_path, _ = _generate_reference(tmp_path)

    completed = _run_tool(
        "psana/psana/gpu/tools/validate_jungfrau_gpu.py",
        "-e",
        EXP,
        "-r",
        str(RUN),
        "--cpu-reference",
        str(reference_path),
        "--compare",
        "both",
        "--events",
        str(EVENTS),
        "--max-events",
        str(MAX_EVENTS),
        "--gpu-profile",
        "off",
    )
    _assert_completed(completed)

    assert completed.stdout.count("allclose=True") == EVENTS * 2
    assert "allclose=False" not in completed.stdout


def _generate_reference(tmp_path):
    reference_path = tmp_path / "jungfrau_cpu_reference.npy"
    completed = _run_tool(
        "psana/psana/gpu/tools/save_jungfrau_cpu_reference.py",
        "-e",
        EXP,
        "-r",
        str(RUN),
        "--events",
        str(EVENTS),
        "--max-events",
        str(MAX_EVENTS),
        "--output",
        str(reference_path),
    )
    _assert_completed(completed)
    return reference_path, completed


def _require_gpu_or_skip():
    try:
        import cupy as cp
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GPU unavailable: CuPy import failed ({exc}). Run this test on a node with GPU.")

    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        pytest.skip(
            f"GPU unavailable: CuPy runtime could not enumerate devices ({exc}). "
            "Run this test on a node with GPU."
        )

    if device_count < 1:
        pytest.skip("GPU unavailable: no CUDA devices detected. Run this test on a node with GPU.")


def _run_tool(script_relative_path, *args):
    repo_root = Path(__file__).resolve().parents[3]
    env = dict(os.environ)
    pythonpath_entries = [str(repo_root / "psana"), str(repo_root / "psdaq")]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    command = [sys.executable, str(repo_root / script_relative_path), *args]
    return subprocess.run(
        command,
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _assert_completed(completed):
    if completed.returncode == 0:
        return
    message = (
        f"command failed with exit code {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    pytest.fail(message)
