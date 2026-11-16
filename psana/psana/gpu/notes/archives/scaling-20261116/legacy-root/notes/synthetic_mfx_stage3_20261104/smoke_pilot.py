"""Serial disk-discovery smoke test; intentionally skips calibration (Stage 4)."""
import argparse
import json
import os
from pathlib import Path

if os.environ.get("PS_FAKESTEP_FLAG", "0") != "0" or os.environ.get("PS_PARALLEL") != "none":
    raise RuntimeError("Run with PS_FAKESTEP_FLAG=0 and PS_PARALLEL=none")

from psana import DataSource
import psana
from xtc_fixture import COUNTS, DEST_EXP, OLD_NOTES, require, save_json


def smoke(root):
    expected = json.loads((OLD_NOTES / "manifest.json").read_text())["source"]["pilot_timestamps"]
    step_counts, step_times = [], []
    for run in DataSource(exp=DEST_EXP, run=1, dir=str(root), live=False, skip_calib_load="all").runs():
        require(run.expt == DEST_EXP and run.runnum == 1, "Wrong public run identity")
        value, docstring = run.Detector("step_value"), run.Detector("step_docstring")
        for index, step in enumerate(run.steps()):
            require(value(step) == index and json.loads(docstring(step))["step"] == index, "Wrong public step value")
            times = []
            for evt in step.events():
                require(value(evt) == index and json.loads(docstring(evt))["step"] == index, "Wrong event scan state")
                times.append(int(evt.timestamp))
            step_counts.append(len(times))
            step_times.extend(times)
    require(step_counts == COUNTS and step_times == expected, "Public step/event mismatch")
    flat_times = []
    for run in DataSource(exp=DEST_EXP, run=1, dir=str(root), live=False, skip_calib_load="all").runs():
        flat_times.extend(int(evt.timestamp) for evt in run.events())
    require(flat_times == expected, "Public event-only iteration mismatch")
    return {"status": "pass", "directory": str(root), "step_counts": step_counts,
            "event_count": len(flat_times), "timestamps_equal_source": True,
            "step_and_event_scan_values": "pass", "psana": psana.__file__,
            "PS_FAKESTEP_FLAG": os.environ.get("PS_FAKESTEP_FLAG"),
            "PS_PARALLEL": os.environ.get("PS_PARALLEL"),
            "calibration": "skipped; full calibrated serial/MPI acceptance remains Stage 4"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    save_json(args.report, smoke(args.root))
