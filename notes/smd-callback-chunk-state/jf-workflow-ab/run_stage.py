"""Run the actual pedestal CLI with observational A/B instrumentation.

Usage: python run_stage.py OUTPUT_DIR -- <jungfrau_dark_proc arguments>
The selected PYTHONPATH supplies either unchanged pre-fix or fixed production.
"""

import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
from mpi4py import MPI

output = Path(sys.argv[1]).resolve()
output.mkdir(parents=True, exist_ok=True)
cli = sys.argv[2:]
if cli and cli[0] == "--":
    cli = cli[1:]
assert "--deploy" not in cli and "-D" not in cli
assert "--submit" not in cli and "--wrapper" not in cli
sys.argv = ["jungfrau_dark_proc", *cli]
rank = MPI.COMM_WORLD.rank
report = dict(rank=rank, size=MPI.COMM_WORLD.size, argv=cli,
              smd_n_events=os.environ.get("PS_SMD_N_EVENTS"), chunks=[],
              callback_calls=0, callback_timestamps=[], raw_timestamps=[],
              summaries=[])

from psana.detector import UtilsJungfrauCalib as ujc
from psana.detector.areadetector import AreaDetectorRaw
from psana.psexp.ds_base import DataSourceBase
from psana.psexp.smdreader_manager import SmdReaderManager
from psana.app.jungfrau_dark_proc import do_main

original_open = ujc.open_DataSource
original_build = SmdReaderManager.build_batch
original_raw = AreaDetectorRaw.raw
original_summary = ujc.DarkProcJungfrau.summary
original_name = ujc.uc.detector_name_short


def observed_open(**kwargs):
    callback = kwargs.get("smd_callback")
    if callback:
        def observed_callback(run):
            report["callback_calls"] += 1
            for evt in callback(run):
                report["callback_timestamps"].append(int(evt.timestamp))
                yield evt
        kwargs["smd_callback"] = observed_callback
    # monitor=False is carried in -k, which open_DataSource parses unchanged.
    return original_open(**kwargs)


def observed_build(self):
    success = original_build(self)
    if success and self.smdr.view_size:
        report["chunks"].append(int(self.smdr.view_size))
    return success


def observed_raw(self, evt, *args, **kwargs):
    raw = original_raw(self, evt, *args, **kwargs)
    if raw is not None:
        report["raw_timestamps"].append(int(evt.timestamp))
        report.setdefault("raw_shape", list(raw.shape))
    return raw


def observed_summary(self):
    result = original_summary(self)
    stem = "rank%d-gain%d" % (rank, self.gmindex)
    arrays = {}
    for name in ("arr_av1", "arr_rms", "arr_sta", "arr_sum0"):
        value = getattr(self, name, None)
        if value is not None:
            arrays[name] = value
    np.savez(output / (stem + ".npz"), **arrays)
    report["summaries"].append(dict(gain=self.gmindex, file=stem + ".npz"))
    return result


def readonly_name(*args, **kwargs):
    kwargs["add_shortname"] = False
    return original_name(*args, **kwargs)


report["sources"] = dict(calibration=ujc.__file__, smdreader=sys.modules[
    "psana.psexp.smdreader_manager"].__file__)
report["resolved_sources"] = {key: os.path.realpath(value)
                              for key, value in report["sources"].items()}
try:
    with patch.object(ujc, "open_DataSource", observed_open), \
            patch.object(SmdReaderManager, "build_batch", observed_build), \
            patch.object(AreaDetectorRaw, "raw", observed_raw), \
            patch.object(ujc.DarkProcJungfrau, "summary", observed_summary), \
            patch.object(ujc.uc, "detector_name_short", readonly_name), \
            patch.object(DataSourceBase, "_get_file_info_from_db", return_value={}):
        do_main()
    report["completed"] = True
except BaseException as exc:
    report["completed"] = False
    report["exception"] = repr(exc)
    raise
finally:
    (output / ("rank%d.json" % rank)).write_text(json.dumps(report, indent=2) + "\n")
