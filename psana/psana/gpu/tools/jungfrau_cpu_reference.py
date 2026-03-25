from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from psana import DataSource

REFERENCE_FORMAT = "psana-jungfrau-cpu-reference-v2"
LEGACY_REFERENCE_FORMAT = "psana-jungfrau-cpu-reference-v1"
DEFAULT_CVERSION = 3


def add_datasource_args(parser):
    source_group = parser.add_argument_group("data source")
    source_group.add_argument(
        "--file",
        default=None,
        help="Input xtc2 file.",
    )
    source_group.add_argument(
        "-e",
        "--exp",
        default=None,
        help="Experiment name for DataSource(exp=..., run=...).",
    )
    source_group.add_argument(
        "-r",
        "--run",
        type=int,
        default=None,
        help="Run number for DataSource(exp=..., run=...).",
    )
    source_group.add_argument(
        "--xtc-dir",
        default=None,
        help="Pass dir=... to DataSource for experiment/run mode.",
    )
    return parser


def datasource_kwargs_from_args(args):
    has_file = args.file is not None
    has_exp_run = args.exp is not None or args.run is not None

    if has_file and has_exp_run:
        raise SystemExit("Use either --file or --exp/--run, not both.")
    if has_file:
        if args.xtc_dir is not None:
            raise SystemExit("--xtc-dir is only valid with --exp/--run.")
        return {"files": args.file}
    if args.exp is not None and args.run is not None:
        ds_kwargs = {"exp": args.exp, "run": args.run}
        if args.xtc_dir is not None:
            ds_kwargs["dir"] = args.xtc_dir
        return ds_kwargs
    if args.exp is None and args.run is None:
        raise SystemExit("Provide either --file or both --exp and --run.")
    raise SystemExit("When using experiment mode, both --exp and --run are required.")


def apply_max_events(ds_kwargs, max_events):
    if max_events is None:
        return dict(ds_kwargs)
    limited = dict(ds_kwargs)
    limited["max_events"] = max_events
    return limited


def default_reference_path(ds_kwargs, detector, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if "exp" in ds_kwargs and "run" in ds_kwargs:
        stem = f"{ds_kwargs['exp']}_r{int(ds_kwargs['run']):04d}_{detector}"
    else:
        stem = f"{Path(ds_kwargs['files']).stem}_{detector}"
    return output_dir / f"{stem}_cpu_ref_{timestamp}.npy"


def create_cpu_reference(ds_kwargs, detector="jungfrau", events=10, cversion=DEFAULT_CVERSION):
    ds = DataSource(**ds_kwargs)
    run = next(ds.runs())
    det = run.Detector(detector)

    timestamps = []
    raw_events = []
    calib_events = []
    for evt in run.events():
        raw = det.raw.raw(evt, copy=False)
        calib = det.raw.calib(evt, cversion=cversion)
        if raw is None or calib is None:
            continue

        raw_host = np.array(raw, dtype=np.uint16, copy=True)
        calib_host = np.array(calib, dtype=np.float32, copy=True)
        if raw_events and raw_host.shape != raw_events[0].shape:
            raise SystemExit(
                f"Reference raw shape changed within run: first={raw_events[0].shape} current={raw_host.shape}"
            )
        if calib_events and calib_host.shape != calib_events[0].shape:
            raise SystemExit(
                f"Reference calib shape changed within run: first={calib_events[0].shape} current={calib_host.shape}"
            )

        timestamps.append(np.uint64(evt.timestamp))
        raw_events.append(raw_host)
        calib_events.append(calib_host)
        if len(calib_events) >= events:
            break

    if not calib_events:
        raise SystemExit("No CPU reference events were collected.")

    raw_stack = np.stack(raw_events, axis=0)
    calib_stack = np.stack(calib_events, axis=0)
    return {
        "format": REFERENCE_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "detector": detector,
        "cversion": int(cversion),
        "events": int(calib_stack.shape[0]),
        "event_timestamps": np.asarray(timestamps, dtype=np.uint64),
        "raw": raw_stack,
        "calib": calib_stack,
        "source": dict(ds_kwargs),
        "run_timestamp": int(getattr(run, "timestamp", 0)),
    }


def save_reference(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, payload, allow_pickle=True)
    return path


def load_reference(path):
    payload = np.load(path, allow_pickle=True)
    if isinstance(payload, np.ndarray) and payload.shape == ():
        payload = payload.item()
    if not isinstance(payload, dict):
        raise SystemExit(f"Unsupported reference payload type in {path}: {type(payload).__name__}")

    fmt = payload.get("format")
    if fmt == LEGACY_REFERENCE_FORMAT:
        payload = dict(payload)
        payload["format"] = REFERENCE_FORMAT
        payload.setdefault("raw", None)
        return payload
    if fmt != REFERENCE_FORMAT:
        raise SystemExit(
            f"Unsupported reference format in {path}: {fmt!r}"
        )
    return payload
