from __future__ import annotations

import argparse

from psana.gpu.tools.jungfrau_cpu_reference import (
    DEFAULT_CVERSION,
    add_datasource_args,
    apply_max_events,
    create_cpu_reference,
    datasource_kwargs_from_args,
    default_reference_path,
    save_reference,
)


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Save a CPU Jungfrau calibration reference file for later GPU validation."
    )
    add_datasource_args(parser)
    parser.add_argument(
        "--detector",
        default="jungfrau",
        help="Detector name to save.",
    )
    parser.add_argument(
        "--events",
        type=int,
        default=10,
        help="Number of CPU-calibrated events to save.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Pass max_events to DataSource while building the reference.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .npy path. Defaults to a timestamped file under .gpu_reference_cache/.",
    )
    parser.add_argument(
        "--output-dir",
        default=".gpu_reference_cache",
        help="Directory used for the default output path.",
    )
    parser.add_argument(
        "--cversion",
        type=int,
        default=DEFAULT_CVERSION,
        help="Calibration version passed to det.raw.calib for the CPU reference.",
    )
    return parser


def main():
    args = _build_parser().parse_args()
    ds_kwargs = apply_max_events(datasource_kwargs_from_args(args), args.max_events)
    payload = create_cpu_reference(
        ds_kwargs=ds_kwargs,
        detector=args.detector,
        events=args.events,
        cversion=args.cversion,
    )
    output_path = args.output or default_reference_path(ds_kwargs, args.detector, args.output_dir)
    saved = save_reference(output_path, payload)
    print(
        f"saved {payload['events']} CPU reference events detector={payload['detector']} "
        f"cversion={payload['cversion']} raw_shape={payload['raw'].shape} "
        f"calib_shape={payload['calib'].shape} path={saved}"
    )
    for index, ts in enumerate(payload["event_timestamps"]):
        print(f"event {index}: timestamp={int(ts)}")


if __name__ == "__main__":
    main()
