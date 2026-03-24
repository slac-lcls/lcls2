import argparse
import numpy as np

try:
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None

from psana import DataSource


def _to_host(arr):
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Validate psana Jungfrau GPU calib against the CPU reference."
    )
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
    parser.add_argument(
        "--detector",
        default="jungfrau",
        help="Detector name to validate.",
    )
    parser.add_argument(
        "--events",
        type=int,
        default=2,
        help="Number of events to compare.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Pass max_events to DataSource to cap events read from the source.",
    )
    parser.add_argument(
        "--gpu-queue-depth",
        type=int,
        default=3,
        help="Queue depth for gpu_detector mode.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-4,
        help="Relative tolerance for np.allclose.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for np.allclose.",
    )
    parser.add_argument(
        "--gpu-profile",
        default="summary",
        choices=("off", "summary", "trace"),
        help="Profiler mode for the GPU DataSource.",
    )
    return parser


def _datasource_kwargs(args):
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


def main():
    args = _build_parser().parse_args()
    ds_kwargs = _datasource_kwargs(args)

    if args.max_events is not None:
        ds_kwargs = dict(ds_kwargs)
        ds_kwargs["max_events"] = args.max_events

    cpu_ds = DataSource(**ds_kwargs)
    cpu_run = next(cpu_ds.runs())
    cpu_det = cpu_run.Detector(args.detector)

    gpu_ds = DataSource(
        **ds_kwargs,
        gpu_detector=args.detector,
        gpu_queue_depth=args.gpu_queue_depth,
        gpu_profile=args.gpu_profile,
    )
    gpu_run = next(gpu_ds.runs())
    gpu_det = gpu_run.Detector(args.detector)

    compared = 0
    for event_index, (cpu_evt, gpu_evt) in enumerate(zip(cpu_run.events(), gpu_run.events())):
        cpu_arr = cpu_det.raw.calib(cpu_evt)
        gpu_arr = gpu_det.raw.calib(gpu_evt)
        gpu_host = _to_host(gpu_arr)
        diff = float(np.max(np.abs(cpu_arr - gpu_host)))
        ok = bool(np.allclose(cpu_arr, gpu_host, rtol=args.rtol, atol=args.atol))
        print(
            f"event {event_index}: gpu_type={type(gpu_arr).__name__} "
            f"shape={gpu_host.shape} max_abs_diff={diff:.6g} allclose={ok}"
        )
        if not ok:
            raise SystemExit(1)
        compared += 1
        if compared >= args.events:
            break

    if compared == 0:
        raise SystemExit("No events were compared.")


if __name__ == "__main__":
    main()
