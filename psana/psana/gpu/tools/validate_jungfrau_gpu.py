from __future__ import annotations

import argparse

import numpy as np

try:
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None

from psana import DataSource
from psana.gpu.tools.jungfrau_cpu_reference import (
    DEFAULT_CVERSION,
    add_datasource_args,
    apply_max_events,
    datasource_kwargs_from_args,
    load_reference,
)


def _to_host(arr):
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def _prepare_for_compare(cpu_arr, gpu_arr):
    cpu_host = np.asarray(cpu_arr)
    gpu_host = np.asarray(gpu_arr)
    note = None
    if gpu_host.shape != cpu_host.shape:
        squeezed_gpu = np.squeeze(gpu_host)
        if squeezed_gpu.shape == cpu_host.shape:
            note = f"gpu_squeezed_from={gpu_host.shape}"
            gpu_host = squeezed_gpu
        else:
            raise SystemExit(
                f"Shape mismatch: cpu={cpu_host.shape} gpu={gpu_host.shape}"
            )
    return cpu_host, gpu_host, note


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Validate psana Jungfrau GPU arrays against a saved CPU reference file."
    )
    add_datasource_args(parser)
    parser.add_argument(
        "--cpu-reference",
        required=True,
        help="Path to a .npy CPU reference file produced by save_jungfrau_cpu_reference.py.",
    )
    parser.add_argument(
        "--detector",
        default="jungfrau",
        help="Detector name to validate.",
    )
    parser.add_argument(
        "--compare",
        default="calib",
        choices=("raw", "calib", "both"),
        help="Which saved array(s) to compare against the GPU run.",
    )
    parser.add_argument(
        "--events",
        type=int,
        default=2,
        help="Number of reference events to compare.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Pass max_events to DataSource to cap events read from the source.",
    )
    parser.add_argument(
        "--gpu-runtime",
        default="default",
        help="GPU runtime selection forwarded to DataSource (default: default).",
    )
    parser.add_argument(
        "--gpu-pipeline",
        default="default",
        help="GPU pipeline selection forwarded to DataSource (default: default).",
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


def _validate_reference_source(args, reference):
    ref_source = reference.get("source") or {}
    if args.detector != reference.get("detector"):
        raise SystemExit(
            f"Detector mismatch: validator asked for {args.detector!r}, reference has {reference.get('detector')!r}"
        )
    if args.exp is not None or args.run is not None:
        if ref_source.get("exp") != args.exp or int(ref_source.get("run", -1)) != int(args.run):
            raise SystemExit(
                f"Reference source mismatch: validator exp/run={args.exp}/{args.run}, "
                f"reference exp/run={ref_source.get('exp')}/{ref_source.get('run')}"
            )
    if args.file is not None and ref_source.get("files") != args.file:
        raise SystemExit(
            f"Reference file mismatch: validator file={args.file!r}, reference file={ref_source.get('files')!r}"
        )
    if args.compare in ("raw", "both") and reference.get("raw") is None:
        raise SystemExit("Selected --compare raw but the reference file does not contain raw arrays.")


def _iter_compare_modes(compare):
    if compare == "both":
        return ("raw", "calib")
    return (compare,)


def _gpu_array_for_mode(gpu_det, gpu_evt, mode, cversion):
    gpu_raw = getattr(gpu_det, "raw")
    if mode == "raw":
        staged = getattr(gpu_raw, "raw_gpu", None)
        if staged is not None:
            arr = staged(gpu_evt, copy=False)
            if arr is not None:
                return arr, getattr(gpu_evt, "_gpu_raw_storage", None)
        return gpu_raw.raw(gpu_evt, copy=False), getattr(gpu_evt, "_gpu_raw_storage", None)

    staged = getattr(gpu_raw, "calib_gpu", None)
    if staged is not None:
        arr = staged(gpu_evt, copy=False)
        if arr is not None:
            return arr, getattr(gpu_evt, "_gpu_calib_storage", None)
    return gpu_raw.calib(gpu_evt, cversion=cversion), getattr(gpu_evt, "_gpu_calib_storage", None)


def _reference_array_for_mode(reference, index, mode):
    return np.asarray(reference[mode][index])


def _compare_one(mode, reference, index, gpu_det, gpu_evt, cversion, rtol, atol):
    gpu_arr, storage = _gpu_array_for_mode(gpu_det, gpu_evt, mode, cversion)
    if gpu_arr is None:
        raise SystemExit(f"GPU {mode} array is missing for event {index}")
    cpu_arr, gpu_host, compare_note = _prepare_for_compare(
        _reference_array_for_mode(reference, index, mode),
        _to_host(gpu_arr),
    )
    diff = float(np.max(np.abs(cpu_arr - gpu_host)))
    ok = bool(np.allclose(cpu_arr, gpu_host, rtol=rtol, atol=atol))
    note_suffix = f" {compare_note}" if compare_note else ""
    print(
        f"event {index} {mode}: timestamp={int(gpu_evt.timestamp)} "
        f"gpu_type={type(gpu_arr).__module__}.{type(gpu_arr).__name__} "
        f"storage={storage} shape={gpu_host.shape} max_abs_diff={diff:.6g} allclose={ok}{note_suffix}"
    )
    if not ok:
        raise SystemExit(1)


def main():
    args = _build_parser().parse_args()
    reference = load_reference(args.cpu_reference)
    _validate_reference_source(args, reference)

    ref_events = int(reference["events"])
    target_events = min(int(args.events), ref_events)
    if target_events <= 0:
        raise SystemExit("Reference file does not contain any events.")

    ds_kwargs = apply_max_events(datasource_kwargs_from_args(args), args.max_events)
    gpu_ds = DataSource(
        **ds_kwargs,
        gpu_detector=args.detector,
        gpu_runtime=args.gpu_runtime,
        gpu_pipeline=args.gpu_pipeline,
        gpu_queue_depth=args.gpu_queue_depth,
        gpu_profile=args.gpu_profile,
    )
    gpu_run = next(gpu_ds.runs())
    gpu_det = gpu_run.Detector(args.detector)

    ref_timestamps = np.asarray(reference["event_timestamps"], dtype=np.uint64)
    cversion = int(reference.get("cversion", DEFAULT_CVERSION))

    compared = 0
    for gpu_evt in gpu_run.events():
        if compared >= target_events:
            break

        gpu_ts = np.uint64(gpu_evt.timestamp)
        ref_ts = ref_timestamps[compared]
        if gpu_ts != ref_ts:
            raise SystemExit(
                f"Timestamp mismatch at event {compared}: gpu={int(gpu_ts)} reference={int(ref_ts)}"
            )

        for mode in _iter_compare_modes(args.compare):
            _compare_one(mode, reference, compared, gpu_det, gpu_evt, cversion, args.rtol, args.atol)
        compared += 1

    if compared < target_events:
        raise SystemExit(
            f"Reference requested {target_events} events, but GPU stream produced only {compared}."
        )


if __name__ == "__main__":
    main()
