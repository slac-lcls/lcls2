#!/usr/bin/env python3

import argparse
import json

import numpy as np
from psana import DataSource


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect epixquad raw/calibrated noise for pixels selected by the "
            "run-embedded asicPixelConfig gain map."
        )
    )
    parser.add_argument("--exp", required=True, help="Experiment name, e.g. ued1016014")
    parser.add_argument("--run", required=True, type=int, help="Run number")
    parser.add_argument(
        "--detector",
        default="epixquad1kfps",
        help="Detector name passed to run.Detector()",
    )
    parser.add_argument(
        "--max-events",
        default=12,
        type=int,
        help="Maximum number of non-empty events to inspect",
    )
    parser.add_argument(
        "--low-value",
        default=8,
        type=int,
        help="Pixel-map code to treat as the low-gain region",
    )
    parser.add_argument(
        "--med-value",
        default=12,
        type=int,
        help="Pixel-map code to treat as the medium/background region",
    )
    parser.add_argument(
        "--indent",
        default=2,
        type=int,
        help="JSON indentation level",
    )
    return parser.parse_args()


def stats(arr):
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def asics_to_segments(pixel_map, raw_shape):
    if pixel_map.shape[0] != 16 or pixel_map.shape[2] != 192:
        raise RuntimeError(f"unexpected asic pixel_map shape {pixel_map.shape}")

    if pixel_map.shape[1] > 176:
        pixel_map = pixel_map[:, :176, :]

    segs = []
    for seg in range(4):
        a0, a1, a2, a3 = pixel_map[4 * seg : 4 * seg + 4]
        top = np.hstack((np.flipud(np.fliplr(a2)), np.flipud(np.fliplr(a1))))
        bot = np.hstack((a3, a0))
        segs.append(np.vstack((top, bot)))

    segs = np.asarray(segs, dtype=np.uint8)
    if segs.shape != raw_shape:
        raise RuntimeError(
            f"reconstructed pixel_map shape {segs.shape} incompatible with raw shape {raw_shape}"
        )
    return segs


def event_summary(evt_idx, det, evt, pixel_map, low_value, med_value):
    raw = det.raw.raw(evt)
    if raw is None:
        return None

    raw = np.asarray(raw)
    raw_adc = raw & 0x3FFF
    seg_pixel_map = asics_to_segments(pixel_map, raw.shape)
    low_mask = seg_pixel_map == low_value
    med_mask = seg_pixel_map == med_value

    summary = {
        "event": evt_idx,
        "raw_shape": list(raw.shape),
        "low_pixels": int(low_mask.sum()),
        "med_pixels": int(med_mask.sum()),
        "raw_low": stats(raw_adc[low_mask]),
        "raw_med": stats(raw_adc[med_mask]),
    }

    calib = det.raw.calib(evt)
    if calib is not None:
        calib = np.asarray(calib)
        if calib.shape == seg_pixel_map.shape:
            summary["calib_low"] = stats(calib[low_mask])
            summary["calib_med"] = stats(calib[med_mask])

    image = det.raw.image(evt, nda=raw_adc)
    if image is not None:
        image = np.asarray(image)
        summary["image_shape"] = list(image.shape)

    return summary


def analyze_run(args):
    ds = DataSource(exp=args.exp, run=args.run)
    run = next(ds.runs())
    det = run.Detector(args.detector)
    seg_cfgs = det.raw._seg_configs()

    asic_pixel_map = np.concatenate(
        [np.asarray(seg_cfgs[i].config.asicPixelConfig, dtype=np.uint8) for i in sorted(seg_cfgs)],
        axis=0,
    )

    result = {
        "exp": args.exp,
        "run": args.run,
        "detector": args.detector,
        "run_asic_pixel_map_shape": list(asic_pixel_map.shape),
        "run_asic_pixel_map_unique": {
            int(v): int(c) for v, c in zip(*np.unique(asic_pixel_map, return_counts=True))
        },
        "trbit": {
            int(i): [int(v) for v in np.asarray(seg_cfgs[i].config.trbit).tolist()]
            for i in sorted(seg_cfgs)
        },
    }

    events = []
    for evt_idx, evt in enumerate(run.events()):
        entry = event_summary(evt_idx, det, evt, asic_pixel_map, args.low_value, args.med_value)
        if entry is None:
            continue
        events.append(entry)
        if len(events) >= args.max_events:
            break

    result["events"] = events
    if events:
        result["aggregate"] = {
            "raw_low_std_mean": float(np.mean([e["raw_low"]["std"] for e in events])),
            "raw_med_std_mean": float(np.mean([e["raw_med"]["std"] for e in events])),
            "raw_low_mean_mean": float(np.mean([e["raw_low"]["mean"] for e in events])),
            "raw_med_mean_mean": float(np.mean([e["raw_med"]["mean"] for e in events])),
        }
        if all("calib_low" in e and "calib_med" in e for e in events):
            result["aggregate"]["calib_low_std_mean"] = float(
                np.mean([e["calib_low"]["std"] for e in events])
            )
            result["aggregate"]["calib_med_std_mean"] = float(
                np.mean([e["calib_med"]["std"] for e in events])
            )

    return result


def main():
    args = parse_args()
    print(json.dumps(analyze_run(args), indent=args.indent))


if __name__ == "__main__":
    main()
