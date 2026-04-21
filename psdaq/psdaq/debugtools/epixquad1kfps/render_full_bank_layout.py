#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np


def epixquad_detector_view(arr):
    assert arr.shape == (4, 352, 384)
    return np.vstack([
        np.hstack([arr[3], arr[2]]),
        np.hstack([arr[1], arr[0]]),
    ])


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render one detector-view image from extracted full-bank runs and "
            "overlay ASIC numbers at each detected active bank region."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory produced by validate_pattern_runs.py for full-bank ASIC runs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Default: <input-dir>/full_bank_all_asics_layout.png",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold on background_deviation.npy to define active pixels",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Figure title",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="If set, display the figure with matplotlib in addition to saving it",
    )
    parser.add_argument(
        "--view-raw",
        action="store_true",
        help="If set, display the custom epixquad_detector_view instead of det.raw.image",
    )
    parser.add_argument(
        "--xtc-dir",
        default=None,
        help="Optional xtc directory to use when resolving det.raw.image geometry",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    import matplotlib
    if not args.show_plot:
        matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    input_dir = args.input_dir.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else input_dir / "full_bank_all_asics_layout.png"
    )

    run_summaries = json.loads((input_dir / "run_summaries.json").read_text())
    runs = [int(row["run"]) for row in run_summaries]
    exp = str(run_summaries[0].get("exp", "unknown")) if run_summaries else "unknown"
    bank_indices = sorted({int(row.get("bank_index", 0)) for row in run_summaries})
    bank_label = (
        f"bank {bank_indices[0]}"
        if len(bank_indices) == 1
        else "banks " + ",".join(str(v) for v in bank_indices)
    )
    title = (
        args.title
        if args.title is not None
        else f"ePix Quad full-{bank_label} ASIC map, {exp}, runs {min(runs)}-{max(runs)} (top 2-bit code)"
    )

    merged = np.zeros((4, 352, 384), dtype=np.uint8)
    label_markers = np.zeros((4, 352, 384), dtype=np.uint8)

    for row in run_summaries:
        run = int(row["run"])
        pattern_index = int(row["pattern_index"])
        asic = pattern_index

        run_dir = input_dir / f"run{run:04d}_pattern{pattern_index:02d}"
        dominant_code = np.load(run_dir / "dominant_code.npy")
        background_deviation = np.load(run_dir / "background_deviation.npy")

        active = background_deviation > args.threshold
        merged[active] = dominant_code[active]

        coords = np.argwhere(active)
        if coords.size == 0:
            continue

        segment = int(coords[0, 0])
        row0, col0 = coords[:, 1].min(), coords[:, 2].min()
        row1, col1 = coords[:, 1].max(), coords[:, 2].max()
        center_row = int(round(0.5 * (row0 + row1)))
        center_col = int(round(0.5 * (col0 + col1)))
        label_markers[segment, center_row, center_col] = asic + 1

    if args.view_raw:
        img = epixquad_detector_view(merged)
        label_img = epixquad_detector_view(label_markers)
    else:
        from psana import DataSource

        detector_name = str(run_summaries[0].get("detector", "epixquad1kfps"))
        ds_kwargs = {"exp": exp, "run": min(runs)}
        if args.xtc_dir:
            ds_kwargs["dir"] = args.xtc_dir
        ds = DataSource(**ds_kwargs)
        run0 = next(ds.runs())
        det = run0.Detector(detector_name)
        evt0 = next(run0.events())
        img = det.raw.image(evt0, merged)
        label_img = det.raw.image(evt0, label_markers)

    fig, ax = plt.subplots(figsize=(10, 9), facecolor="white")
    ax.imshow(img, origin="lower", cmap="gray_r", vmin=0, vmax=1)
    ax.set_title(title)

    # Module boundaries
    for y in (352,):
        ax.axhline(y - 0.5, color="0.2", linewidth=1.5)
    for x in (384,):
        ax.axvline(x - 0.5, color="0.2", linewidth=1.5)

    # ASIC boundaries
    for y in (176, 528):
        ax.axhline(y - 0.5, color="0.45", linewidth=0.8)
    for x in (192, 576):
        ax.axvline(x - 0.5, color="0.45", linewidth=0.8)

    for asic in range(16):
        hits = np.argwhere(label_img == (asic + 1))
        if hits.size == 0:
            continue
        y, x = hits[0]
        ax.text(
            float(x),
            float(y),
            str(asic),
            color="red",
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5),
        )

    ax.set_xlim(-0.5, img.shape[1] - 0.5)
    ax.set_ylim(-0.5, img.shape[0] - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output, dpi=200, facecolor="white", edgecolor="white")
    if args.show_plot:
        plt.show()
    else:
        plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
