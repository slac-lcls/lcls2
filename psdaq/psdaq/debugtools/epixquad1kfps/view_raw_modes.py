import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from psana import DataSource
from psana.detector.UtilsEpix10ka import gain_maps_epix10ka_any

mpl.rcParams["font.size"] = 8

MODE_NAMES = ("FH", "FM", "FL", "AHL_H", "AML_M", "AHL_L", "AML_L")
MODE_COLORS = (
    "#b2182b",
    "#ef8a62",
    "#2166ac",
    "#762a83",
    "#999999",
    "#4dac26",
    "#1b7837",
)
TOP2_NAMES = ("00", "01", "10", "11")
TOP2_COLORS = ("#2f2f2f", "#1f78b4", "#e66101", "#984ea3")


def epixquad_detector_view(arr):
    assert arr.shape == (4, 352, 384)
    return np.vstack([
        np.hstack([arr[3], arr[0]]),
        np.hstack([arr[2], arr[1]]),
    ])


def parse_args():
    parser = argparse.ArgumentParser(
        description="View epixquad raw, raw upper bits, and decoded gain mode."
    )
    parser.add_argument("-e", "--exp", required=True, help="Experiment name")
    parser.add_argument("-r", "--run", required=True, type=int, help="Run number")
    parser.add_argument("--max-events", type=int, default=100, help="Maximum events to scan")
    parser.add_argument("--evt-idx", type=int, default=50, help="Event index to display")
    parser.add_argument(
        "--view-raw",
        action="store_true",
        help="If set, use epixquad_detector_view instead of det.raw.image",
    )
    parser.add_argument(
        "--mask-npy",
        type=Path,
        default=None,
        help="Optional mask/gainmap numpy file to show as the first panel",
    )
    return parser.parse_args()


def _mask_view(mask, *, det, evt, use_raw_view):
    if mask.ndim == 3 and mask.shape == (4, 352, 384):
        mask = mask.astype(np.uint8)
        return epixquad_detector_view(mask) if use_raw_view else det.raw.image(evt, mask)
    if mask.ndim == 2:
        return mask
    raise ValueError(f"unsupported mask shape: {mask.shape}")


def _gain_index_map(det_raw, evt):
    gmaps = gain_maps_epix10ka_any(det_raw, evt)
    if gmaps is None:
        return None
    return np.select(
        gmaps,
        [0, 1, 2, 3, 4, 5, 6],
        default=-1,
    ).astype(np.int8)


def _image_view(det, evt, arr, use_raw_view):
    return epixquad_detector_view(arr) if use_raw_view else det.raw.image(evt, arr)


def _discrete_image(ax, img, *, labels, colors, title):
    cmap = ListedColormap(colors)
    bounds = np.arange(-0.5, len(labels) + 0.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)
    im = ax.imshow(img, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=np.arange(len(labels)))
    cbar.ax.set_yticklabels(labels)
    ax.set_title(title)


def main():
    args = parse_args()

    mask = None
    if args.mask_npy is not None:
        mask = np.load(args.mask_npy)

    ds = DataSource(exp=args.exp, run=args.run, max_events=args.max_events)
    run = next(ds.runs())
    det = run.Detector("epixquad1kfps")

    for i_evt, evt in enumerate(run.events()):
        raw = det.raw.raw(evt)
        if raw is None:
            continue

        top2 = ((raw >> 14) & 0x3).astype(np.uint8)
        gain_index = _gain_index_map(det.raw, evt)
        if gain_index is None:
            continue

        if i_evt != args.evt_idx:
            continue

        print(f"Event {i_evt}: {evt.timestamp} {raw.shape}")
        top2_u, top2_c = np.unique(top2, return_counts=True)
        print("top2 unique/counts:", list(zip(top2_u.tolist(), top2_c.tolist())))
        gi_u, gi_c = np.unique(gain_index, return_counts=True)
        print(
            "gain-index unique/counts:",
            [
                (MODE_NAMES[idx] if idx >= 0 else "unknown", int(count))
                for idx, count in zip(gi_u.tolist(), gi_c.tolist())
            ],
        )

        mask_img = None if mask is None else _mask_view(mask, det=det, evt=evt, use_raw_view=args.view_raw)
        ncols = 4 if mask_img is not None else 3
        fig, axs = plt.subplots(1, ncols, squeeze=False, figsize=(4.0 * ncols, 4.2))

        raw14 = raw & 0x3FFF
        vmin, vmax = np.percentile(raw14, [1, 99])
        raw_img = _image_view(det, evt, raw14, args.view_raw)
        top2_img = _image_view(det, evt, top2, args.view_raw)
        gain_index_img = _image_view(det, evt, gain_index.astype(np.float32), args.view_raw)
        view_label = "custom detector view" if args.view_raw else "det.raw.image"

        col = 0
        if mask_img is not None:
            axs[0, col].imshow(mask_img, origin="lower", cmap="gray_r", vmin=0, vmax=1)
            axs[0, col].set_title(f"Mask NPY: {args.mask_npy.name}")
            col += 1

        axs[0, col].imshow(raw_img, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        axs[0, col].set_title(f"Raw14 ({view_label})")

        _discrete_image(
            axs[0, col + 1],
            top2_img,
            labels=TOP2_NAMES,
            colors=TOP2_COLORS,
            title=f"Top 2 Raw Bits ({view_label})",
        )
        _discrete_image(
            axs[0, col + 2],
            gain_index_img,
            labels=MODE_NAMES,
            colors=MODE_COLORS,
            title=f"Decoded Gain Mode ({view_label})",
        )

        for ax in axs.ravel():
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        plt.show()
        run.terminate()
        return

    raise RuntimeError(f"did not find valid event index {args.evt_idx} within {args.max_events} events")


if __name__ == "__main__":
    main()
