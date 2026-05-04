import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from psana import DataSource

mpl.rcParams["font.size"] = 8

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
        description="View epixquad raw, calib, and raw upper bits."
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
        "--show-calib",
        action="store_true",
        help="If set, add a det.raw.calib panel",
    )
    parser.add_argument(
        "--mask-npy",
        type=Path,
        default=None,
        help="Optional mask numpy file to show as the first panel",
    )
    return parser.parse_args()


def _mask_view(mask, *, det, evt, use_raw_view):
    if mask.ndim == 3 and mask.shape == (4, 352, 384):
        mask = mask.astype(np.uint8)
        return epixquad_detector_view(mask) if use_raw_view else det.raw.image(evt, mask)
    if mask.ndim == 2:
        return mask
    raise ValueError(f"unsupported mask shape: {mask.shape}")


def _image_view(det, evt, arr, use_raw_view):
    return epixquad_detector_view(arr) if use_raw_view else det.raw.image(evt, arr)


def _percentile_limits(arr, percentiles=(1, 99)):
    finite = np.asarray(arr)[np.isfinite(arr)]
    if finite.size == 0:
        return None, None
    vmin, vmax = np.percentile(finite, percentiles)
    if vmin == vmax:
        return None, None
    return vmin, vmax


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

        if i_evt != args.evt_idx:
            continue

        print(f"Event {i_evt}: {evt.timestamp} {raw.shape}")
        top2_u, top2_c = np.unique(top2, return_counts=True)
        print("top2 unique/counts:", list(zip(top2_u.tolist(), top2_c.tolist())))

        calib = None
        if args.show_calib:
            calib = det.raw.calib(evt)
            if calib is None:
                raise RuntimeError(f"det.raw.calib returned None for event index {i_evt}")
            print(f"calib shape: {calib.shape}")

        mask_img = None if mask is None else _mask_view(mask, det=det, evt=evt, use_raw_view=args.view_raw)
        ncols = 2
        if mask_img is not None:
            ncols += 1
        if args.show_calib:
            ncols += 1
        fig, axs = plt.subplots(1, ncols, squeeze=False, figsize=(4.0 * ncols, 4.2))

        raw14 = raw & 0x3FFF
        vmin, vmax = _percentile_limits(raw14)
        raw_img = _image_view(det, evt, raw14, args.view_raw)
        top2_img = _image_view(det, evt, top2, args.view_raw)
        view_label = "custom detector view" if args.view_raw else "det.raw.image"

        col = 0
        if mask_img is not None:
            axs[0, col].imshow(mask_img, origin="lower", cmap="gray_r", vmin=0, vmax=1)
            axs[0, col].set_title(f"Mask NPY: {args.mask_npy.name}")
            col += 1

        axs[0, col].imshow(raw_img, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        axs[0, col].set_title(f"Raw14 ({view_label})")
        col += 1

        if args.show_calib:
            calib_img = _image_view(det, evt, calib, args.view_raw)
            calib_vmin, calib_vmax = _percentile_limits(calib_img)
            axs[0, col].imshow(
                calib_img,
                origin="lower",
                cmap="viridis",
                vmin=calib_vmin,
                vmax=calib_vmax,
            )
            axs[0, col].set_title(f"Calib ({view_label})")
            col += 1

        _discrete_image(
            axs[0, col],
            top2_img,
            labels=TOP2_NAMES,
            colors=TOP2_COLORS,
            title=f"Top 2 Raw Bits ({view_label})",
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
