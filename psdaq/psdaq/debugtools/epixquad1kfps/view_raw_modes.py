import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from psdaq.configdb.epixquad_layout import raw_detector_view
from psana import DataSource

mpl.rcParams["font.size"] = 8

TOP2_NAMES = ("00", "01", "10", "11")
TOP2_COLORS = ("#2f2f2f", "#1f78b4", "#e66101", "#984ea3")
MISMATCH_COLORS = {"FP": "#ff2d2d", "FN": "#00d5ff"}


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
        "--show-ped-sub",
        action="store_true",
        help="If set, add raw14 minus FM and FL pedestal panels",
    )
    parser.add_argument(
        "--fm-ped-index",
        type=int,
        default=1,
        help="Pedestal gain index to use for fixed medium subtraction",
    )
    parser.add_argument(
        "--fl-ped-index",
        type=int,
        default=2,
        help="Pedestal gain index to use for fixed low subtraction",
    )
    parser.add_argument(
        "--mask-npy",
        type=Path,
        default=None,
        help="Optional mask numpy file to show as the first panel",
    )
    parser.add_argument(
        "--fp-fn-csv",
        type=Path,
        default=None,
        help="Optional diagnosis CSV with FP/FN pixels to overlay",
    )
    return parser.parse_args()


def _mask_view(mask, *, det, evt, use_raw_view):
    if mask.ndim == 3 and mask.shape == (4, 352, 384):
        mask = mask.astype(np.uint8)
        return raw_detector_view(mask) if use_raw_view else det.raw.image(evt, mask)
    if mask.ndim == 2:
        return mask
    raise ValueError(f"unsupported mask shape: {mask.shape}")


def _image_view(det, evt, arr, use_raw_view):
    return raw_detector_view(arr) if use_raw_view else det.raw.image(evt, arr)


def _load_mismatch_labels(csv_path, run_number):
    labels = np.zeros((4, 352, 384), dtype=np.uint8)
    counts = {"FP": 0, "FN": 0}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if "run" in row and row["run"] and int(row["run"]) != run_number:
                continue
            kind = row.get("kind", "").upper()
            if kind not in ("FP", "FN"):
                continue
            segment = int(row["segment"])
            raw_row = int(row["raw_row"])
            raw_col = int(row["raw_col"])
            labels[segment, raw_row, raw_col] = 1 if kind == "FP" else 2
            counts[kind] += 1
    return labels, counts


def _overlay_mismatches(ax, label_img):
    if label_img is None:
        return
    for code, kind in ((1, "FP"), (2, "FN")):
        ys, xs = np.nonzero(label_img == code)
        if xs.size == 0:
            continue
        ax.scatter(
            xs,
            ys,
            s=14,
            marker="s",
            facecolors="none",
            edgecolors=MISMATCH_COLORS[kind],
            linewidths=0.8,
            label=kind,
        )


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
    mismatch_labels = None
    mismatch_counts = None
    if args.fp_fn_csv is not None:
        mismatch_labels, mismatch_counts = _load_mismatch_labels(args.fp_fn_csv, args.run)
        print(
            f"loaded FP/FN overlay from {args.fp_fn_csv}: "
            f"FP={mismatch_counts['FP']} FN={mismatch_counts['FN']} for run {args.run}"
        )

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

        ped_fm = None
        ped_fl = None
        if args.show_ped_sub:
            pedestals = det.raw._pedestals()
            if pedestals is None:
                raise RuntimeError("det.raw._pedestals returned None")
            if pedestals.shape[1:] != raw.shape:
                raise RuntimeError(
                    f"pedestal shape {pedestals.shape} is not compatible with raw shape {raw.shape}"
                )
            if max(args.fm_ped_index, args.fl_ped_index) >= pedestals.shape[0]:
                raise RuntimeError(
                    f"pedestal shape {pedestals.shape} does not include requested indices "
                    f"FM={args.fm_ped_index}, FL={args.fl_ped_index}"
                )
            ped_fm = np.asarray(pedestals[args.fm_ped_index], dtype=np.float32)
            ped_fl = np.asarray(pedestals[args.fl_ped_index], dtype=np.float32)
            print(
                "pedestals:",
                f"shape={pedestals.shape}",
                f"FM index={args.fm_ped_index}",
                f"FL index={args.fl_ped_index}",
            )

        mask_img = None if mask is None else _mask_view(mask, det=det, evt=evt, use_raw_view=args.view_raw)
        mismatch_img = (
            None
            if mismatch_labels is None
            else _image_view(det, evt, mismatch_labels, args.view_raw)
        )
        ncols = 2
        if mask_img is not None:
            ncols += 1
        if args.show_calib:
            ncols += 1
        if args.show_ped_sub:
            ncols += 2
        fig, axs = plt.subplots(1, ncols, squeeze=False, figsize=(4.0 * ncols, 4.2))

        raw14 = raw & 0x3FFF
        vmin, vmax = _percentile_limits(raw14)
        raw_img = _image_view(det, evt, raw14, args.view_raw)
        top2_img = _image_view(det, evt, top2, args.view_raw)
        view_label = "custom detector view" if args.view_raw else "det.raw.image"

        col = 0
        if mask_img is not None:
            axs[0, col].imshow(mask_img, origin="lower", cmap="gray_r", vmin=0, vmax=1)
            _overlay_mismatches(axs[0, col], mismatch_img)
            axs[0, col].set_title(f"Mask NPY: {args.mask_npy.name}")
            col += 1

        axs[0, col].imshow(raw_img, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        _overlay_mismatches(axs[0, col], mismatch_img)
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
            _overlay_mismatches(axs[0, col], mismatch_img)
            axs[0, col].set_title(f"Calib ({view_label})")
            col += 1

        if args.show_ped_sub:
            raw14_float = raw14.astype(np.float32)
            fm_sub = raw14_float - ped_fm
            fl_sub = raw14_float - ped_fl
            fm_sub_img = _image_view(det, evt, fm_sub, args.view_raw)
            fl_sub_img = _image_view(det, evt, fl_sub, args.view_raw)
            sub_vmin, sub_vmax = _percentile_limits(
                np.concatenate([
                    np.ravel(fm_sub_img[np.isfinite(fm_sub_img)]),
                    np.ravel(fl_sub_img[np.isfinite(fl_sub_img)]),
                ])
            )

            axs[0, col].imshow(
                fm_sub_img,
                origin="lower",
                cmap="coolwarm",
                vmin=sub_vmin,
                vmax=sub_vmax,
            )
            _overlay_mismatches(axs[0, col], mismatch_img)
            axs[0, col].set_title(f"Raw14 - ped FM[{args.fm_ped_index}]")
            col += 1

            axs[0, col].imshow(
                fl_sub_img,
                origin="lower",
                cmap="coolwarm",
                vmin=sub_vmin,
                vmax=sub_vmax,
            )
            _overlay_mismatches(axs[0, col], mismatch_img)
            axs[0, col].set_title(f"Raw14 - ped FL[{args.fl_ped_index}]")
            col += 1

        _discrete_image(
            axs[0, col],
            top2_img,
            labels=TOP2_NAMES,
            colors=TOP2_COLORS,
            title=f"Top 2 Raw Bits ({view_label})",
        )
        _overlay_mismatches(axs[0, col], mismatch_img)

        for ax in axs.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        if mismatch_img is not None:
            handles, labels = axs[0, -1].get_legend_handles_labels()
            if handles:
                fig.legend(
                    handles[:2],
                    labels[:2],
                    loc="upper right",
                    bbox_to_anchor=(0.995, 0.995),
                    frameon=True,
                )

        plt.tight_layout()
        plt.show()
        run.terminate()
        return

    raise RuntimeError(f"did not find valid event index {args.evt_idx} within {args.max_events} events")


if __name__ == "__main__":
    main()
