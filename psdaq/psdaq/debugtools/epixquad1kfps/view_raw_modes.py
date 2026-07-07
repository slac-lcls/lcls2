import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from psdaq.configdb.epixquad_layout import (
    DETECTOR_VIEW_SHAPE,
    EPIXVIEWER_DECODED_SHAPE,
    RAW_SHAPE,
    detector_view_to_raw,
    epixviewer_decoded_to_daq_raw,
    raw_detector_view,
)

mpl.rcParams["font.size"] = 8

TOP2_NAMES = ("00", "01", "10", "11")
TOP2_COLORS = ("#2f2f2f", "#1f78b4", "#e66101", "#984ea3")
MISMATCH_COLORS = {"FP": "#ff2d2d", "FN": "#00d5ff"}
MISMATCH_NAMES = ("ok", "FP", "FN")
MISMATCH_PANEL_COLORS = ("#202020", MISMATCH_COLORS["FP"], MISMATCH_COLORS["FN"])

GAINBIT_MASK = np.uint16(0x4000)
TILED_USABLE_SHAPE = DETECTOR_VIEW_SHAPE
DECODED_FULL_SHAPE = EPIXVIEWER_DECODED_SHAPE


def _parse_vc(value):
    if str(value).lower() in ("any", "all", "*"):
        return None
    return int(value, 0)


def _parse_expected_gainbit(value):
    bit = int(value, 0)
    if bit not in (0, 1):
        raise argparse.ArgumentTypeError("expected gainbit must be 0 or 1")
    return bool(bit)


def parse_args():
    parser = argparse.ArgumentParser(
        description="View epixquad raw, calib, and raw upper bits."
    )
    parser.add_argument("-e", "--exp", help="Experiment name")
    parser.add_argument("-r", "--run", type=int, help="Run number")
    parser.add_argument(
        "--dat-file",
        type=Path,
        default=None,
        help="Optional Rogue StreamWriter .dat/.data capture to decode instead of psana DataSource",
    )
    parser.add_argument("--max-events", type=int, default=100, help="Maximum events/frames to scan")
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
    parser.add_argument(
        "--expected-gainbit",
        type=_parse_expected_gainbit,
        default=None,
        help="For --dat-file: constant expected raw bit-14 value, 0 or 1",
    )
    parser.add_argument(
        "--expected-gainbit-map",
        "--expected-gainbit-npy",
        dest="expected_gainbit_map",
        type=Path,
        default=None,
        help=(
            "For --dat-file: optional .npy expected bit-14 map; accepts DAQ raw "
            "(4,352,384), detector-view tiled (704,768), or ePixViewer decoded (712,768)"
        ),
    )
    parser.add_argument(
        "--data-channel",
        type=int,
        default=1,
        help="For --dat-file: Rogue file channel to decode; ePixQuad VC0 images use channel 1",
    )
    parser.add_argument(
        "--vc",
        type=_parse_vc,
        default=None,
        help="For --dat-file: optional payload byte0 low-nibble VC filter; default any",
    )
    parser.add_argument(
        "--camera",
        default="ePixQuad",
        help="For --dat-file: ePixViewer camera decoder name",
    )
    parser.add_argument(
        "--bit-mask",
        type=lambda value: int(value, 0),
        default=0x7FFF,
        help="For --dat-file: decoder bit mask; 0x7fff keeps bit 14 and masks sign bit",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path for the plot",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive plot window",
    )
    args = parser.parse_args()

    if args.dat_file is None:
        if not args.exp or args.run is None:
            parser.error("DataSource mode requires --exp and --run, or use --dat-file")
    else:
        if args.show_calib or args.show_ped_sub:
            parser.error("--show-calib and --show-ped-sub require DataSource mode")
        if args.fp_fn_csv is not None:
            parser.error("--fp-fn-csv is a DataSource overlay; use --expected-gainbit/map with --dat-file")

    if args.expected_gainbit is not None and args.expected_gainbit_map is not None:
        parser.error("use only one of --expected-gainbit or --expected-gainbit-map")
    if args.dat_file is None and (
        args.expected_gainbit is not None or args.expected_gainbit_map is not None
    ):
        parser.error("--expected-gainbit/map are only used with --dat-file")
    if args.max_events <= 0:
        parser.error("--max-events must be positive")
    if args.evt_idx < 0:
        parser.error("--evt-idx must be >= 0")
    return args


def _mask_view(mask, *, det, evt, use_raw_view):
    if mask.ndim == 3 and mask.shape == (4, 352, 384):
        mask = mask.astype(np.uint8)
        return raw_detector_view(mask) if use_raw_view else det.raw.image(evt, mask)
    if mask.ndim == 2:
        return mask
    raise ValueError(f"unsupported mask shape: {mask.shape}")


def _image_view(det, evt, arr, use_raw_view):
    return raw_detector_view(arr) if use_raw_view else det.raw.image(evt, arr)


def _tiled_to_raw_detector(tiled):
    tiled = np.asarray(tiled)
    if tiled.shape != TILED_USABLE_SHAPE:
        raise ValueError(f"expected detector-view tiled shape {TILED_USABLE_SHAPE}, got {tiled.shape}")
    return detector_view_to_raw(tiled)


def _decoded_to_raw_detector(decoded):
    return epixviewer_decoded_to_daq_raw(decoded)


def _load_expected_gainbit_raw(args):
    if args.expected_gainbit is not None:
        return np.full(RAW_SHAPE, args.expected_gainbit, dtype=bool), f"constant {int(args.expected_gainbit)}"

    if args.expected_gainbit_map is None:
        return None, None

    expected = np.load(args.expected_gainbit_map).astype(bool, copy=False)
    if expected.shape == RAW_SHAPE:
        return expected, f"{args.expected_gainbit_map} DAQ raw shape {expected.shape}"
    if expected.shape == TILED_USABLE_SHAPE:
        return _tiled_to_raw_detector(expected), (
            f"{args.expected_gainbit_map} detector-view tiled shape {expected.shape}"
        )
    if expected.shape == DECODED_FULL_SHAPE:
        return _decoded_to_raw_detector(expected), (
            f"{args.expected_gainbit_map} ePixViewer decoded shape {expected.shape}"
        )
    raise ValueError(
        "Expected gainbit map has unsupported shape "
        f"{expected.shape}; expected {RAW_SHAPE}, {TILED_USABLE_SHAPE}, or {DECODED_FULL_SHAPE}"
    )


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


def _payload_bytes(data):
    return np.asarray(data).view(np.uint8).tobytes()


def _top2(raw):
    return ((raw >> np.uint16(14)) & np.uint16(0x3)).astype(np.uint8)


def _counter_update(counter, values):
    unique, counts = np.unique(values, return_counts=True)
    for value, count in zip(unique, counts):
        counter[int(value)] += int(count)


def _load_rogue_modules():
    try:
        from psdaq.utils import enable_epix_quad1kfps  # noqa: F401
    except Exception:
        # The DAQ environment may already have the required paths.
        pass

    try:
        import ePixViewer.Cameras as cameras
        from pyrogue.utilities.fileio import FileReader
    except Exception as exc:
        print("Failed to import Rogue/ePixQuad reader modules.", file=sys.stderr)
        print("Source setup_env.sh first and confirm SUBMODULEDIR is set.", file=sys.stderr)
        print(f"SUBMODULEDIR={os.environ.get('SUBMODULEDIR', '<unset>')}", file=sys.stderr)
        print(f"Import error: {exc!r}", file=sys.stderr)
        raise

    return cameras, FileReader


def _decode_dat_frames(args):
    if not args.dat_file.exists():
        raise FileNotFoundError(f"Data file does not exist: {args.dat_file}")

    cameras, FileReader = _load_rogue_modules()
    cam = cameras.Camera(cameraType=args.camera)
    cam.bitMask = np.uint16(args.bit_mask)
    decoded_shape = (cam.sensorHeight, cam.sensorWidth)
    min_image_payload = 32 + (cam.sensorHeight * cam.sensorWidth * np.dtype(np.uint16).itemsize)

    expected_raw, expected_source = _load_expected_gainbit_raw(args)
    fp_occurrences = np.zeros(RAW_SHAPE, dtype=np.uint32) if expected_raw is not None else None
    fn_occurrences = np.zeros(RAW_SHAPE, dtype=np.uint32) if expected_raw is not None else None

    file_channels = Counter()
    vc_counts = Counter()
    payload_sizes = Counter()
    top2_counter = Counter()
    decoded_frames = []
    selected = 0
    short_records = 0
    image_sized_records = 0
    decode_errors = 0

    reader = FileReader(str(args.dat_file))
    for header, data in reader.records():
        file_channels[int(header.channel)] += 1
        payload = _payload_bytes(data)
        payload_sizes[len(payload)] += 1
        if not payload:
            continue

        vc = payload[0] & 0xF
        vc_counts[vc] += 1
        if int(header.channel) != args.data_channel or (args.vc is not None and vc != args.vc):
            continue

        selected += 1
        if len(payload) < min_image_payload:
            short_records += 1
            continue

        image_sized_records += 1
        if len(decoded_frames) >= args.max_events:
            continue

        try:
            _, ready, raw_frame = cam.buildImageFrame(currentRawData=[], newRawData=bytearray(payload))
            if not ready:
                continue
            image = cam.descrambleImage(bytearray(raw_frame))
        except Exception as exc:
            decode_errors += 1
            if decode_errors <= 5:
                print(f"Decode error for record {reader.totCount}: {exc}", file=sys.stderr)
            continue

        image_u16 = np.asarray(image, dtype=np.uint16)
        if image_u16.shape != decoded_shape:
            decode_errors += 1
            if decode_errors <= 5:
                print(
                    f"Skipping non-image record {reader.totCount}: "
                    f"decoded shape {tuple(image_u16.shape)} expected {decoded_shape}",
                    file=sys.stderr,
                )
            continue

        raw = _decoded_to_raw_detector(image_u16)
        top2 = _top2(raw)
        _counter_update(top2_counter, top2)
        if expected_raw is not None:
            gainbit = (raw & GAINBIT_MASK) != 0
            false_positive = gainbit & ~expected_raw
            false_negative = expected_raw & ~gainbit
            np.add(fp_occurrences, false_positive, out=fp_occurrences, casting="unsafe")
            np.add(fn_occurrences, false_negative, out=fn_occurrences, casting="unsafe")
        decoded_frames.append(raw)

    if not decoded_frames:
        raise RuntimeError(
            f"No decoded image frames found in {args.dat_file}. "
            f"Minimum image payload for decoded shape {decoded_shape} is {min_image_payload} bytes."
        )
    if args.evt_idx >= len(decoded_frames):
        raise RuntimeError(
            f"--evt-idx {args.evt_idx} is outside decoded frame range 0..{len(decoded_frames) - 1}; "
            f"increase --max-events or choose a smaller index"
        )

    mismatch_labels = None
    mismatch_counts = None
    if expected_raw is not None:
        mismatch_labels = np.zeros(RAW_SHAPE, dtype=np.uint8)
        mismatch_labels[fp_occurrences > 0] = 1
        mismatch_labels[fn_occurrences > 0] = 2
        mismatch_counts = {
            "FP": int(np.count_nonzero(fp_occurrences)),
            "FN": int(np.count_nonzero(fn_occurrences)),
            "FP_occurrences": int(np.sum(fp_occurrences)),
            "FN_occurrences": int(np.sum(fn_occurrences)),
        }

    summary = {
        "decoded_shape": decoded_shape,
        "file_channels": file_channels,
        "vc_counts": vc_counts,
        "payload_sizes": payload_sizes,
        "selected_records": selected,
        "short_records": short_records,
        "image_sized_records": image_sized_records,
        "decode_errors": decode_errors,
        "decoded_frames": len(decoded_frames),
        "top2_counter": top2_counter,
        "expected_source": expected_source,
        "mismatch_counts": mismatch_counts,
    }
    return decoded_frames[args.evt_idx], expected_raw, mismatch_labels, summary


def _print_dat_summary(summary):
    print(f"decoded_shape: {summary['decoded_shape']}")
    print(f"selected_records_channel: {summary['selected_records']}")
    print(f"decoded_image_frames: {summary['decoded_frames']}")
    print(f"short_selected_records_below_image_payload: {summary['short_records']}")
    print(f"image_sized_selected_records: {summary['image_sized_records']}")
    print(f"decode_errors: {summary['decode_errors']}")
    print("top2_bit_counts ((raw >> 14) & 0x3), converted to DAQ raw layout:")
    for value in range(4):
        print(f"  {value}: {summary['top2_counter'].get(value, 0)}")
    if summary["expected_source"] is not None:
        counts = summary["mismatch_counts"]
        print("FP/FN location summary, DAQ raw layout:")
        print(f"  expected_source: {summary['expected_source']}")
        print(f"  FP unique={counts['FP']} occurrences={counts['FP_occurrences']}")
        print(f"  FN unique={counts['FN']} occurrences={counts['FN_occurrences']}")


def _finish_plot(fig, args):
    if args.output is not None:
        fig.savefig(args.output, dpi=160)
        print(f"saved plot: {args.output}")
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


def _plot_dat_frame(args, raw, expected_raw, mismatch_labels, summary):
    _print_dat_summary(summary)

    raw14 = raw & np.uint16(0x3FFF)
    top2 = _top2(raw)
    raw_img = raw_detector_view(raw14)
    top2_img = raw_detector_view(top2)
    mask_img = None if expected_raw is None else raw_detector_view(expected_raw.astype(np.uint8))
    mismatch_img = None if mismatch_labels is None else raw_detector_view(mismatch_labels)

    ncols = 2
    if mask_img is not None:
        ncols += 1
    if mismatch_img is not None:
        ncols += 1
    fig, axs = plt.subplots(1, ncols, squeeze=False, figsize=(4.0 * ncols, 4.2))

    col = 0
    if mask_img is not None:
        axs[0, col].imshow(mask_img, origin="lower", cmap="gray_r", vmin=0, vmax=1)
        _overlay_mismatches(axs[0, col], mismatch_img)
        axs[0, col].set_title("Expected gainbit")
        col += 1

    vmin, vmax = _percentile_limits(raw14)
    axs[0, col].imshow(raw_img, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    _overlay_mismatches(axs[0, col], mismatch_img)
    axs[0, col].set_title(f"Raw14 frame {args.evt_idx} (raw detector view)")
    col += 1

    if mismatch_img is not None:
        _discrete_image(
            axs[0, col],
            mismatch_img,
            labels=MISMATCH_NAMES,
            colors=MISMATCH_PANEL_COLORS,
            title=f"FP/FN locations across {summary['decoded_frames']} frames",
        )
        col += 1

    _discrete_image(
        axs[0, col],
        top2_img,
        labels=TOP2_NAMES,
        colors=TOP2_COLORS,
        title="Top 2 Raw Bits (raw detector view)",
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
    _finish_plot(fig, args)


def main():
    args = parse_args()

    if args.dat_file is not None:
        raw, expected_raw, mismatch_labels, summary = _decode_dat_frames(args)
        _plot_dat_frame(args, raw, expected_raw, mismatch_labels, summary)
        return

    mask = None
    if args.mask_npy is not None:
        mask = np.load(args.mask_npy)

    from psana import DataSource

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

        top2 = _top2(raw)

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
        _finish_plot(fig, args)
        run.terminate()
        return

    raise RuntimeError(f"did not find valid event index {args.evt_idx} within {args.max_events} events")


if __name__ == "__main__":
    main()
