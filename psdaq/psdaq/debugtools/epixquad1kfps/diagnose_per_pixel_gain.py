#!/usr/bin/env python3

"""Diagnose epixquad1kfps per-pixel fixed-gain readback.

The input gain map is expected in raw detector coordinates, matching
``det.raw.raw(evt)`` with shape ``(4, 352, 384)``. Map value 1 means fixed low
and map value 0 means fixed medium.
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from psana import DataSource

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from psdaq.configdb.epixquad_layout import RAW_ASIC_LAYOUT, RAW_SHAPE


def _parse_int_list(text):
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Compare epixquad1kfps raw top-bit gain codes against a binary FL/FM map."
    )
    parser.add_argument("-e", "--exp", required=True, help="Experiment name")
    parser.add_argument(
        "-r",
        "--runs",
        required=True,
        help="Comma-separated run numbers to inspect",
    )
    parser.add_argument(
        "--map",
        required=True,
        type=Path,
        help="Binary raw-view .npy gain map, shape (4,352,384); 1=FL, 0=FM",
    )
    parser.add_argument(
        "--detector",
        default="epixquad1kfps",
        help="Detector name passed to run.Detector()",
    )
    parser.add_argument(
        "--events",
        type=int,
        default=50,
        help="Number of valid raw events to inspect per run",
    )
    parser.add_argument(
        "--fm-code",
        type=int,
        default=None,
        help="Expected raw top2 code for map value 0/FM. Default: infer from first event.",
    )
    parser.add_argument(
        "--fl-code",
        type=int,
        default=None,
        help="Expected raw top2 code for map value 1/FL. Default: infer from first event.",
    )
    parser.add_argument(
        "--xtc-dir",
        default=None,
        help="Optional XTC directory for psana DataSource.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output for FP/FN pixels.",
    )
    parser.add_argument(
        "--max-print",
        type=int,
        default=0,
        help="Maximum FP/FN rows to print per run. Use 0 for all rows.",
    )
    parser.add_argument(
        "--no-value-analysis",
        action="store_true",
        help="Disable raw14 mean/RMS extraction and FM/FL-like noise classification.",
    )
    parser.add_argument(
        "--min-ref-pixels",
        type=int,
        default=100,
        help="Minimum agreed FM/FL reference pixels for local value classification.",
    )
    parser.add_argument(
        "--no-pedestal-analysis",
        action="store_true",
        help="Disable FM/FL pedestal residual classification.",
    )
    parser.add_argument(
        "--fm-ped-index",
        type=int,
        default=1,
        help="Pedestal gain index for FM. Default 1 for ePix10ka gain order FH,FM,FL,...",
    )
    parser.add_argument(
        "--fl-ped-index",
        type=int,
        default=2,
        help="Pedestal gain index for FL. Default 2 for ePix10ka gain order FH,FM,FL,...",
    )
    return parser.parse_args()


def _load_map(path):
    gain_map = np.load(path)
    if gain_map.shape != RAW_SHAPE:
        raise ValueError(f"expected map shape {RAW_SHAPE}, got {gain_map.shape} from {path}")
    labels = set(int(v) for v in np.unique(gain_map))
    if not labels.issubset({0, 1}):
        raise ValueError(f"expected binary map labels 0/1, got {sorted(labels)}")
    return np.asarray(gain_map, dtype=bool)


def _top2(raw):
    return ((raw >> 14) & 0x3).astype(np.uint8)


def _valid_raw_images(run, det, limit):
    event_indices = []
    top2_images = []
    raw14_images = []
    for event_index, evt in enumerate(run.events()):
        if len(top2_images) >= limit:
            break
        raw = det.raw.raw(evt)
        if raw is None:
            continue
        event_indices.append(int(event_index))
        top2_images.append(_top2(raw))
        raw14_images.append((raw & 0x3FFF).astype(np.uint16))
    return event_indices, top2_images, raw14_images


def _infer_code(top2, mask):
    codes = {}
    for label, name in ((0, "FM"), (1, "FL")):
        region = top2[mask == bool(label)]
        values, counts = np.unique(region, return_counts=True)
        order = np.argsort(counts)[::-1]
        codes[name] = int(values[order[0]])
    return codes


def _raw_to_asic_bank_coord(segment, row, col):
    for layout in RAW_ASIC_LAYOUT:
        r0, r1 = layout["row_slice"]
        c0, c1 = layout["col_slice"]
        if not (r0 <= row < r1 and c0 <= col < c1):
            continue

        raw_local_row = int(row - r0)
        raw_local_col = int(col - c0)
        if layout["operator"] == "identity":
            prog_row = raw_local_row
            prog_col = raw_local_col
        elif layout["operator"] == "rot180":
            prog_row = 175 - raw_local_row
            prog_col = 191 - raw_local_col
        else:
            raise ValueError(f"unsupported ASIC operator {layout['operator']!r}")

        return {
            "asic": int(4 * segment + layout["slot"]),
            "bank": int(prog_col // 48),
            "bank_row": int(prog_row),
            "bank_col": int(prog_col % 48),
            "segment": int(segment),
            "raw_row": int(row),
            "raw_col": int(col),
        }

    raise ValueError(f"raw coordinate outside ASIC layout: segment={segment} row={row} col={col}")


def _asic_bank_mask(asic, bank):
    if not (0 <= int(asic) < 16):
        raise ValueError(f"invalid ASIC index {asic}")
    if not (0 <= int(bank) < 4):
        raise ValueError(f"invalid bank index {bank}")

    segment = int(asic) // 4
    slot = int(asic) % 4
    mask = np.zeros(RAW_SHAPE, dtype=bool)
    layout = next(layout for layout in RAW_ASIC_LAYOUT if int(layout["slot"]) == slot)
    r0, r1 = layout["row_slice"]
    c0, c1 = layout["col_slice"]
    raw_local_cols = np.arange(c1 - c0)
    if layout["operator"] == "identity":
        prog_cols = raw_local_cols
    elif layout["operator"] == "rot180":
        prog_cols = (c1 - c0 - 1) - raw_local_cols
    else:
        raise ValueError(f"unsupported ASIC operator {layout['operator']!r}")

    bank_cols = raw_local_cols[(prog_cols // 48) == int(bank)]
    mask[segment, r0:r1, c0 + bank_cols] = True
    return mask


def _mismatch_rows(run_number, first_top2, mask, fm_code, fl_code):
    expected = np.where(mask, fl_code, fm_code).astype(np.uint8)
    fp = (first_top2 == fl_code) & ~mask
    fn = mask & (first_top2 != fl_code)
    rows = []
    for kind, mismatch_mask in (("FP", fp), ("FN", fn)):
        for segment, row, col in np.argwhere(mismatch_mask):
            coord = _raw_to_asic_bank_coord(int(segment), int(row), int(col))
            observed = int(first_top2[segment, row, col])
            rows.append(
                {
                    "run": int(run_number),
                    "kind": kind,
                    "asic": coord["asic"],
                    "bank": coord["bank"],
                    "bank_row": coord["bank_row"],
                    "bank_col": coord["bank_col"],
                    "segment": coord["segment"],
                    "raw_row": coord["raw_row"],
                    "raw_col": coord["raw_col"],
                    "expected_top2": int(expected[segment, row, col]),
                    "observed_top2": observed,
                }
            )
    rows.sort(key=lambda r: (r["kind"], r["asic"], r["bank"], r["bank_row"], r["bank_col"]))
    return rows


def _robust_center_scale(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None, None
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale = 1.4826 * mad
    if scale <= 0:
        scale = float(np.std(values))
    if scale <= 0:
        scale = 1.0
    return center, scale


def _add_pedestal_residual(row, stack, ped_fm, ped_fl):
    segment = int(row["segment"])
    raw_row = int(row["raw_row"])
    raw_col = int(row["raw_col"])
    values = stack[:, segment, raw_row, raw_col]
    fm_pedestal = float(ped_fm[segment, raw_row, raw_col])
    fl_pedestal = float(ped_fl[segment, raw_row, raw_col])
    fm_residual = values - fm_pedestal
    fl_residual = values - fl_pedestal
    fm_abs_score = float(np.median(np.abs(fm_residual)))
    fl_abs_score = float(np.median(np.abs(fl_residual)))

    row["ped_fm"] = "%.3f" % fm_pedestal
    row["ped_fl"] = "%.3f" % fl_pedestal
    row["fm_ped_resid_median"] = "%.3f" % float(np.median(fm_residual))
    row["fl_ped_resid_median"] = "%.3f" % float(np.median(fl_residual))
    row["fm_ped_abs_resid_median"] = "%.3f" % fm_abs_score
    row["fl_ped_abs_resid_median"] = "%.3f" % fl_abs_score
    row["pedestal_score_fm"] = "%.3f" % fm_abs_score
    row["pedestal_score_fl"] = "%.3f" % fl_abs_score
    row["pedestal_like_gain"] = "FM" if fm_abs_score <= fl_abs_score else "FL"


def _blank_pedestal_residual(row):
    row["ped_fm"] = ""
    row["ped_fl"] = ""
    row["fm_ped_resid_median"] = ""
    row["fl_ped_resid_median"] = ""
    row["fm_ped_abs_resid_median"] = ""
    row["fl_ped_abs_resid_median"] = ""
    row["pedestal_score_fm"] = ""
    row["pedestal_score_fl"] = ""
    row["pedestal_like_gain"] = "not_run"


def _reference_mask(first_top2, gain_map, fm_code, fl_code, scope_mask, gain_name):
    if gain_name == "FM":
        return scope_mask & ~gain_map & (first_top2 == fm_code)
    if gain_name == "FL":
        return scope_mask & gain_map & (first_top2 == fl_code)
    raise ValueError(f"unsupported gain name {gain_name!r}")


def _scope_for_row(row, first_top2, gain_map, fm_code, fl_code, min_ref_pixels, bank_masks):
    same_bank = bank_masks[(row["asic"], row["bank"])]
    for scope_name, scope_mask in (
        ("same_asic_bank", same_bank),
        ("same_detector", np.ones(RAW_SHAPE, dtype=bool)),
    ):
        fm_count = int(np.count_nonzero(_reference_mask(first_top2, gain_map, fm_code, fl_code, scope_mask, "FM")))
        fl_count = int(np.count_nonzero(_reference_mask(first_top2, gain_map, fm_code, fl_code, scope_mask, "FL")))
        if fm_count >= min_ref_pixels and fl_count >= min_ref_pixels:
            return scope_name, scope_mask, fm_count, fl_count

    return "insufficient_reference", same_bank, 0, 0


def _add_value_analysis(
    rows,
    raw14_images,
    first_top2,
    gain_map,
    fm_code,
    fl_code,
    min_ref_pixels,
    ped_fm=None,
    ped_fl=None,
):
    if not rows or not raw14_images:
        return

    stack = np.stack(raw14_images).astype(np.float32)
    mean = stack.mean(axis=0)
    rms = stack.std(axis=0, ddof=1) if stack.shape[0] > 1 else np.zeros(RAW_SHAPE, dtype=np.float32)

    bank_masks = {
        (int(row["asic"]), int(row["bank"])): _asic_bank_mask(row["asic"], row["bank"])
        for row in rows
    }
    reference_cache = {}

    for row in rows:
        segment = int(row["segment"])
        raw_row = int(row["raw_row"])
        raw_col = int(row["raw_col"])
        row["raw14_first"] = int(stack[0, segment, raw_row, raw_col])
        row["raw14_mean"] = "%.3f" % float(mean[segment, raw_row, raw_col])
        row["raw14_rms"] = "%.3f" % float(rms[segment, raw_row, raw_col])
        if ped_fm is not None and ped_fl is not None:
            _add_pedestal_residual(row, stack, ped_fm, ped_fl)
        else:
            _blank_pedestal_residual(row)

        scope_name, scope_mask, fm_count, fl_count = _scope_for_row(
            row,
            first_top2,
            gain_map,
            fm_code,
            fl_code,
            min_ref_pixels,
            bank_masks,
        )
        row["value_ref_scope"] = scope_name
        row["fm_ref_pixels"] = int(fm_count)
        row["fl_ref_pixels"] = int(fl_count)

        if scope_name == "insufficient_reference":
            row["fm_ref_rms_median"] = ""
            row["fl_ref_rms_median"] = ""
            row["noise_like_gain"] = "unknown"
            row["noise_score_fm"] = ""
            row["noise_score_fl"] = ""
            continue

        cache_key = (scope_name, int(row["asic"]), int(row["bank"]))
        if scope_name == "same_detector":
            cache_key = (scope_name, -1, -1)

        if cache_key not in reference_cache:
            fm_mask = _reference_mask(first_top2, gain_map, fm_code, fl_code, scope_mask, "FM")
            fl_mask = _reference_mask(first_top2, gain_map, fm_code, fl_code, scope_mask, "FL")
            fm_center, fm_scale = _robust_center_scale(rms[fm_mask])
            fl_center, fl_scale = _robust_center_scale(rms[fl_mask])
            reference_cache[cache_key] = (fm_center, fm_scale, fl_center, fl_scale)

        fm_center, fm_scale, fl_center, fl_scale = reference_cache[cache_key]
        pixel_rms = float(rms[segment, raw_row, raw_col])
        fm_score = abs(pixel_rms - fm_center) / fm_scale
        fl_score = abs(pixel_rms - fl_center) / fl_scale
        row["fm_ref_rms_median"] = "%.3f" % fm_center
        row["fl_ref_rms_median"] = "%.3f" % fl_center
        row["noise_score_fm"] = "%.3f" % fm_score
        row["noise_score_fl"] = "%.3f" % fl_score
        row["noise_like_gain"] = "FM" if fm_score <= fl_score else "FL"


def _print_rows(rows, max_print):
    if not rows:
        return
    selected = rows if max_print == 0 else rows[:max_print]
    value_fields = "raw14_mean" in rows[0]
    header = "kind asic bank bank_row bank_col observed_top2 expected_top2 segment raw_row raw_col"
    if value_fields:
        header += " raw14_mean raw14_rms noise_like_gain pedestal_like_gain value_ref_scope"
    print(header)
    for row in selected:
        line = (
            "{kind:>2} {asic:>4} {bank:>4} {bank_row:>8} {bank_col:>8} "
            "{observed_top2:>13} {expected_top2:>13} {segment:>7} {raw_row:>7} {raw_col:>7}".format(
                **row
            )
        )
        if value_fields:
            line += (
                " {raw14_mean:>10} {raw14_rms:>9} {noise_like_gain:>15}"
                " {pedestal_like_gain:>18} {value_ref_scope:>15}"
            ).format(**row)
        print(line)
    if len(selected) != len(rows):
        print(f"... {len(rows) - len(selected)} rows omitted by --max-print")


def _write_csv(path, rows):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run",
        "kind",
        "asic",
        "bank",
        "bank_row",
        "bank_col",
        "segment",
        "raw_row",
        "raw_col",
        "expected_top2",
        "observed_top2",
        "raw14_first",
        "raw14_mean",
        "raw14_rms",
        "noise_like_gain",
        "noise_score_fm",
        "noise_score_fl",
        "value_ref_scope",
        "fm_ref_pixels",
        "fl_ref_pixels",
        "fm_ref_rms_median",
        "fl_ref_rms_median",
        "ped_fm",
        "ped_fl",
        "fm_ped_resid_median",
        "fl_ped_resid_median",
        "fm_ped_abs_resid_median",
        "fl_ped_abs_resid_median",
        "pedestal_like_gain",
        "pedestal_score_fm",
        "pedestal_score_fl",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = _parse_args()
    mask = _load_map(args.map)
    runs = _parse_int_list(args.runs)

    print(f"map: {args.map}")
    print(f"map_shape: {mask.shape} FL_pixels={int(mask.sum())} FM_pixels={int(mask.size - mask.sum())}")

    all_rows = []
    for run_number in runs:
        ds_kwargs = {"exp": args.exp, "run": run_number, "detectors": [args.detector]}
        if args.xtc_dir:
            ds_kwargs["dir"] = args.xtc_dir
        ds = DataSource(**ds_kwargs)
        run = next(ds.runs())
        det = run.Detector(args.detector)
        ped_fm = None
        ped_fl = None
        if not args.no_pedestal_analysis:
            pedestals = det.raw._pedestals()
            if pedestals is None:
                print(f"\nrun {run_number}: pedestal analysis disabled because det.raw._pedestals() returned None")
            elif pedestals.shape[1:] != RAW_SHAPE:
                print(
                    f"\nrun {run_number}: pedestal analysis disabled because pedestal shape "
                    f"{pedestals.shape} is not compatible with raw shape {RAW_SHAPE}"
                )
            elif max(args.fm_ped_index, args.fl_ped_index) >= pedestals.shape[0]:
                print(
                    f"\nrun {run_number}: pedestal analysis disabled because pedestal shape "
                    f"{pedestals.shape} does not include requested indices "
                    f"FM={args.fm_ped_index}, FL={args.fl_ped_index}"
                )
            else:
                ped_fm = np.asarray(pedestals[args.fm_ped_index], dtype=np.float32)
                ped_fl = np.asarray(pedestals[args.fl_ped_index], dtype=np.float32)

        event_indices, top2_images, raw14_images = _valid_raw_images(run, det, args.events)
        if not top2_images:
            print(f"\nrun {run_number}: no valid raw events")
            continue
        first_event_index = event_indices[0]
        first_top2 = top2_images[0]
        inferred = _infer_code(first_top2, mask)
        fm_code = inferred["FM"] if args.fm_code is None else int(args.fm_code)
        fl_code = inferred["FL"] if args.fl_code is None else int(args.fl_code)

        diff_counts = [int(np.count_nonzero(img != first_top2)) for img in top2_images]
        stable = all(count == 0 for count in diff_counts)

        counts = Counter(int(v) for v in first_top2.ravel())
        rows = _mismatch_rows(run_number, first_top2, mask, fm_code, fl_code)
        if not args.no_value_analysis:
            _add_value_analysis(
                rows,
                raw14_images,
                first_top2,
                mask,
                fm_code,
                fl_code,
                args.min_ref_pixels,
                ped_fm=ped_fm,
                ped_fl=ped_fl,
            )
        kind_counts = Counter(row["kind"] for row in rows)
        noise_counts = Counter(row.get("noise_like_gain", "not_run") for row in rows)
        pedestal_counts = Counter(row.get("pedestal_like_gain", "not_run") for row in rows)
        all_rows.extend(rows)

        print(f"\nrun {run_number}:")
        print(f"  first_valid_event_index: {first_event_index}")
        print(f"  top2_counts_first_event: {dict(sorted(counts.items()))}")
        print(f"  inferred_codes: FM={inferred['FM']} FL={inferred['FL']}  used_codes: FM={fm_code} FL={fl_code}")
        if stable:
            print(f"  first_{len(top2_images)}_gainbit_images_same_as_first: True")
        else:
            print(f"  first_{len(top2_images)}_gainbit_images_same_as_first: False")
            print(f"  pixels_different_from_first_by_event: {diff_counts}")
        print(f"  FP_pixels: {kind_counts.get('FP', 0)}")
        print(f"  FN_pixels: {kind_counts.get('FN', 0)}")
        if not args.no_value_analysis:
            print(f"  noise_like_gain_counts_for_mismatches: {dict(sorted(noise_counts.items()))}")
            if not args.no_pedestal_analysis and ped_fm is not None and ped_fl is not None:
                print(f"  pedestal_like_gain_counts_for_mismatches: {dict(sorted(pedestal_counts.items()))}")
        _print_rows(rows, args.max_print)
        run.terminate()

    _write_csv(args.csv, all_rows)
    if args.csv is not None:
        print(f"\nwrote_csv: {args.csv}")


if __name__ == "__main__":
    main()
