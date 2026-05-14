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


def _valid_top2_images(run, det, limit):
    event_indices = []
    images = []
    for event_index, evt in enumerate(run.events()):
        if len(images) >= limit:
            break
        raw = det.raw.raw(evt)
        if raw is None:
            continue
        event_indices.append(int(event_index))
        images.append(_top2(raw))
    return event_indices, images


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


def _print_rows(rows, max_print):
    if not rows:
        return
    selected = rows if max_print == 0 else rows[:max_print]
    print("kind asic bank bank_row bank_col observed_top2 expected_top2 segment raw_row raw_col")
    for row in selected:
        print(
            "{kind:>2} {asic:>4} {bank:>4} {bank_row:>8} {bank_col:>8} "
            "{observed_top2:>13} {expected_top2:>13} {segment:>7} {raw_row:>7} {raw_col:>7}".format(
                **row
            )
        )
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

        event_indices, images = _valid_top2_images(run, det, args.events)
        if not images:
            print(f"\nrun {run_number}: no valid raw events")
            continue
        first_event_index = event_indices[0]
        first_top2 = images[0]
        inferred = _infer_code(first_top2, mask)
        fm_code = inferred["FM"] if args.fm_code is None else int(args.fm_code)
        fl_code = inferred["FL"] if args.fl_code is None else int(args.fl_code)

        diff_counts = [int(np.count_nonzero(img != first_top2)) for img in images]
        stable = all(count == 0 for count in diff_counts)

        counts = Counter(int(v) for v in first_top2.ravel())
        rows = _mismatch_rows(run_number, first_top2, mask, fm_code, fl_code)
        kind_counts = Counter(row["kind"] for row in rows)
        all_rows.extend(rows)

        print(f"\nrun {run_number}:")
        print(f"  first_valid_event_index: {first_event_index}")
        print(f"  top2_counts_first_event: {dict(sorted(counts.items()))}")
        print(f"  inferred_codes: FM={inferred['FM']} FL={inferred['FL']}  used_codes: FM={fm_code} FL={fl_code}")
        if stable:
            print(f"  first_{len(images)}_gainbit_images_same_as_first: True")
        else:
            print(f"  first_{len(images)}_gainbit_images_same_as_first: False")
            print(f"  pixels_different_from_first_by_event: {diff_counts}")
        print(f"  FP_pixels: {kind_counts.get('FP', 0)}")
        print(f"  FN_pixels: {kind_counts.get('FN', 0)}")
        _print_rows(rows, args.max_print)
        run.terminate()

    _write_csv(args.csv, all_rows)
    if args.csv is not None:
        print(f"\nwrote_csv: {args.csv}")


if __name__ == "__main__":
    main()
