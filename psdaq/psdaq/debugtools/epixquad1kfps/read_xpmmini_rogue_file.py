#!/usr/bin/env python3

"""Read ePixQuad Rogue StreamWriter data and summarize gainbit values."""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from psdaq.configdb.epixquad_layout import (
    DAQ_RAW_FRAME_BYTES,
    DETECTOR_VIEW_SHAPE,
    EPIXVIEWER_DECODED_SHAPE,
    RAW_ASIC_LAYOUT,
    RAW_SHAPE,
    detector_view_to_raw,
    epixviewer_decoded_to_daq_raw,
    rogue_payload_to_daq_raw,
)

GAINBIT_MASK = np.uint16(0x4000)
RAW_VIEW_SHAPE = RAW_SHAPE
TILED_VIEW_SHAPE = DETECTOR_VIEW_SHAPE
DECODED_FULL_SHAPE = EPIXVIEWER_DECODED_SHAPE


def _parse_vc(value):
    if str(value).lower() in ("any", "all", "*"):
        return None
    return int(value, 0)


def _parse_expected_gainbit(value):
    bit = int(value, 0)
    if bit not in (0, 1):
        raise argparse.ArgumentTypeError("--expected-gainbit must be 0 or 1")
    return bool(bit)


def _parse_status_indices(text, nindices):
    if str(text).lower() in ("any", "all", "*"):
        return None

    indices = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        index = int(item, 0)
        if index < 0 or index >= nindices:
            raise ValueError(
                f"pixel status index {index} is outside available range 0-{nindices - 1}"
            )
        indices.append(index)
    if not indices:
        raise ValueError("--pixel-status-indices did not contain any indices")
    return sorted(set(indices))


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Decode ePixQuad Rogue .dat/.data captures and summarize bit 14 gainbit."
    )
    parser.add_argument("data_file", help="Rogue StreamWriter output file")
    parser.add_argument(
        "--data-channel",
        type=int,
        default=1,
        help="Rogue file channel for VC0 image data; ePixQuad.Top writes VC0 to channel 1",
    )
    parser.add_argument(
        "--vc",
        type=_parse_vc,
        default=None,
        help="optional payload byte0 low-nibble filter; use any to disable, default any",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=20,
        help="maximum decoded image frames to summarize",
    )
    parser.add_argument(
        "--camera",
        default="ePixQuad",
        help="ePixViewer camera decoder name",
    )
    parser.add_argument(
        "--bit-mask",
        type=lambda value: int(value, 0),
        default=0x7FFF,
        help="decoder bit mask; 0x7fff keeps bit 14 gainbit and masks sign bit",
    )
    parser.add_argument(
        "--save-prefix",
        help="optional prefix for saving first DAQ raw/gainbit arrays as .npy",
    )
    parser.add_argument(
        "--expected-gainbit",
        type=_parse_expected_gainbit,
        default=None,
        help="constant expected bit14 value for every DAQ raw pixel, 0 or 1",
    )
    parser.add_argument(
        "--expected-gainbit-map",
        "--expected-gainbit-npy",
        dest="expected_gainbit_map",
        default=None,
        help=(
            "optional .npy 0/1 expected bit14 map; accepts DAQ raw "
            "(4,352,384), detector-view tiled (704,768), or ePixViewer "
            "decoded (712,768)"
        ),
    )
    parser.add_argument(
        "--fpfn-max-pixels",
        type=int,
        default=32,
        help="maximum FP and FN coordinates to print in each summary; use 0 for counts only",
    )
    parser.add_argument(
        "--quiet-decode-errors",
        action="store_true",
        help="suppress individual decode error messages",
    )
    parser.add_argument(
        "--max-decode-error-reports",
        type=int,
        default=5,
        help="maximum number of per-record decode errors to print",
    )
    parser.add_argument(
        "--dump-first",
        type=int,
        default=0,
        help="dump header and leading payload bytes for the first N selected records",
    )
    parser.add_argument(
        "--pixel-status-npz",
        default=None,
        help=(
            "optional pseudo-pedestal .npz containing pixel_status or bad_mask; "
            "FP/FN pixels are grouped by whether their status is bad"
        ),
    )
    parser.add_argument(
        "--pixel-status-indices",
        default="any",
        help=(
            "comma-separated pixel_status gain indices to use, e.g. 0,1 for FH/FM; "
            "default any marks a pixel bad if any status index is nonzero"
        ),
    )
    args = parser.parse_args()
    if args.expected_gainbit is not None and args.expected_gainbit_map:
        parser.error("use only one of --expected-gainbit or --expected-gainbit-map")
    if args.fpfn_max_pixels < 0:
        parser.error("--fpfn-max-pixels must be >= 0")
    return args


def _load_modules():
    try:
        from psdaq.utils import enable_epix_quad1kfps  # noqa: F401
        from pyrogue.utilities.fileio import FileReader
    except Exception as exc:
        print("Failed to import Rogue/ePixQuad reader modules.", file=sys.stderr)
        print("Source setup_env.sh first.", file=sys.stderr)
        print(f"SUBMODULEDIR={os.environ.get('SUBMODULEDIR', '<unset>')}", file=sys.stderr)
        print(f"Import error: {exc!r}", file=sys.stderr)
        raise SystemExit(1) from exc

    return FileReader


def _payload_bytes(data):
    return np.asarray(data).view(np.uint8).tobytes()


def _counter_update(counter, values):
    unique, counts = np.unique(values, return_counts=True)
    for value, count in zip(unique, counts):
        counter[int(value)] += int(count)


def _load_expected_gainbit(args):
    if args.expected_gainbit is not None:
        expected = np.full(RAW_SHAPE, args.expected_gainbit, dtype=bool)
        return expected, f"constant {int(args.expected_gainbit)}"

    if not args.expected_gainbit_map:
        return None, None

    expected_path = args.expected_gainbit_map
    if not os.path.exists(expected_path):
        raise FileNotFoundError(f"Expected gainbit map does not exist: {expected_path}")

    expected = np.load(expected_path).astype(bool, copy=False)
    if expected.shape == RAW_SHAPE:
        return expected, f"{expected_path} DAQ raw shape {expected.shape}"

    if expected.shape == TILED_VIEW_SHAPE:
        raw = detector_view_to_raw(expected)
        return raw, f"{expected_path} detector-view tiled shape {expected.shape}"

    if expected.shape == DECODED_FULL_SHAPE:
        raw = epixviewer_decoded_to_daq_raw(expected)
        return raw, f"{expected_path} ePixViewer decoded shape {expected.shape}"

    raise ValueError(
        "Expected gainbit map has unsupported shape "
        f"{expected.shape}; expected {RAW_SHAPE}, {TILED_VIEW_SHAPE}, "
        f"or {DECODED_FULL_SHAPE}"
    )


def _status_source_label(path, key, shape, indices):
    if indices is None:
        index_text = "any nonzero status index"
    else:
        index_text = "indices " + ",".join(str(index) for index in indices)
    return f"{path} {key} shape {shape}; {index_text}"


def _load_pixel_status_bad_mask(args):
    if not args.pixel_status_npz:
        return None, None
    if not os.path.exists(args.pixel_status_npz):
        raise FileNotFoundError(f"Pixel-status npz does not exist: {args.pixel_status_npz}")

    with np.load(args.pixel_status_npz, allow_pickle=False) as data:
        if "pixel_status" in data:
            key = "pixel_status"
            status = data[key]
            bad_values = status != 0
        elif "bad_mask" in data:
            key = "bad_mask"
            status = data[key]
            bad_values = status.astype(bool, copy=False)
        else:
            raise KeyError(
                f"{args.pixel_status_npz} does not contain pixel_status or bad_mask"
            )

        if bad_values.shape == RAW_SHAPE:
            return bad_values.copy(), f"{args.pixel_status_npz} {key} shape {bad_values.shape}"

        if bad_values.ndim == 4 and bad_values.shape[1:] == RAW_SHAPE:
            indices = _parse_status_indices(args.pixel_status_indices, bad_values.shape[0])
            if indices is None:
                bad_mask = np.any(bad_values, axis=0)
            else:
                bad_mask = np.any(bad_values[indices], axis=0)
            return bad_mask.copy(), _status_source_label(
                args.pixel_status_npz,
                key,
                bad_values.shape,
                indices,
            )

        raise ValueError(
            f"{args.pixel_status_npz} {key} has unsupported shape {bad_values.shape}; "
            f"expected {RAW_SHAPE} or (N,{RAW_SHAPE[0]},{RAW_SHAPE[1]},{RAW_SHAPE[2]})"
        )


def _raw_area_masks():
    usable = np.ones(RAW_SHAPE, dtype=bool)
    control = np.zeros(RAW_SHAPE, dtype=bool)
    return usable, control


def _raw_pixel_area_text(segment, row, col):
    for layout in RAW_ASIC_LAYOUT:
        r0, r1 = layout["row_slice"]
        c0, c1 = layout["col_slice"]
        if not (r0 <= row < r1 and c0 <= col < c1):
            continue

        raw_local_row = row - r0
        raw_local_col = col - c0
        if layout["operator"] == "identity":
            bank_row = raw_local_row
            prog_col = raw_local_col
        elif layout["operator"] == "rot180":
            bank_row = (r1 - r0 - 1) - raw_local_row
            prog_col = (c1 - c0 - 1) - raw_local_col
        else:
            raise ValueError(f"unsupported raw-map operator {layout['operator']!r}")

        asic = 4 * int(segment) + int(layout["slot"])
        return (
            f"area=usable asic={asic} bank={int(prog_col // 48)} "
            f"bank_row={int(bank_row)} bank_col={int(prog_col % 48)}"
        )

    return "area=unmapped"


def _print_pixel_list(label, coords, counts, max_pixels, status_bad_mask):
    if not max_pixels or coords.size == 0:
        return

    order = np.argsort(counts)[::-1]
    print(f"  {label}, top {min(max_pixels, len(coords))} by occurrence:")
    for index in order[:max_pixels]:
        segment = int(coords[index, 0])
        row = int(coords[index, 1])
        col = int(coords[index, 2])
        count = int(counts[index])
        status_text = ""
        if status_bad_mask is not None:
            status_text = (
                " pixel_status=bad"
                if status_bad_mask[segment, row, col]
                else " pixel_status=ok"
            )
        print(
            f"    segment={segment} row={row} col={col} occurrences={count} "
            f"{_raw_pixel_area_text(segment, row, col)}{status_text}"
        )


def _print_mismatch_summary(
    label,
    occurrences,
    usable_mask,
    control_mask,
    max_pixels,
    status_bad_mask=None,
):
    total = int(np.sum(occurrences))
    seen = occurrences > 0
    unique = int(np.count_nonzero(seen))
    usable_occ = int(np.sum(occurrences[usable_mask]))
    control_occ = int(np.sum(occurrences[control_mask]))
    usable_unique = int(np.count_nonzero(seen & usable_mask))
    control_unique = int(np.count_nonzero(seen & control_mask))
    other_unique = unique - usable_unique - control_unique
    other_occ = total - usable_occ - control_occ

    print(f"{label}:")
    print(f"  total_occurrences: {total}")
    print(f"  unique_pixels: {unique}")
    print(f"  usable: occurrences={usable_occ} unique_pixels={usable_unique}")
    print(f"  control: occurrences={control_occ} unique_pixels={control_unique}")
    if other_unique or other_occ:
        print(f"  other: occurrences={other_occ} unique_pixels={other_unique}")

    if status_bad_mask is not None:
        status_bad_seen = seen & status_bad_mask
        status_ok_seen = seen & ~status_bad_mask
        status_bad_occ = int(np.sum(occurrences[status_bad_mask]))
        status_ok_occ = total - status_bad_occ
        print("  pixel_status:")
        print(
            f"    status_bad: occurrences={status_bad_occ} "
            f"unique_pixels={int(np.count_nonzero(status_bad_seen))}"
        )
        print(
            f"    status_ok: occurrences={status_ok_occ} "
            f"unique_pixels={int(np.count_nonzero(status_ok_seen))}"
        )

    if not max_pixels or unique == 0:
        return

    coords = np.argwhere(seen)
    counts = occurrences[seen]
    if status_bad_mask is None:
        _print_pixel_list("pixels", coords, counts, max_pixels, status_bad_mask)
        return

    is_status_bad = np.array(
        [status_bad_mask[int(segment), int(row), int(col)] for segment, row, col in coords],
        dtype=bool,
    )
    _print_pixel_list(
        "status_bad pixels",
        coords[is_status_bad],
        counts[is_status_bad],
        max_pixels,
        status_bad_mask,
    )
    _print_pixel_list(
        "status_ok pixels",
        coords[~is_status_bad],
        counts[~is_status_bad],
        max_pixels,
        status_bad_mask,
    )


def main():
    args = _parse_args()
    if not os.path.exists(args.data_file):
        print(f"Data file does not exist: {args.data_file}", file=sys.stderr)
        return 1

    FileReader = _load_modules()
    decoded_shape = RAW_SHAPE
    min_image_payload = DAQ_RAW_FRAME_BYTES
    expected_gainbit, expected_source = _load_expected_gainbit(args)
    pixel_status_bad_mask, pixel_status_source = _load_pixel_status_bad_mask(args)
    usable_mask, control_mask = _raw_area_masks()
    fp_occurrences = np.zeros(RAW_SHAPE, dtype=np.uint32) if expected_gainbit is not None else None
    fn_occurrences = np.zeros(RAW_SHAPE, dtype=np.uint32) if expected_gainbit is not None else None

    file_channels = Counter()
    vc_counts = Counter()
    payload_sizes = Counter()
    image_start_skip_words = Counter()
    decoded = 0
    selected = 0
    decode_errors = 0
    short_records = 0
    image_sized_records = 0
    total_gainbit_ones = 0
    total_pixels = 0
    top2_counter = Counter()
    first_raw = None
    first_gainbit = None
    first_gainbit_equal = True

    reader = FileReader(args.data_file)
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
        if selected <= args.dump_first:
            leading = payload[:64].hex(" ")
            print(
                f"selected record {selected}: "
                f"file_channel={int(header.channel)} size={len(payload)} "
                f"flags=0x{int(header.flags):04x} error={int(header.error)} "
                f"payload_vc={vc} first_bytes={leading}"
            )

        if len(payload) < min_image_payload:
            short_records += 1
            continue

        image_sized_records += 1
        if decoded >= args.max_frames:
            continue

        try:
            raw_u16, skip_words = rogue_payload_to_daq_raw(payload, bit_mask=args.bit_mask)
        except Exception as exc:
            decode_errors += 1
            if not args.quiet_decode_errors and decode_errors <= args.max_decode_error_reports:
                print(f"Decode error for record {reader.totCount}: {exc}", file=sys.stderr)
            continue

        image_start_skip_words[skip_words] += 1
        gainbit = (raw_u16 & GAINBIT_MASK) != 0
        top2 = (raw_u16 >> np.uint16(14)) & np.uint16(0x3)

        if first_raw is None:
            first_raw = raw_u16.copy()
            first_gainbit = gainbit.copy()
        elif first_gainbit_equal and not np.array_equal(first_gainbit, gainbit):
            first_gainbit_equal = False

        ones = int(np.count_nonzero(gainbit))
        pixels = int(gainbit.size)
        total_gainbit_ones += ones
        total_pixels += pixels
        _counter_update(top2_counter, top2)
        decoded += 1

        fp_count = fn_count = None
        if expected_gainbit is not None:
            false_positive = gainbit & ~expected_gainbit
            false_negative = expected_gainbit & ~gainbit
            fp_count = int(np.count_nonzero(false_positive))
            fn_count = int(np.count_nonzero(false_negative))
            np.add(fp_occurrences, false_positive, out=fp_occurrences, casting="unsafe")
            np.add(fn_occurrences, false_negative, out=fn_occurrences, casting="unsafe")

        line = (
            f"frame {decoded:04d}: raw_shape={tuple(raw_u16.shape)} "
            f"gainbit_ones={ones} gainbit_fraction={ones / pixels:.6f}"
        )
        if expected_gainbit is not None:
            line += f" false_positive={fp_count} false_negative={fn_count}"
        print(line)

    print()
    print("Rogue records by file channel:")
    for channel, count in sorted(file_channels.items()):
        print(f"  channel {channel}: {count}")

    print("Payload VC counts from byte0 low nibble:")
    for vc, count in sorted(vc_counts.items()):
        print(f"  vc {vc}: {count}")
    print("Payload sizes:")
    for size, count in sorted(payload_sizes.items()):
        print(f"  {size} bytes: {count}")
    if image_start_skip_words:
        print("Image start offsets:")
        for skip_words, count in sorted(image_start_skip_words.items()):
            print(f"  {skip_words} uint16 words ({skip_words * 2} bytes): {count}")

    print()
    vc_label = "any" if args.vc is None else str(args.vc)
    print(f"selected_records_channel_{args.data_channel}_vc_{vc_label}: {selected}")
    print(f"decoded_image_frames: {decoded}")
    print(f"short_selected_records_below_image_payload: {short_records}")
    print(f"image_sized_selected_records: {image_sized_records}")
    print(f"decode_errors: {decode_errors}")
    if decoded:
        print(f"decoded_shape: {decoded_shape}")
        print(f"daq_raw_shape: {tuple(first_raw.shape)}")
        print(f"total_gainbit_ones: {total_gainbit_ones}")
        print(f"total_pixels_checked: {total_pixels}")
        print(f"overall_gainbit_fraction: {total_gainbit_ones / total_pixels:.6f}")
        print("top2_bit_counts ((raw >> 14) & 0x3):")
        for value in range(4):
            print(f"  {value}: {top2_counter.get(value, 0)}")
        print(f"gainbit_image_same_for_decoded_frames: {first_gainbit_equal}")
        if expected_gainbit is not None:
            print()
            print("False-positive/false-negative gainbit summary:")
            print(f"  expected_source: {expected_source}")
            if pixel_status_source is not None:
                print(f"  pixel_status_source: {pixel_status_source}")
            print("  false_positive means observed gainbit=1 while expected=0")
            print("  false_negative means observed gainbit=0 while expected=1")
            _print_mismatch_summary(
                "false_positive_pixels",
                fp_occurrences,
                usable_mask,
                control_mask,
                args.fpfn_max_pixels,
                pixel_status_bad_mask,
            )
            _print_mismatch_summary(
                "false_negative_pixels",
                fn_occurrences,
                usable_mask,
                control_mask,
                args.fpfn_max_pixels,
                pixel_status_bad_mask,
            )
    elif expected_gainbit is not None:
        print()
        print("False-positive/false-negative gainbit summary: unavailable")
        print(f"  expected_source: {expected_source}")
        if pixel_status_source is not None:
            print(f"  pixel_status_source: {pixel_status_source}")
        print("  no decoded image frames were available")
        print(f"  minimum DAQ raw image payload content: {min_image_payload} bytes")
        print("  FP/FN checks require decoded image frames, not only short packet records")

    if args.save_prefix and first_raw is not None:
        raw_path = f"{args.save_prefix}_first_raw_u16.npy"
        gain_path = f"{args.save_prefix}_first_gainbit.npy"
        np.save(raw_path, first_raw)
        np.save(gain_path, first_gainbit.astype(np.uint8))
        print()
        print(f"saved {raw_path}")
        print(f"saved {gain_path}")

    if not decoded:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
