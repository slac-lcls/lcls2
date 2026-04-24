import argparse
import os
from pathlib import Path

import numpy as np

from psdaq.configdb.epixquad_cdict import epixquad_cdict
from psdaq.configdb.epixquad_gainmap_mask import (
    IMG_COLS,
    IMG_ROWS,
    MAJOR_BOUNDARY_X,
    MAJOR_BOUNDARY_Y,
    PREVIEW_SCALE,
    _label_colors,
    _paint_line,
    _write_png,
)


gain_dict = {
    'H': {'value': 0xc, 'trbit': 1},
    'M': {'value': 0xc, 'trbit': 0},
    'L': {'value': 0x8, 'trbit': 0},
    'AHL': {'value': 0x0, 'trbit': 1},
    'AML': {'value': 0x0, 'trbit': 0},
}

# Fixed low uses pixel value 0x8 and is treated here as trbit-flexible.
# That allows an ASIC to mix L with either the trbit=0 family (M/AML)
# or the trbit=1 family (H/AHL), while still rejecting ASIC-local mixes
# that require both trbit families for non-L gains.
TRBIT_FLEXIBLE_GAINS = {'L'}

ASIC_ROWS = 176
ASIC_COLS = 192
ASIC_COUNT = 16
MODULE_COUNT = 4
MODULE_ROWS = ASIC_ROWS * 2
MODULE_COLS = ASIC_COLS * 2
RAW_SHAPE = (MODULE_COUNT, MODULE_ROWS, MODULE_COLS)


def copyValues(din, dout, k=None):
    if k is not None and ':RO' in k:
        return
    if isinstance(din, dict):
        for key, value in din.items():
            copyValues(value, dout, key if k is None else k + '.' + key)
    else:
        v = dout.get(k, withtype=True)
        if v is None:
            pass
        elif len(v) > 2:
            print(f'Skipping {k}')
        elif len(v) == 1:
            print(f'Updating {k}')
            dout.set(k, din, 'UINT8')
        else:
            print(f'Updating {k}')
            dout.set(k, din, v[0])


def _read_raw_detector_layout(fname):
    labels = np.load(fname)
    if labels.shape != RAW_SHAPE:
        raise ValueError(
            f'Expected raw detector .npy shape {RAW_SHAPE}, got {labels.shape} from {fname}'
        )

    unique_labels = sorted(int(v) for v in np.unique(labels))
    if any(label not in (0, 1) for label in unique_labels):
        raise ValueError(
            f'Raw detector .npy input must be binary with labels [0, 1], got {unique_labels}'
        )
    return np.asarray(labels, dtype=np.uint8)


def _assemble_detector_chunks(detector_chunks):
    top = np.hstack([detector_chunks[3], detector_chunks[2]])
    bottom = np.hstack([detector_chunks[1], detector_chunks[0]])
    return np.vstack([top, bottom])


def _default_detector_mask_output(input_path):
    path = Path(input_path)
    return path.with_name(f'{path.stem}_detector_mask.npy')


def _default_detector_preview_output(input_path):
    path = Path(input_path)
    return path.with_name(f'{path.stem}_detector_mask.png')


def _save_detector_preview(path, detector_chunks):
    assembled = _assemble_detector_chunks(detector_chunks)
    labels = sorted(int(v) for v in np.unique(assembled))
    colors = _label_colors(labels)
    label_to_color = {
        label: np.array(color, dtype=np.uint8)
        for label, color in zip(labels, colors)
    }

    rgb = np.zeros((IMG_ROWS, IMG_COLS, 3), dtype=np.uint8)
    for label, color in label_to_color.items():
        rgb[assembled == label] = color

    grid_color = np.array((120, 120, 120), dtype=np.uint8)
    major_color = np.array((0, 0, 0), dtype=np.uint8)

    for x in range(ASIC_COLS, IMG_COLS, ASIC_COLS):
        _paint_line(rgb, x - 1, min(x + 1, IMG_COLS), 0, IMG_ROWS, grid_color)
    for y in range(ASIC_ROWS, IMG_ROWS, ASIC_ROWS):
        _paint_line(rgb, 0, IMG_COLS, y - 1, min(y + 1, IMG_ROWS), grid_color)

    _paint_line(rgb, MAJOR_BOUNDARY_X - 4, min(MAJOR_BOUNDARY_X + 4, IMG_COLS), 0, IMG_ROWS, major_color)
    _paint_line(rgb, 0, IMG_COLS, MAJOR_BOUNDARY_Y - 2, min(MAJOR_BOUNDARY_Y + 2, IMG_ROWS), major_color)

    scaled = np.repeat(np.repeat(rgb, PREVIEW_SCALE, axis=0), PREVIEW_SCALE, axis=1)
    _write_png(path, scaled)


def _compare_detector_chunks(detector_chunks, compare_path):
    reference = np.load(compare_path)
    if reference.shape != (MODULE_COUNT, MODULE_ROWS, MODULE_COLS):
        raise ValueError(
            f'Expected comparison mask shape {(MODULE_COUNT, MODULE_ROWS, MODULE_COLS)}, '
            f'got {reference.shape} from {compare_path}'
        )

    identical = np.array_equal(detector_chunks, reference)
    print(f'Comparison with {compare_path}: identical={identical}')
    if identical:
        return

    diff_mask = detector_chunks != reference
    total_diff = int(np.count_nonzero(diff_mask))
    print(f'  differing pixels: {total_diff} / {detector_chunks.size}')
    for seg in range(MODULE_COUNT):
        seg_diff = int(np.count_nonzero(diff_mask[seg]))
        print(f'  module {seg}: differing pixels = {seg_diff}')

    detector_values = sorted(int(v) for v in np.unique(detector_chunks))
    reference_values = sorted(int(v) for v in np.unique(reference))
    print(f'  detector-chunk unique labels: {detector_values}')
    print(f'  reference unique labels:      {reference_values}')


def _emit_raw_detector_mask_artifacts(detector_chunks, input_path, detector_mask_output=None,
                                      detector_preview_output=None, compare_mask_npy=None):
    if detector_chunks.shape != RAW_SHAPE:
        raise ValueError(
            f'Expected raw detector chunk shape {RAW_SHAPE}, got {detector_chunks.shape}'
        )

    mask_output = Path(detector_mask_output) if detector_mask_output else _default_detector_mask_output(input_path)
    preview_output = Path(detector_preview_output) if detector_preview_output else _default_detector_preview_output(input_path)

    np.save(mask_output, detector_chunks)
    _save_detector_preview(preview_output, detector_chunks)

    print(f'Wrote detector-shaped mask array: {mask_output} shape={detector_chunks.shape}')
    print(f'Wrote detector preview PNG:       {preview_output}')

    if compare_mask_npy:
        _compare_detector_chunks(detector_chunks, compare_mask_npy)


def _parse_label_maps(entries):
    label_map = {}
    for entry in entries:
        try:
            label_str, gain = entry.split(':', 1)
        except ValueError as exc:
            raise ValueError(f'Invalid --map entry {entry!r}; expected LABEL:GAIN') from exc

        try:
            label = int(label_str)
        except ValueError as exc:
            raise ValueError(f'Invalid label {label_str!r} in --map entry {entry!r}') from exc

        if label < 0 or label > 255:
            raise ValueError(f'Label {label} out of range for uint8 mask values')
        if gain not in gain_dict:
            raise ValueError(f'Unknown gain {gain!r} in --map entry {entry!r}')
        if label in label_map and label_map[label] != gain:
            raise ValueError(f'Conflicting gain mapping for label {label}: {label_map[label]} vs {gain}')
        label_map[label] = gain
    if not label_map:
        raise ValueError('At least one --map LABEL:GAIN entry is required in label-map mode')
    return label_map


def _summary_from_counts(counts):
    return ', '.join(f'{int(label)}:{int(count)}' for label, count in sorted(counts.items()))


def _resolve_asic_trbit(gains_present):
    fixed_gains = [gain for gain in gains_present if gain not in TRBIT_FLEXIBLE_GAINS]
    fixed_trbits = {gain_dict[gain]['trbit'] for gain in fixed_gains}

    if len(fixed_trbits) > 1:
        raise ValueError(
            f'Incompatible non-L gains {sorted(fixed_gains)} require multiple trbit values'
        )

    if fixed_trbits:
        return fixed_trbits.pop()

    # Pure-L ASICs default to trbit=0 for deterministic behavior.
    return 0


RAW_ASIC_LAYOUT = (
    {'slot': 0, 'row_slice': (176, 352), 'col_slice': (192, 384)},
    {'slot': 1, 'row_slice': (0, 176),   'col_slice': (192, 384)},
    {'slot': 2, 'row_slice': (0, 176),   'col_slice': (0, 192)},
    {'slot': 3, 'row_slice': (176, 352), 'col_slice': (0, 192)},
)


def _label_map_to_raw_pixel_map(raw_labels, label_map):
    unique_labels = sorted(int(v) for v in np.unique(raw_labels))
    if any(label not in (0, 1) for label in unique_labels):
        raise ValueError(
            f'Raw detector .npy input must be binary with labels [0, 1], got {unique_labels}'
        )

    missing = [label for label in (0, 1) if label not in label_map]
    if missing:
        raise ValueError(
            f'Raw detector .npy input requires --map entries for labels 0 and 1, missing {missing}'
        )

    d = {'user.gain_mode': 5}
    pixel_map_raw = np.zeros_like(raw_labels, dtype=np.uint8)

    for label in (0, 1):
        gain = label_map[label]
        pixel_map_raw[raw_labels == label] = gain_dict[gain]['value']

    for segment in range(MODULE_COUNT):
        seg_labels = raw_labels[segment]
        for layout in RAW_ASIC_LAYOUT:
            r0, r1 = layout['row_slice']
            c0, c1 = layout['col_slice']
            asic = 4 * segment + layout['slot']
            asic_labels = seg_labels[r0:r1, c0:c1]

            labels_present, label_counts = np.unique(asic_labels, return_counts=True)
            labels_present = [int(v) for v in labels_present]
            counts_dict = {int(label): int(count) for label, count in zip(labels_present, label_counts)}

            gains_present = sorted({label_map[label] for label in labels_present})
            try:
                trbit = _resolve_asic_trbit(gains_present)
            except ValueError as exc:
                raise ValueError(
                    f'ASIC {asic} mixes gains with incompatible trbit values: '
                    f'{gains_present} from labels {labels_present}'
                ) from exc

            d[f'expert.EpixQuad.Epix10kaSaci{asic}.trbit'] = trbit
            print(
                f'ASIC {asic:02d}: labels [{_summary_from_counts(counts_dict)}] -> '
                f'gains {gains_present}, trbit={trbit}'
            )

    d['user.pixel_map_raw'] = np.asarray(pixel_map_raw, dtype=np.uint8)
    return d


def main():
    parser = argparse.ArgumentParser(description='Update epixquad raw gain map in configdb')
    parser.add_argument('--file', help='input raw detector mask .npy with shape (4,352,384)', type=str, required=True)
    parser.add_argument(
        '--detector-mask-output',
        help='optional output .npy for detector-shaped chunks with shape (4, 352, 384)',
        default=None,
    )
    parser.add_argument(
        '--detector-preview-output',
        help='optional output .png preview for the detector-shaped chunks',
        default=None,
    )
    parser.add_argument(
        '--compare-mask-npy',
        help='optional detector-shaped .npy mask to compare against',
        default=None,
    )
    parser.add_argument(
        '--map',
        action='append',
        default=[],
        metavar='LABEL:GAIN',
        help='label-to-gain mapping for the binary raw mask; required for labels 0 and 1 (example: --map 0:H --map 1:AHL)',
    )
    parser.add_argument('--dev', help='use development db', action='store_true')
    parser.add_argument('--inst', help='instrument', type=str, default='ued')
    parser.add_argument('--alias', help='alias name', type=str, default='BEAM')
    parser.add_argument('--name', help='detector name', type=str, default='epixquad')
    parser.add_argument('--segm', help='detector segment', type=int, default=0)
    parser.add_argument('--id', help='device id/serial num', type=str, default='serial1234')
    parser.add_argument('--user', help='user for HTTP authentication', type=str, default='uedopr')
    parser.add_argument('--password', help='password for HTTP authentication', type=str, default=os.getenv('CONFIGDB_AUTH'))
    parser.add_argument('--test', help='test transformation', action='store_true')
    args = parser.parse_args()

    import psdaq.configdb.configdb as cdb

    detname = f'{args.name}_{args.segm}'
    db = 'devconfigdb' if args.dev else 'configdb'
    url = f'https://pswww.slac.stanford.edu/ws-auth/{db}/ws/'

    mycdb = cdb.configdb(url, args.inst, False, root='configDB', user=args.user, password=args.password)
    cfg = mycdb.get_configuration(args.alias, detname)

    if cfg is None:
        raise ValueError(
            'Config for instrument/detname %s/%s not found. dbase url: %s, db_name: %s, config_style: %s'
            % (args.inst, detname, url, 'configDB', args.alias)
        )

    top = epixquad_cdict()
    copyValues(cfg, top)

    label_map = _parse_label_maps(args.map) if args.map else None
    if label_map is None:
        raise ValueError('Raw detector .npy input requires --map 0:GAIN --map 1:GAIN')

    input_path = Path(args.file)
    if input_path.suffix != '.npy':
        raise ValueError(f'Only raw detector .npy input is supported, got {args.file}')

    raw_labels = _read_raw_detector_layout(args.file)
    print(f'Read raw detector mask {args.file} with shape {raw_labels.shape} and labels {sorted(int(v) for v in np.unique(raw_labels))}')
    _emit_raw_detector_mask_artifacts(
        raw_labels,
        args.file,
        detector_mask_output=args.detector_mask_output,
        detector_preview_output=args.detector_preview_output,
        compare_mask_npy=args.compare_mask_npy,
    )
    print(f'Raw detector label map mode: {label_map}')
    d = _label_map_to_raw_pixel_map(raw_labels, label_map)

    copyValues(d, top)

    top.set('user.gain_mode', d['user.gain_mode'])
    print('Setting user.pixel_map_raw')
    top.set('user.pixel_map_raw', d['user.pixel_map_raw'])

    top.setInfo('epix10kaquad', args.name, args.segm, args.id, 'No comment')
    if not args.test:
        mycdb.modify_device(args.alias, top)


if __name__ == '__main__':
    main()
