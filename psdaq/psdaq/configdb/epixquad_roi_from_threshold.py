#!/usr/bin/env python3

"""Generate an epixquad ROI mask from thresholded psana data.

This standalone helper reads detector data directly with psana, accumulates a
binary ROI mask across events, optionally expands the ROI with a diamond
neighborhood, and can emit a detector-panel preview PNG plus a deployable
store-layout gainmap text file for epixquad_store_gainmap.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
from psana import DataSource

from psdaq.configdb.epixquad_gainmap_mask import _assembled_to_store_layout

ASIC_ROWS = 176
ASIC_COLS = 192
MODULE_COUNT = 4
RAW_GAINBITS_MASK = (1 << 14) - 1


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Generate an epixquad ROI mask from thresholded psana data'
    )
    parser.add_argument('-e', '--exp', help='experiment code, for example ued1015999')
    parser.add_argument('-r', '--run', type=int, help='run number')
    parser.add_argument(
        '--xtc-dir',
        default=None,
        help='optional xtc directory passed through to psana.DataSource(dir=...)',
    )
    parser.add_argument(
        '--detname',
        default='epixquad1kfps',
        help='detector name passed to run.Detector() (default: epixquad1kfps)',
    )
    parser.add_argument(
        '--detobj',
        choices=('raw', 'calib'),
        default='calib',
        help='detector data object to threshold (default: calib)',
    )
    parser.add_argument('-t', '--threshold', type=float, help='threshold applied to each event')
    parser.add_argument(
        '--expand-radius',
        type=int,
        default=0,
        help='diamond ROI expansion radius in pixels applied after thresholding',
    )
    parser.add_argument(
        '--max-nevents',
        type=int,
        default=50,
        help='maximum number of good events to accumulate (default: 50)',
    )
    parser.add_argument(
        '-p',
        '--path',
        default='.',
        help='output directory to create/use for generated files (default: current directory)',
    )
    parser.add_argument(
        '--write-png',
        '--writePng',
        dest='write_png',
        action='store_true',
        help='write an assembled detector-panel PNG overlay of the accumulated ROI mask',
    )
    parser.add_argument(
        '--write-gainmap-txt',
        '--writeGainmapTxt',
        dest='write_gainmap_txt',
        action='store_true',
        help='write a deployable epixquad gainmap text file with ROI->0 and background->1',
    )
    parser.add_argument(
        '--test-diamond',
        '--testDiamond',
        dest='test_diamond',
        action='store_true',
        help='create a 5x5 mask with 3 peaks and print the result of binary mask expansion with radius=2',
    )
    return parser.parse_args()


def diamond_offsets(radius, debug_print=False):
    if debug_print:
        print(f'Calculating diamond offsets for radius {radius}')
    offsets = []
    for dr in range(-radius, radius + 1):
        max_dc = radius - abs(dr)
        for dc in range(-max_dc, max_dc + 1):
            offsets.append((dr, dc))
            if debug_print:
                print(f'Diamond offset: ({dr}, {dc})')
    return offsets


def expand_binary_mask(mask, radius, debug_print=False):
    """Dilate a binary mask by a Manhattan-distance diamond of the given radius.

    Each True pixel in ``mask`` acts as a seed. The output marks any pixel whose
    row/column offset ``(dr, dc)`` from at least one seed satisfies
    ``abs(dr) + abs(dc) <= radius``. This is binary morphological dilation with
    a diamond-shaped structuring element in the L1/Manhattan metric.

    The implementation pads the mask by ``radius`` pixels on each side, then
    ORs together shifted mask-sized views from the padded array. Padding keeps
    the shifted reads in bounds while the output remains the same shape as the
    input mask.
    """
    if radius <= 0:
        return np.asarray(mask, dtype=bool)

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim < 2:
        raise ValueError(f'expand_binary_mask expects at least 2 dimensions, got {mask.ndim}')

    pad_width = [(0, 0)] * mask.ndim
    pad_width[-2] = (radius, radius)
    pad_width[-1] = (radius, radius)
    padded = np.pad(mask, pad_width, mode='constant', constant_values=False)

    row_start = radius
    row_stop = radius + mask.shape[-2]
    col_start = radius
    col_stop = radius + mask.shape[-1]

    expanded = np.zeros_like(mask, dtype=bool)
    for dr, dc in diamond_offsets(radius, debug_print=debug_print):
        expanded |= padded[..., row_start + dr:row_stop + dr, col_start + dc:col_stop + dc]
    return expanded


def _print_debug_mask(title, mask):
    print(title)
    print(np.asarray(mask, dtype=int))


def _print_diamond_expansion_walkthrough(mask, radius, max_shifts=3):
    mask = np.asarray(mask, dtype=bool)
    offsets = diamond_offsets(radius, debug_print=True)

    pad_width = [(0, 0)] * mask.ndim
    pad_width[-2] = (radius, radius)
    pad_width[-1] = (radius, radius)
    padded = np.pad(mask, pad_width, mode='constant', constant_values=False)

    row_start = radius
    row_stop = radius + mask.shape[-2]
    col_start = radius
    col_stop = radius + mask.shape[-1]

    _print_debug_mask('padded mask:', padded)

    expanded = np.zeros_like(mask, dtype=bool)
    for idx, (dr, dc) in enumerate(offsets[:max_shifts], start=1):
        shifted = padded[..., row_start + dr:row_stop + dr, col_start + dc:col_stop + dc]
        _print_debug_mask(f'shift {idx} with offset ({dr}, {dc}):', shifted)
        expanded |= shifted
        _print_debug_mask(f'expanded after shift {idx}:', expanded)

    expanded = expand_binary_mask(mask, radius=radius)
    _print_debug_mask(f'expanded with radius={radius}:', expanded)


def _validate_module_shape(array, name):
    array = np.asarray(array)
    if array.ndim != 3 or array.shape[0] != MODULE_COUNT:
        raise ValueError(f'{name} expects shape ({MODULE_COUNT}, rows, cols), got {array.shape}')
    if array.shape[1:] != (ASIC_ROWS * 2, ASIC_COLS * 2):
        raise ValueError(
            f'{name} expects module plane shape {(ASIC_ROWS * 2, ASIC_COLS * 2)}, got {array.shape[1:]}'
        )
    return array


def assemble_epixquad_panel(array):
    array = _validate_module_shape(array, 'assemble_epixquad_panel')
    top = np.hstack([array[3], array[2]])
    bottom = np.hstack([array[1], array[0]])
    return np.vstack([top, bottom])


def write_roi_panel_png(image, roi_mask, png_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    panel_image = assemble_epixquad_panel(image)
    panel_mask = assemble_epixquad_panel(np.asarray(roi_mask, dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 9), dpi=160)
    ax.imshow(panel_image, cmap='gray', origin='upper')
    ax.imshow(
        np.ma.masked_where(~panel_mask, panel_mask),
        cmap='Reds',
        alpha=0.45,
        origin='upper',
        interpolation='none',
    )
    ax.contour(panel_mask.astype(float), levels=[0.5], colors=['cyan'], linewidths=0.6)
    ax.set_title('ROI overlay detector panel')
    ax.set_xlabel('col')
    ax.set_ylabel('row')
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)


def write_gainmap_txt(roi_mask, output_path):
    assembled = assemble_epixquad_panel(np.asarray(roi_mask, dtype=bool))
    assembled_labels = np.where(assembled, 0, 1).astype(np.uint8)
    store_mask = _assembled_to_store_layout(assembled_labels)
    np.savetxt(output_path, store_mask, fmt='%u')
    return store_mask


def _read_frames(det, evt, detobj):
    if detobj == 'raw':
        frames = det.raw.raw(evt)
        if frames is None:
            return None
        return frames & RAW_GAINBITS_MASK
    if detobj == 'calib':
        return det.raw.calib(evt)
    raise ValueError(f'Unsupported detobj: {detobj}')


def _output_stem(output_dir, runnum, detobj, expand_radius):
    label = detobj
    if expand_radius > 0:
        label += f'_expand{expand_radius}'
    return output_dir / f'roiFromAboveThreshold_r{runnum}_c0_{label}'


def main():
    args = _parse_args()

    if args.test_diamond:
        test_mask = np.zeros((5, 5), dtype=bool)
        test_mask[2, 2] = True
        _print_debug_mask('test mask:', test_mask)
        _print_diamond_expansion_walkthrough(test_mask, radius=2, max_shifts=3)
        return
    if args.exp is None:
        raise ValueError('--exp is required unless --test-diamond is used')
    if args.run is None:
        raise ValueError('--run is required unless --test-diamond is used')
    if args.threshold is None:
        raise ValueError('--threshold is required unless --test-diamond is used')
    if args.expand_radius < 0:
        raise ValueError(f'expand radius must be non-negative, got {args.expand_radius}')
    if args.max_nevents <= 0:
        raise ValueError(f'max-nevents must be > 0, got {args.max_nevents}')

    output_dir = Path(args.path)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds_kwargs = {
        'exp': args.exp,
        'run': args.run,
        'max_events': args.max_nevents,
    }
    if args.xtc_dir is not None:
        ds_kwargs['dir'] = args.xtc_dir

    ds = DataSource(**ds_kwargs)
    run = next(ds.runs())
    det = run.Detector(args.detname)

    accumulated = None
    preview_frames = None
    ngood = 0

    for evt in run.events():
        frames = _read_frames(det, evt, args.detobj)
        if frames is None:
            continue

        frames = _validate_module_shape(frames, 'event frames')
        if preview_frames is None:
            preview_frames = np.asarray(frames)

        thresholded = np.asarray(frames >= args.threshold, dtype=bool)
        if args.expand_radius > 0:
            thresholded = expand_binary_mask(thresholded, args.expand_radius)

        if accumulated is None:
            accumulated = thresholded.copy()
        else:
            accumulated |= thresholded

        ngood += 1
        if ngood % 100 == 0:
            print(f'n good events analyzed: {ngood}')
            print(f'aboveThreshold pixels: {int(accumulated.sum())}')
        if ngood >= args.max_nevents:
            break

    if accumulated is None:
        raise RuntimeError('No valid detector events found')

    stem = _output_stem(output_dir, args.run, args.detobj, args.expand_radius)
    npy_path = Path(f'{stem}.npy')
    np.save(npy_path, accumulated)
    print(npy_path)

    if args.write_png:
        if preview_frames is None:
            raise RuntimeError('cannot write PNG without at least one valid event')
        png_path = Path(f'{stem}_panel.png')
        write_roi_panel_png(preview_frames, accumulated, png_path)
        print(png_path)

    if args.write_gainmap_txt:
        gainmap_path = Path(f'{stem}_gainmap.txt')
        store_mask = write_gainmap_txt(accumulated, gainmap_path)
        print(gainmap_path)
        labels = sorted(int(v) for v in np.unique(store_mask))
        print(f'gainmap shape {store_mask.shape}, labels {labels}')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
