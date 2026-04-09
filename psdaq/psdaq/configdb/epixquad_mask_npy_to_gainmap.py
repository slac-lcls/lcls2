#!/usr/bin/env python3

"""Convert a detector-shaped epixquad mask ndarray into gainmap text.

This helper is intended for Mask Editor outputs saved in detector coordinates
with shape ``(4, 352, 384)`` and labels such as ``0`` for ROI and ``1`` for
background. It assembles the module planes in the same panel order used by the
other epixquad configdb utilities and writes the deployable store-layout text
file consumed by ``epixquad_store_gainmap``.
"""

import argparse
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from psdaq.configdb.epixquad_gainmap_mask import (
    _assembled_to_store_layout,
    _label_colors,
    _paint_line,
    _write_png,
    ASIC_COLS,
    ASIC_ROWS,
    IMG_COLS,
    IMG_ROWS,
    MAJOR_BOUNDARY_X,
    MAJOR_BOUNDARY_Y,
    PREVIEW_SCALE,
)


MODULE_COUNT = 4
MODULE_ROWS = ASIC_ROWS * 2
MODULE_COLS = ASIC_COLS * 2


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Convert a detector-shaped epixquad mask ndarray into store-layout gainmap text'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Input detector-shaped mask .npy file with shape (4, 352, 384)',
    )
    parser.add_argument(
        '-o',
        '--output',
        required=True,
        help='Output store-layout gainmap text file with shape (1408, 384)',
    )
    parser.add_argument(
        '--assembled-output',
        default=None,
        help='Optional assembled-view PNG preview for inspection',
    )
    parser.add_argument(
        '--geometry-file',
        default=None,
        help='Optional geometry .data/.txt file used to render a Mask Editor-style preview',
    )
    parser.add_argument(
        '--geometry-output',
        default=None,
        help='Optional PNG preview rendered through the supplied geometry file',
    )
    return parser.parse_args()


def _read_detector_mask(path):
    mask = np.load(path)
    if mask.shape != (MODULE_COUNT, MODULE_ROWS, MODULE_COLS):
        raise ValueError(
            f'Expected detector mask shape {(MODULE_COUNT, MODULE_ROWS, MODULE_COLS)}, got {mask.shape} from {path}'
        )
    return np.asarray(mask)


def _assemble_epixquad_panel(mask):
    top = np.hstack([mask[3], mask[2]])
    bottom = np.hstack([mask[1], mask[0]])
    return np.vstack([top, bottom])


def _save_preview(path, mask):
    labels = sorted(int(v) for v in np.unique(mask))
    colors = _label_colors(labels)
    label_to_color = {label: np.array(color, dtype=np.uint8) for label, color in zip(labels, colors)}

    rgb = np.zeros((IMG_ROWS, IMG_COLS, 3), dtype=np.uint8)
    for label, color in label_to_color.items():
        rgb[mask == label] = color

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


def _save_geometry_preview(path, detector_mask, geometry_file):
    from psana.pscalib.geometry.GeometryAccess import GeometryAccess, img_from_pixel_arrays

    geo = GeometryAccess(geometry_file)
    if not geo.is_valid():
        raise ValueError(f'Failed to load geometry from {geometry_file}')

    rows, cols = geo.get_pixel_coord_indexes(do_tilt=True, cframe=0)
    geom_view = img_from_pixel_arrays(rows, cols, W=np.asarray(detector_mask, dtype=np.uint8), vbase=1)

    labels = sorted(int(v) for v in np.unique(geom_view))
    colors = _label_colors(labels)
    label_to_color = {label: np.array(color, dtype=np.uint8) for label, color in zip(labels, colors)}

    rgb = np.zeros((geom_view.shape[0], geom_view.shape[1], 3), dtype=np.uint8)
    for label, color in label_to_color.items():
        rgb[geom_view == label] = color

    _write_png(path, rgb)


def main():
    args = _parse_args()

    detector_mask = _read_detector_mask(args.input)
    assembled = _assemble_epixquad_panel(detector_mask)
    store_mask = _assembled_to_store_layout(np.asarray(assembled, dtype=np.uint8))
    np.savetxt(args.output, store_mask, fmt='%u')

    if args.assembled_output:
        assembled_path = Path(args.assembled_output)
        if assembled_path.suffix.lower() != '.png':
            raise ValueError(f'assembled-output must be a .png file: {assembled_path}')
        _save_preview(assembled_path, assembled)

    if args.geometry_output:
        if not args.geometry_file:
            raise ValueError('--geometry-output requires --geometry-file')
        geometry_path = Path(args.geometry_output)
        if geometry_path.suffix.lower() != '.png':
            raise ValueError(f'geometry-output must be a .png file: {geometry_path}')
        _save_geometry_preview(geometry_path, detector_mask, args.geometry_file)

    unique, counts = np.unique(detector_mask, return_counts=True)
    count_str = ', '.join(f'{int(label)}:{int(count)}' for label, count in zip(unique, counts))
    print(f'Read detector mask {args.input} with shape {detector_mask.shape}')
    print(f'Assembled mask shape {assembled.shape}')
    print(f'Wrote {args.output} with shape {store_mask.shape}')
    print(f'Label counts = {count_str}')
    if args.assembled_output:
        print(f'Wrote assembled PNG preview to {args.assembled_output}')
    if args.geometry_output:
        print(f'Wrote geometry PNG preview to {args.geometry_output} using {args.geometry_file}')
    print('Next step: upload with epixquad_store_gainmap --file ... --map LABEL:GAIN')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
