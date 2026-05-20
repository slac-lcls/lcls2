#!/usr/bin/env python3

"""
Convert an assembled epixquad label image into store-layout gainmap text.

The input mask is expected in assembled detector coordinates with shape 704x768.
This helper writes the 1408x384 store-layout text file consumed by
epixquad_store_gainmap.
"""

import argparse
from pathlib import Path
import sys

import numpy as np

from psdaq.configdb.epixquad_gainmap_mask import (
    ASIC_COLS,
    ASIC_ROWS,
    IMG_COLS,
    IMG_ROWS,
    MAJOR_BOUNDARY_X,
    MAJOR_BOUNDARY_Y,
    PREVIEW_SCALE,
    _assembled_to_store_layout,
    _label_colors,
    _paint_line,
    _write_png,
)


def _read_assembled_mask(path):
    mask = np.genfromtxt(path, dtype=np.uint8)
    if mask.shape != (IMG_ROWS, IMG_COLS):
        raise ValueError(
            f'Expected assembled mask shape {(IMG_ROWS, IMG_COLS)}, got {mask.shape} from {path}'
        )
    return mask


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


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Convert an assembled epixquad mask into store-layout gainmap text'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Input assembled mask text file with shape 704x768',
    )
    parser.add_argument(
        '-o',
        '--output',
        required=True,
        help='Output store-layout mask text file with shape 1408x384',
    )
    parser.add_argument(
        '--assembled-output',
        default=None,
        help='Optional assembled-view PNG preview for inspection',
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    assembled = _read_assembled_mask(args.input)
    store_mask = _assembled_to_store_layout(assembled)
    np.savetxt(args.output, store_mask, fmt='%u')

    if args.assembled_output:
        assembled_path = Path(args.assembled_output)
        if assembled_path.suffix.lower() != '.png':
            raise ValueError(f'assembled-output must be a .png file: {assembled_path}')
        _save_preview(assembled_path, assembled)

    unique, counts = np.unique(assembled, return_counts=True)
    count_str = ', '.join(f'{int(label)}:{int(count)}' for label, count in zip(unique, counts))
    print(f'Read assembled mask {args.input} with shape {assembled.shape}')
    print(f'Wrote {args.output} with shape {store_mask.shape}')
    print(f'Label counts = {count_str}')
    if args.assembled_output:
        print(f'Wrote assembled PNG preview to {args.assembled_output}')
    print('Next step: upload with epixquad_store_gainmap --file ... --map LABEL:GAIN')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
