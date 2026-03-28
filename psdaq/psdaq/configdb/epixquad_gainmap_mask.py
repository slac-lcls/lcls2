#!/usr/bin/env python3

"""
Generate epixquad gain-map masks for configdb workflows.

Outputs:
- store-layout text file for psdaq/psdaq/configdb/epixquad_store_gainmap.py
- optional assembled-detector PNG preview for visual inspection

Default mask semantics:
- label 0: pixels inside the requested radius
- label 1: pixels outside the requested radius

3-gain mask semantics:
- label 0: pixels inside the requested radius within the beam-center quadrants
- label 1: pixels outside the requested radius but still within the beam-center quadrants
- outer label: pixels outside the beam-center quadrants

The beam-center quadrants are the dynamic 2x2 ASIC block selected from the beam
center in assembled-detector coordinates.
"""

import argparse
from pathlib import Path
import struct
import sys
import zlib

import numpy as np


ASIC_ROWS = 176
ASIC_COLS = 192
ASIC_GRID_ROWS = 4
ASIC_GRID_COLS = 4
IMG_ROWS = ASIC_GRID_ROWS * ASIC_ROWS
IMG_COLS = ASIC_GRID_COLS * ASIC_COLS
DEFAULT_CENTER_X = (IMG_COLS - 1) / 2.0
DEFAULT_CENTER_Y = (IMG_ROWS - 1) / 2.0
MAJOR_BOUNDARY_X = IMG_COLS // 2
MAJOR_BOUNDARY_Y = IMG_ROWS // 2
PREVIEW_SCALE = 2


def _nearest_block_index(center, step, max_index):
    centers = np.arange(1, max_index + 1, dtype=np.float64) * step
    return int(np.argmin(np.abs(centers - center)))


def _beam_center_block(center_x, center_y):
    if not (0.0 <= center_x < IMG_COLS and 0.0 <= center_y < IMG_ROWS):
        raise ValueError(
            f"Beam center ({center_x:.3f}, {center_y:.3f}) is outside assembled image "
            f"bounds 0..{IMG_COLS - 1}, 0..{IMG_ROWS - 1}"
        )

    block_col = _nearest_block_index(center_x, ASIC_COLS, ASIC_GRID_COLS - 1)
    block_row = _nearest_block_index(center_y, ASIC_ROWS, ASIC_GRID_ROWS - 1)

    x0 = block_col * ASIC_COLS
    x1 = x0 + 2 * ASIC_COLS
    y0 = block_row * ASIC_ROWS
    y1 = y0 + 2 * ASIC_ROWS
    return block_row, block_col, x0, x1, y0, y1


def _validate_radius(radius, center_x, center_y, x0, x1, y0, y1):
    if radius <= 0:
        raise ValueError(f"Radius must be > 0: {radius}")
    if radius > ASIC_ROWS:
        raise ValueError(f"Radius must be <= ASIC height ({ASIC_ROWS}): {radius}")

    max_contained = min(center_x - x0, x1 - center_x, center_y - y0, y1 - center_y)
    if radius > max_contained:
        raise ValueError(
            f"Radius {radius} extends outside selected beam-center quadrants; "
            f"maximum fully contained radius is {max_contained:.3f}"
        )


def _build_mask(radius, center_x, center_y, outer_label, x0, x1, y0, y1):
    mask = np.full((IMG_ROWS, IMG_COLS), outer_label, dtype=np.uint8)

    yy, xx = np.indices((y1 - y0, x1 - x0), dtype=np.float64)
    yy += y0
    xx += x0
    rr = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)

    block = np.ones((y1 - y0, x1 - x0), dtype=np.uint8)
    block[rr < radius] = 0
    mask[y0:y1, x0:x1] = block
    return mask


def _build_binary_mask(radius, center_x, center_y):
    yy, xx = np.indices((IMG_ROWS, IMG_COLS), dtype=np.float64)
    rr = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)

    mask = np.ones((IMG_ROWS, IMG_COLS), dtype=np.uint8)
    mask[rr < radius] = 0
    return mask


def _assembled_to_store_layout(mask):
    tiles = []
    for half in np.vsplit(mask, 2):
        tiles.extend(np.hsplit(half, 2))

    elems = np.asarray([np.asarray(tiles[i], dtype=np.uint8) for i in (3, 2, 1, 0)], dtype=np.uint8)
    return elems.reshape((elems.shape[0] * elems.shape[1], elems.shape[2]))


def _label_colors(labels):
    preferred = {
        0: (138, 43, 226),
        1: (217, 95, 2),
        2: (255, 234, 0),
        3: (31, 120, 180),
        4: (51, 160, 44),
        5: (227, 26, 28),
    }
    fallback = [
        (166, 206, 227),
        (178, 223, 138),
        (251, 154, 153),
        (202, 178, 214),
        (253, 191, 111),
        (177, 89, 40),
    ]

    colors = []
    for idx, label in enumerate(labels):
        colors.append(preferred.get(label, fallback[idx % len(fallback)]))
    return colors


def _paint_line(rgb, x0, x1, y0, y1, color):
    rgb[y0:y1, x0:x1] = color


def _paint_rect_border(rgb, x0, x1, y0, y1, color, thickness=2, dash=16):
    for x in range(x0, x1, dash * 2):
        xe = min(x + dash, x1)
        rgb[max(y0, 0):min(y0 + thickness, rgb.shape[0]), max(x, 0):min(xe, rgb.shape[1])] = color
        rgb[max(y1 - thickness, 0):min(y1, rgb.shape[0]), max(x, 0):min(xe, rgb.shape[1])] = color
    for y in range(y0, y1, dash * 2):
        ye = min(y + dash, y1)
        rgb[max(y, 0):min(ye, rgb.shape[0]), max(x0, 0):min(x0 + thickness, rgb.shape[1])] = color
        rgb[max(y, 0):min(ye, rgb.shape[0]), max(x1 - thickness, 0):min(x1, rgb.shape[1])] = color


def _write_png(path, rgb):
    height, width, channels = rgb.shape
    if channels != 3:
        raise ValueError(f'PNG writer expects RGB image, got shape {rgb.shape}')

    raw = b''.join(b'\x00' + rgb[row].tobytes() for row in range(height))
    compressed = zlib.compress(raw, level=9)

    def chunk(tag, data):
        return (
            struct.pack('>I', len(data)) +
            tag +
            data +
            struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)
        )

    ihdr = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    png = [
        b'\x89PNG\r\n\x1a\n',
        chunk(b'IHDR', ihdr),
        chunk(b'IDAT', compressed),
        chunk(b'IEND', b''),
    ]
    Path(path).write_bytes(b''.join(png))


def _save_preview(path, mask, center_x, center_y, outer_label, x0, x1, y0, y1):
    labels = sorted(int(v) for v in np.unique(mask))
    colors = _label_colors(labels)
    label_to_color = {label: np.array(color, dtype=np.uint8) for label, color in zip(labels, colors)}

    rgb = np.zeros((IMG_ROWS, IMG_COLS, 3), dtype=np.uint8)
    for label, color in label_to_color.items():
        rgb[mask == label] = color

    grid_color = np.array((120, 120, 120), dtype=np.uint8)
    major_color = np.array((0, 0, 0), dtype=np.uint8)
    box_color = np.array((0, 0, 0), dtype=np.uint8)

    for x in range(ASIC_COLS, IMG_COLS, ASIC_COLS):
        _paint_line(rgb, x - 1, min(x + 1, IMG_COLS), 0, IMG_ROWS, grid_color)
    for y in range(ASIC_ROWS, IMG_ROWS, ASIC_ROWS):
        _paint_line(rgb, 0, IMG_COLS, y - 1, min(y + 1, IMG_ROWS), grid_color)

    _paint_line(rgb, MAJOR_BOUNDARY_X - 4, min(MAJOR_BOUNDARY_X + 4, IMG_COLS), 0, IMG_ROWS, major_color)
    _paint_line(rgb, 0, IMG_COLS, MAJOR_BOUNDARY_Y - 2, min(MAJOR_BOUNDARY_Y + 2, IMG_ROWS), major_color)
    _paint_rect_border(rgb, x0, x1, y0, y1, box_color, thickness=2, dash=18)

    center_ix = int(round(center_x))
    center_iy = int(round(center_y))
    _paint_line(rgb, max(center_ix - 2, 0), min(center_ix + 3, IMG_COLS), max(center_iy - 12, 0), min(center_iy + 13, IMG_ROWS), major_color)
    _paint_line(rgb, max(center_ix - 12, 0), min(center_ix + 13, IMG_COLS), max(center_iy - 2, 0), min(center_iy + 3, IMG_ROWS), major_color)

    scaled = np.repeat(np.repeat(rgb, PREVIEW_SCALE, axis=0), PREVIEW_SCALE, axis=1)
    _write_png(path, scaled)


def _parse_args():
    parser = argparse.ArgumentParser(description='Generate epixquad gain-map mask input for configdb workflows')
    parser.add_argument('--3gain-mode', dest='three_gain_mode', action='store_true',
                        help='Emit a 3-label mask using beam-center quadrants plus an outer region')
    parser.add_argument('-r', '--radius', type=float, required=True,
                        help='Radius in pixels for label 0')
    parser.add_argument('--dx', type=float, default=0.0,
                        help='Beam-center x displacement in pixels from default center')
    parser.add_argument('--dy', type=float, default=0.0,
                        help='Beam-center y displacement in pixels from default center')
    parser.add_argument('--center-x', type=float, default=DEFAULT_CENTER_X,
                        help=f'Base x center in pixels (default {DEFAULT_CENTER_X:.1f})')
    parser.add_argument('--center-y', type=float, default=DEFAULT_CENTER_Y,
                        help=f'Base y center in pixels (default {DEFAULT_CENTER_Y:.1f})')
    parser.add_argument('--outer-label', type=int, default=2,
                        help='Label used for all pixels outside the beam-center quadrants (default 2)')
    parser.add_argument('-o', '--output', default='epixquad_gainmap_mask.txt',
                        help='Output mask file in epixquad_store_gainmap.py --file layout (1408x384 text)')
    parser.add_argument('--assembled-output', default=None,
                        help='Optional assembled-view PNG preview for inspection')
    return parser.parse_args()


def main():
    args = _parse_args()

    if args.outer_label in (0, 1) and args.three_gain_mode:
        raise ValueError(f'outer-label must not collide with radial labels 0/1: {args.outer_label}')

    center_x = args.center_x + args.dx
    center_y = args.center_y + args.dy
    if args.three_gain_mode:
        block_row, block_col, x0, x1, y0, y1 = _beam_center_block(center_x, center_y)
        _validate_radius(args.radius, center_x, center_y, x0, x1, y0, y1)
        assembled = _build_mask(args.radius, center_x, center_y, args.outer_label, x0, x1, y0, y1)
    else:
        if args.radius <= 0:
            raise ValueError(f"Radius must be > 0: {args.radius}")
        block_row = block_col = None
        x0, x1, y0, y1 = 0, IMG_COLS, 0, IMG_ROWS
        assembled = _build_binary_mask(args.radius, center_x, center_y)

    store_mask = _assembled_to_store_layout(assembled)

    np.savetxt(args.output, store_mask, fmt='%u')

    if args.assembled_output:
        assembled_path = Path(args.assembled_output)
        if assembled_path.suffix.lower() != '.png':
            raise ValueError(f'assembled-output must be a .png file: {assembled_path}')
        _save_preview(assembled_path, assembled, center_x, center_y, args.outer_label, x0, x1, y0, y1)

    unique, counts = np.unique(assembled, return_counts=True)
    count_str = ', '.join(f'{int(label)}:{int(count)}' for label, count in zip(unique, counts))
    print(f'Wrote {args.output} with shape {store_mask.shape}')
    print(f'Beam center (x, y) = ({center_x:.3f}, {center_y:.3f}) pixels')
    if args.three_gain_mode:
        print(f'Beam-center quadrant block row/col = ({block_row}, {block_col})')
        print(f'Beam-center quadrant bounds x=[{x0}, {x1}), y=[{y0}, {y1})')
    else:
        print('Binary mask semantics: 0 = inside radius, 1 = outside radius')
    print(f'Radius = {args.radius}')
    print(f'Label counts = {count_str}')
    if args.assembled_output:
        print(f'Wrote assembled PNG preview to {args.assembled_output}')
    if args.three_gain_mode:
        print(f'Suggested upload: epixquad_store_gainmap --file {args.output} --map 0:L --map 1:M --map {args.outer_label}:H ...')
    else:
        print('Suggested uploads:')
        print(f'  epixquad_store_gainmap --file {args.output} --map 0:L --map 1:M ...')
        print(f'  epixquad_store_gainmap --file {args.output} --map 0:L --map 1:H ...')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
