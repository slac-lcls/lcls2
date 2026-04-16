#!/usr/bin/env python3

"""Dump deployed epixquad gain-map orientation diagnostics.

This helper reads the currently deployed epixquad configuration from configdb,
reconstructs the stored ``user.pixel_map`` array, and writes a set of
inspection-friendly ``.npy`` files that show how the map is arranged:

- as stored in ``user.pixel_map`` (16 ASICs with 2 padded rows),
- in store-layout panel order,
- in assembled detector view,
- in the same panel convention used by psana control-bit construction,
- with an optional synthetic raw bit-14 plane derived from selected pixel-map
  values, plus the corresponding combined cbits and gain-index products.

The synthetic raw output is diagnostic only. It does not attempt to model real
detector data; it only sets the event gain bit for selected configured pixel
codes so the orientation can be inspected without taking data.
"""

import argparse
import os
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ''):
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root / 'psdaq'))

ASIC_ROWS = 176
ASIC_COLS = 192
PADDED_ASIC_ROWS = 178
ASICS_PER_SEGMENT = 4
SEGMENT_COUNT = 4
TOTAL_ASICS = ASICS_PER_SEGMENT * SEGMENT_COUNT
SEGMENT_ROWS = ASIC_ROWS * 2
SEGMENT_COLS = ASIC_COLS * 2
STORE_ROWS = SEGMENT_ROWS * SEGMENT_COUNT
STORE_COLS = SEGMENT_COLS
ASSEMBLED_ROWS = SEGMENT_ROWS * 2
ASSEMBLED_COLS = SEGMENT_COLS * 2

DATA_GAIN_BIT = 1 << 14
GAIN_BIT_SHIFT = 9
TRBIT_BIT = 1 << 4


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Dump deployed epixquad gain-map orientation diagnostics'
    )
    parser.add_argument('--inst', default='ued', help='Instrument name (default: ued)')
    parser.add_argument('--alias', default='BEAM', help='Configuration alias (default: BEAM)')
    parser.add_argument('--name', default='epixquad1kfps', help='Detector name (default: epixquad1kfps)')
    parser.add_argument('--segm', type=int, default=0, help='Detector segment number (default: 0)')
    parser.add_argument('--user', default='uedopr', help='ConfigDB user name')
    parser.add_argument(
        '--password',
        default=os.getenv('CONFIGDB_AUTH'),
        help='ConfigDB password (default: CONFIGDB_AUTH env var)',
    )
    parser.add_argument('--dev', action='store_true', help='Use development configdb')
    parser.add_argument(
        '-o',
        '--output-dir',
        required=True,
        help='Directory for output .npy/.txt files',
    )
    parser.add_argument(
        '--gainbit-values',
        default='8',
        help=(
            'Comma-separated pixel-map codes whose pixels should get synthetic raw bit14. '
            'Values are parsed with int(token, 0); default: 8'
        ),
    )
    return parser.parse_args()


def _parse_value_list(spec):
    values = []
    for token in spec.split(','):
        token = token.strip()
        if not token:
            continue
        values.append(int(token, 0))
    if not values:
        raise ValueError('At least one --gainbit-values entry is required')
    return tuple(sorted(set(values)))


def _connect_configdb(args):
    import psdaq.configdb.configdb as cdb

    db = 'devconfigdb' if args.dev else 'configdb'
    url = f'https://pswww.slac.stanford.edu/ws-auth/{db}/ws/'
    return cdb.configdb(
        url,
        args.inst,
        False,
        root='configDB',
        user=args.user,
        password=args.password,
    )


def _fetch_configuration(args):
    detname = f'{args.name}_{args.segm}'
    cfg = _connect_configdb(args).get_configuration(args.alias, detname)
    if cfg is None:
        raise ValueError(
            f'Configuration not found for instrument/detname {args.inst}/{detname} alias {args.alias}'
        )
    return cfg, detname


def _extract_user_pixel_map(cfg):
    if 'user' not in cfg or 'pixel_map' not in cfg['user']:
        raise KeyError('Configuration does not contain user.pixel_map')

    arr = np.asarray(cfg['user']['pixel_map'], dtype=np.uint8)
    expected = TOTAL_ASICS * PADDED_ASIC_ROWS * ASIC_COLS
    if arr.size != expected:
        raise ValueError(
            f'user.pixel_map has {arr.size} elements, expected {expected} '
            f'for shape {(TOTAL_ASICS, PADDED_ASIC_ROWS, ASIC_COLS)}'
        )

    return arr.reshape(TOTAL_ASICS, PADDED_ASIC_ROWS, ASIC_COLS)


def _extract_trbits(cfg):
    try:
        eq = cfg['expert']['EpixQuad']
    except KeyError as exc:
        raise KeyError('Configuration does not contain expert.EpixQuad trbit settings') from exc

    trbits = []
    for i in range(TOTAL_ASICS):
        key = f'Epix10kaSaci{i}'
        if key not in eq or 'trbit' not in eq[key]:
            raise KeyError(f'Configuration missing {key}.trbit')
        trbits.append(int(eq[key]['trbit']))
    return np.asarray(trbits, dtype=np.uint8)


def _active_user_pixel_map(user_pixel_map):
    return np.asarray(user_pixel_map[:, :ASIC_ROWS, :], dtype=np.uint8)


def _panel_from_asics(asic_block):
    """Mirror the panel assembly used by psana cbits_config_epix10ka."""
    return np.vstack((
        np.hstack((
            np.flipud(np.fliplr(asic_block[2])),
            np.flipud(np.fliplr(asic_block[1])),
        )),
        np.hstack((
            asic_block[3],
            asic_block[0],
        )),
    ))


def _store_layout_from_asics(active_asics):
    panels = [
        _panel_from_asics(active_asics[i * ASICS_PER_SEGMENT:(i + 1) * ASICS_PER_SEGMENT])
        for i in range(SEGMENT_COUNT)
    ]
    return np.vstack(tuple(panels))


def _panels_from_asics(active_asics):
    return np.stack(tuple(
        _panel_from_asics(active_asics[i * ASICS_PER_SEGMENT:(i + 1) * ASICS_PER_SEGMENT])
        for i in range(SEGMENT_COUNT)
    ))


def _assembled_detector_from_panels(panels):
    top = np.hstack((panels[3], panels[2]))
    bottom = np.hstack((panels[1], panels[0]))
    return np.vstack((top, bottom))


def _cbits_config_panel(asic_block, trbits):
    cbits = _panel_from_asics(asic_block) & 12
    cbits = np.asarray(cbits, dtype=np.uint8)
    rowsh = ASIC_ROWS
    colsh = ASIC_COLS

    if np.all(trbits):
        return np.bitwise_or(cbits, TRBIT_BIT)
    if not np.any(trbits):
        return cbits

    if trbits[2]:
        np.bitwise_or(cbits[:rowsh, :colsh], TRBIT_BIT, out=cbits[:rowsh, :colsh])
    if trbits[3]:
        np.bitwise_or(cbits[rowsh:, :colsh], TRBIT_BIT, out=cbits[rowsh:, :colsh])
    if trbits[0]:
        np.bitwise_or(cbits[rowsh:, colsh:], TRBIT_BIT, out=cbits[rowsh:, colsh:])
    if trbits[1]:
        np.bitwise_or(cbits[:rowsh, colsh:], TRBIT_BIT, out=cbits[:rowsh, colsh:])
    return cbits


def _cbits_config_detector(active_asics, trbits):
    panels = []
    for seg in range(SEGMENT_COUNT):
        start = seg * ASICS_PER_SEGMENT
        stop = start + ASICS_PER_SEGMENT
        panels.append(_cbits_config_panel(active_asics[start:stop], trbits[start:stop]))
    return np.stack(tuple(panels))


def _cbits_config_and_data(cbits_cfg, raw):
    datagainbit = np.bitwise_and(raw, DATA_GAIN_BIT)
    databit05 = np.right_shift(datagainbit, GAIN_BIT_SHIFT)
    return np.bitwise_or(cbits_cfg, databit05), databit05


def _gain_maps_from_cbits(cbits):
    cbits_m60 = cbits & 60
    cbits_m28 = cbits & 28
    cbits_m12 = cbits & 12
    return (
        cbits_m28 == 28,
        cbits_m28 == 12,
        cbits_m12 == 8,
        cbits_m60 == 16,
        cbits_m60 == 0,
        cbits_m60 == 48,
        cbits_m60 == 32,
    )


def _gain_index_from_cbits(cbits):
    gmaps = _gain_maps_from_cbits(cbits)
    return np.select(gmaps, (0, 1, 2, 3, 4, 5, 6), default=10).astype(np.uint16)


def _display_map(arr, base=1000.0, step=1000.0):
    values = sorted(np.unique(arr).tolist())
    display = np.zeros(arr.shape, dtype=np.float32)
    legend = []
    for i, value in enumerate(values):
        shown = base + step * i
        display[arr == value] = shown
        legend.append((value, shown))
    return display, legend


def _save_legend(path, legend):
    lines = ['source_value display_value']
    lines.extend(f'{src} {dst:g}' for src, dst in legend)
    Path(path).write_text('\n'.join(lines) + '\n')


def _stats_lines(name, arr):
    unique, counts = np.unique(arr, return_counts=True)
    lines = [f'{name}: shape={arr.shape} dtype={arr.dtype} size={arr.size}']
    for value, count in zip(unique, counts):
        frac = float(count) / float(arr.size)
        lines.append(f'  value {int(value)}: count={int(count)} fraction={frac:.6%}')
    return lines


def _synthetic_raw_from_active_map(active_asics, gainbit_values):
    active_codes = np.asarray(active_asics & 12, dtype=np.uint8)
    selected = np.isin(active_codes, gainbit_values)
    panels = np.stack(tuple(
        _panel_from_asics(selected[i * ASICS_PER_SEGMENT:(i + 1) * ASICS_PER_SEGMENT])
        for i in range(SEGMENT_COUNT)
    ))
    raw = np.zeros(panels.shape, dtype=np.uint16)
    raw[panels] = DATA_GAIN_BIT
    return raw


def _write_outputs(output_dir, arrays, manifests, legends):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in arrays:
        np.save(output_dir / name, arr)
    for name, lines in manifests:
        Path(output_dir / name).write_text('\n'.join(lines) + '\n')
    for name, legend in legends:
        _save_legend(output_dir / name, legend)


def main():
    args = _parse_args()
    gainbit_values = _parse_value_list(args.gainbit_values)

    cfg, detname = _fetch_configuration(args)
    user_pixel_map = _extract_user_pixel_map(cfg)
    trbits = _extract_trbits(cfg)
    active_asics = _active_user_pixel_map(user_pixel_map)

    store_layout = _store_layout_from_asics(active_asics)
    panels = _panels_from_asics(active_asics)
    assembled = _assembled_detector_from_panels(panels)

    cbits_cfg = _cbits_config_detector(active_asics, trbits)
    gain_index_cfg = _gain_index_from_cbits(cbits_cfg)

    synthetic_raw = _synthetic_raw_from_active_map(active_asics, gainbit_values)
    cbits_total, databit05 = _cbits_config_and_data(cbits_cfg, synthetic_raw)
    gain_index_total = _gain_index_from_cbits(cbits_total)

    active_display, active_legend = _display_map(active_asics & 12)
    store_display, store_legend = _display_map(store_layout & 12)
    assembled_display, assembled_legend = _display_map(assembled & 12)
    cbits_cfg_display, cbits_cfg_legend = _display_map(cbits_cfg)
    gain_index_cfg_display, gain_index_cfg_legend = _display_map(gain_index_cfg)
    databit05_display, databit05_legend = _display_map(databit05)
    cbits_total_display, cbits_total_legend = _display_map(cbits_total)
    gain_index_total_display, gain_index_total_legend = _display_map(gain_index_total)

    manifest = [
        f'detector={detname}',
        f'alias={args.alias}',
        f'instrument={args.inst}',
        f'gainbit_values={list(gainbit_values)}',
        f'user_pixel_map_shape={tuple(user_pixel_map.shape)}',
        f'active_user_pixel_map_shape={tuple(active_asics.shape)}',
        'trbits_per_asic=' + ' '.join(str(int(v)) for v in trbits),
    ]
    for seg in range(SEGMENT_COUNT):
        start = seg * ASICS_PER_SEGMENT
        stop = start + ASICS_PER_SEGMENT
        manifest.append(
            f'trbits_segment_{seg}=' + ' '.join(str(int(v)) for v in trbits[start:stop])
        )

    manifest.extend(_stats_lines('user_pixel_map_all', user_pixel_map))
    manifest.extend(_stats_lines('user_pixel_map_active_mask12', active_asics & 12))
    manifest.extend(_stats_lines('cbits_cfg', cbits_cfg))
    manifest.extend(_stats_lines('gain_index_cfg', gain_index_cfg))
    manifest.extend(_stats_lines('synthetic_databit05', databit05))
    manifest.extend(_stats_lines('cbits_total', cbits_total))
    manifest.extend(_stats_lines('gain_index_total', gain_index_total))

    arrays = [
        ('00_user_pixel_map.npy', user_pixel_map),
        ('01_user_pixel_map_active.npy', active_asics),
        ('02_user_pixel_map_active_display.npy', active_display),
        ('03_store_layout.npy', store_layout),
        ('04_store_layout_display.npy', store_display),
        ('05_assembled_detector.npy', assembled),
        ('06_assembled_detector_display.npy', assembled_display),
        ('07_psana_panel_codes.npy', panels),
        ('08_psana_cbits_cfg.npy', cbits_cfg),
        ('09_psana_cbits_cfg_display.npy', cbits_cfg_display),
        ('10_gain_index_cfg.npy', gain_index_cfg),
        ('11_gain_index_cfg_display.npy', gain_index_cfg_display),
        ('12_synthetic_raw.npy', synthetic_raw),
        ('13_databit05.npy', databit05),
        ('14_databit05_display.npy', databit05_display),
        ('15_psana_cbits_total.npy', cbits_total),
        ('16_psana_cbits_total_display.npy', cbits_total_display),
        ('17_gain_index_total.npy', gain_index_total),
        ('18_gain_index_total_display.npy', gain_index_total_display),
    ]

    manifests = [
        ('manifest.txt', manifest),
    ]

    legends = [
        ('02_user_pixel_map_active_display_legend.txt', active_legend),
        ('04_store_layout_display_legend.txt', store_legend),
        ('06_assembled_detector_display_legend.txt', assembled_legend),
        ('09_psana_cbits_cfg_display_legend.txt', cbits_cfg_legend),
        ('11_gain_index_cfg_display_legend.txt', gain_index_cfg_legend),
        ('14_databit05_display_legend.txt', databit05_legend),
        ('16_psana_cbits_total_display_legend.txt', cbits_total_legend),
        ('18_gain_index_total_display_legend.txt', gain_index_total_legend),
    ]

    output_dir = Path(args.output_dir)
    _write_outputs(output_dir, arrays, manifests, legends)

    print(f'Wrote diagnostics to: {output_dir}')
    print(f'Detector: {detname} alias={args.alias} instrument={args.inst}')
    print(f'Synthetic raw bit14 selected for pixel-map codes: {list(gainbit_values)}')
    print('Key files:')
    print('  00_user_pixel_map.npy           stored config array (16,178,192)')
    print('  03_store_layout.npy             vertically stacked segment layout (1408,384)')
    print('  05_assembled_detector.npy       detector-view assembly (704,768)')
    print('  08_psana_cbits_cfg.npy          config-only cbits in psana panel view (4,352,384)')
    print('  12_synthetic_raw.npy            diagnostic raw with only bit14 set')
    print('  15_psana_cbits_total.npy        config + synthetic raw combined cbits')
    print('  17_gain_index_total.npy         gain index decoded from combined cbits')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
