import argparse
import os

import numpy as np

from psdaq.configdb.epixquad_cdict import epixquad_cdict


gain_dict = {
    'H': {'value': 0xc, 'trbit': 1},
    'M': {'value': 0xc, 'trbit': 0},
    'L': {'value': 0x8, 'trbit': 0},
    'AHL': {'value': 0x0, 'trbit': 1},
    'AML': {'value': 0x0, 'trbit': 0},
}

STORE_ROWS = 1408
STORE_COLS = 384
ASIC_ROWS = 176
ASIC_COLS = 192
ASIC_COUNT = 16


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


def _read_store_layout(fname):
    labels = np.genfromtxt(fname, dtype=np.uint8)
    if labels.shape != (STORE_ROWS, STORE_COLS):
        raise ValueError(
            f'Expected gain-map file shape {(STORE_ROWS, STORE_COLS)}, got {labels.shape} from {fname}'
        )
    return labels


def _store_layout_to_asics(store_labels):
    elems = np.vsplit(store_labels, 4)
    asics = []
    for elem in elems:
        quadrants = []
        for half in np.vsplit(elem, 2):
            quadrants.extend(np.hsplit(half, 2))
        asics.extend([
            np.asarray(quadrants[3], dtype=np.uint8),
            np.flipud(np.fliplr(quadrants[1])),
            np.flipud(np.fliplr(quadrants[0])),
            np.asarray(quadrants[2], dtype=np.uint8),
        ])
    return asics


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


def _legacy_binary_map(store_labels, gains):
    if gain_dict[gains[0]]['trbit'] != gain_dict[gains[1]]['trbit']:
        raise ValueError(f'Incompatible gains {gains} for pixel configuration')

    vgain0 = gain_dict[gains[0]]['value']
    vgain1 = gain_dict[gains[1]]['value']
    unique_labels = sorted(int(v) for v in np.unique(store_labels))
    if any(label not in (0, 1) for label in unique_labels):
        raise ValueError(
            f'Legacy --gain mode only supports binary masks with labels 0/1, got {unique_labels}'
        )

    print(f'Legacy gain mode: 0 -> {gains[0]}, 1 -> {gains[1]}')
    mapped = store_labels * (vgain1 - vgain0) + vgain0

    d = {}
    trbit = gain_dict[gains[0]]['trbit']
    for i in range(ASIC_COUNT):
        d[f'expert.EpixQuad.Epix10kaSaci{i}.trbit'] = trbit

    pixel_maps = _store_layout_to_asics(mapped.astype(np.uint8))
    d['user.pixel_map'] = np.asarray(np.pad(pixel_maps, ((0, 0), (0, 2), (0, 0))), dtype=np.uint8)
    return d


def _label_map_to_pixel_map(store_labels, label_map):
    label_asics = _store_layout_to_asics(store_labels)
    pixel_asics = []
    d = {}

    for i, asic_labels in enumerate(label_asics):
        labels_present, label_counts = np.unique(asic_labels, return_counts=True)
        labels_present = [int(v) for v in labels_present]
        counts_dict = {int(label): int(count) for label, count in zip(labels_present, label_counts)}

        missing = [label for label in labels_present if label not in label_map]
        if missing:
            raise ValueError(f'ASIC {i} uses unmapped labels {missing}; add --map entries for them')

        gains_present = sorted({label_map[label] for label in labels_present})
        trbits = {gain_dict[gain]['trbit'] for gain in gains_present}
        if len(trbits) != 1:
            raise ValueError(
                f'ASIC {i} mixes gains with incompatible trbit values: '
                f'{gains_present} from labels {labels_present}'
            )

        trbit = trbits.pop()
        d[f'expert.EpixQuad.Epix10kaSaci{i}.trbit'] = trbit

        pixel_asic = np.zeros_like(asic_labels, dtype=np.uint8)
        for label in labels_present:
            gain = label_map[label]
            pixel_asic[asic_labels == label] = gain_dict[gain]['value']
        pixel_asics.append(pixel_asic)

        print(
            f'ASIC {i:02d}: labels [{_summary_from_counts(counts_dict)}] -> '
            f'gains {gains_present}, trbit={trbit}'
        )

    d['user.pixel_map'] = np.asarray(np.pad(pixel_asics, ((0, 0), (0, 2), (0, 0))), dtype=np.uint8)
    return d


def epixquad_readmap(fname, gains=None, label_map=None):
    store_labels = _read_store_layout(fname)
    print(f'Read {fname} with shape {store_labels.shape} and labels {sorted(int(v) for v in np.unique(store_labels))}')

    if label_map is not None:
        return _label_map_to_pixel_map(store_labels, label_map)
    if gains is None:
        raise ValueError('Either gains or label_map must be provided')
    return _legacy_binary_map(store_labels, gains)


def main():
    parser = argparse.ArgumentParser(description='Update epixquad gain map in configdb')
    parser.add_argument('--file', help='input pixel mask in store layout', type=str, required=True)
    parser.add_argument(
        '--gain',
        help='legacy binary mask gains for labels [0 1]',
        default=['M', 'L'],
        nargs=2,
        choices=gain_dict.keys(),
    )
    parser.add_argument(
        '--map',
        action='append',
        default=[],
        metavar='LABEL:GAIN',
        help='label-to-gain mapping for general masks, may be repeated (example: --map 0:H --map 1:AHL --map 2:M)',
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
    if label_map is not None:
        print(f'Label map mode: {label_map}')
        d = epixquad_readmap(args.file, label_map=label_map)
    else:
        d = epixquad_readmap(args.file, gains=args.gain)

    copyValues(d, top)

    print('Setting user.pixel_map')
    top.set('user.pixel_map', d['user.pixel_map'])

    top.setInfo('epix10kaquad', args.name, args.segm, args.id, 'No comment')
    if not args.test:
        mycdb.modify_device(args.alias, top)


if __name__ == '__main__':
    main()
