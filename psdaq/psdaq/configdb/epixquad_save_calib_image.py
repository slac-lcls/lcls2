#!/usr/bin/env python3

"""Save representative epixquad calib/image arrays for manual mask editing."""

import argparse
from pathlib import Path
import re

import numpy as np
from psana import DataSource


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Save one epixquad calib array, assembled image, and optional geometry for Mask Editor'
    )
    parser.add_argument('-e', '--exp', required=True, help='experiment code, for example ued1015999')
    parser.add_argument('-r', '--run', required=True, type=int, help='run number')
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
        '--event-index',
        type=int,
        default=0,
        help='0-based index among valid detector events to save (default: 0)',
    )
    parser.add_argument(
        '--max-nevents',
        type=int,
        default=100,
        help='maximum number of events to scan while searching for a valid event (default: 100)',
    )
    parser.add_argument(
        '-p',
        '--path',
        default='.',
        help='output directory to create/use for generated files (default: current directory)',
    )
    parser.add_argument(
        '--write-geometry',
        action='store_true',
        help='also write the geometry text used by psana in .data format for Mask Editor',
    )
    return parser.parse_args()


def _sanitize_label(text):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', text)


def _output_stem(output_dir, runnum, detname, event_index):
    det_label = _sanitize_label(detname)
    return output_dir / f'epixquad_mask_editor_r{runnum}_e{event_index:03d}_{det_label}'


def _geometry_text(det):
    geotxt, _meta = det.raw._det_geotxt_and_meta()
    if geotxt is not None:
        return geotxt, 'db'
    return det.raw._det_geotxt_default(), 'default'


def main():
    args = _parse_args()
    if args.event_index < 0:
        raise ValueError(f'event-index must be non-negative, got {args.event_index}')
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

    calib = None
    image = None
    selected_good_event = None

    ngood = 0
    for evt in run.events():
        calib_candidate = det.raw.calib(evt)
        if calib_candidate is None:
            continue
        if ngood == args.event_index:
            calib = np.asarray(calib_candidate)
            image = det.raw.image(evt, nda=calib)
            selected_good_event = ngood
            break
        ngood += 1

    if calib is None:
        raise RuntimeError(
            f'No valid detector event found for event-index={args.event_index} within max-nevents={args.max_nevents}'
        )
    if image is None:
        raise RuntimeError('det.raw.image(evt, nda=calib) returned None; geometry may be unavailable')

    stem = _output_stem(output_dir, args.run, args.detname, selected_good_event)

    calib_path = Path(f'{stem}_calib.npy')
    np.save(calib_path, calib)
    print(calib_path)
    print(f'calib shape {calib.shape}, dtype {calib.dtype}')

    image_path = Path(f'{stem}_img.npy')
    np.save(image_path, image)
    print(image_path)
    print(f'image shape {image.shape}, dtype {image.dtype}')

    if args.write_geometry:
        geotxt, source = _geometry_text(det)
        geometry_path = Path(f'{stem}_geometry.data')
        geometry_path.write_text(geotxt)
        print(geometry_path)
        print(f'geometry source {source}')


if __name__ == '__main__':
    main()
