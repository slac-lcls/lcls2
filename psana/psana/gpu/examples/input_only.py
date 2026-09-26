"""Inspect parsed GPU inputs without calibration or a user callback.

Example: PS_PARALLEL=none python input_only.py mfx100848724 51 jungfrau
         --directory /sdf/data/lcls/ds/prj/public01/xtc

Explicit field.on_cpu copies input data for this diagnostic. There are no
synthetic calib/raw/image results and no automatic output transfers.
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment')
    parser.add_argument('run', type=int)
    parser.add_argument('detector')
    parser.add_argument('--directory')
    parser.add_argument('--algorithm', default='raw')
    parser.add_argument('--field', default='raw')
    parser.add_argument('--events', type=int, default=5)
    args = parser.parse_args()
    from psana import DataSource
    ds = DataSource(exp=args.experiment, run=args.run, dir=args.directory,
                    gpu_det=args.detector, skip_calib_load=[args.detector],
                    max_events=args.events)
    for run in ds.runs():
        for evt in run.events():
            values = evt.gpu.detector(args.detector).field(args.algorithm, args.field).on_cpu
            print(int(evt.timestamp), {segment: (array.shape, str(array.dtype))
                                       for segment, array in values.items()})


if __name__ == '__main__':
    main()
