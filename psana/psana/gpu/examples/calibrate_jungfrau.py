"""External-driver example: batched calibration through the public GpuTask API.

Copy this script and jungfrau_calibration.py to a user directory. For example:
  PS_PARALLEL=none python calibrate_jungfrau.py mfx100848724 51 \
    --directory /sdf/data/lcls/ds/prj/public01/xtc --events 13 --batch-size 5

The default uses pedestals/gains only. --offset and --status-bits 0xffff opt
into additional constants; missing requested keys fail explicitly. No common
mode, geometry, edge or neighbor mask is applied. --stextra-bits selects
status_extra bits. For default CPU-v3 parity, enable available offsets and all
bits of both available status arrays (0xffffffffffffffff). Full calibrated images are
published for this Stage 5a example; output-copy cost belongs in its timings.
"""
import argparse
from contextlib import closing

from jungfrau_calibration import JungfrauCalibration


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment')
    p.add_argument('run', type=int)
    p.add_argument('--directory')
    p.add_argument('--detector', default='jungfrau')
    p.add_argument('--events', type=int, default=13)
    p.add_argument('--batch-size', type=int, default=5)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--offset', action='store_true')
    p.add_argument('--status-bits', type=lambda text: int(text, 0), default=0)
    p.add_argument('--stextra-bits', type=lambda text: int(text, 0), default=0)
    a = p.parse_args()
    if min(a.events, a.batch_size, a.depth) <= 0:
        p.error('events, batch-size and depth must be positive')
    from psana import DataSource
    from psana.gpu import GpuTask
    analysis = JungfrauCalibration(a.detector, use_offset=a.offset, status_bits=a.status_bits,
                                   stextra_bits=a.stextra_bits)
    task = GpuTask(analysis, inputs=analysis.inputs, calibconst=analysis.calibconst)
    ds = DataSource(exp=a.experiment, run=a.run, dir=a.directory,
                    detectors=[a.detector], gpu_det=a.detector, gpu_fn=task,
                    batch_size=a.batch_size, n_gpu_streams=a.depth, max_events=a.events)
    for run in ds.runs():
        with closing(run.events()) as events:
            for event in events:
                image = event.gpu.get(analysis.output).on_cpu
                print(int(event.timestamp), image.shape, float(image.sum(dtype='float64')),
                      flush=True)
    if analysis.calls:
        print(f'calibration callbacks={analysis.calls}, events={analysis.events}', flush=True)


if __name__ == '__main__':
    main()
