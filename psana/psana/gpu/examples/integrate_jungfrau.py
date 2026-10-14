"""External batched calibration/integration driver.

Copy this file, jungfrau_calibration.py and jungfrau_azimuthal_integration.py to
one user directory. --bins is an explicit NumPy .npz containing bin_ids (P,H,W)
and edges (nbins+1,). Values -1 exclude pixels; other values index the bins.
Prepare geometry, beam parameters and bin edges in user code before this run.
"""
import argparse
from contextlib import closing
import numpy as np
from jungfrau_azimuthal_integration import JungfrauAzimuthalIntegration


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment')
    p.add_argument('run', type=int)
    p.add_argument('--directory')
    p.add_argument('--bins', required=True)
    p.add_argument('--events', type=int, default=13)
    p.add_argument('--batch-size', type=int, default=5)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--offset', action='store_true')
    p.add_argument('--status-bits', type=lambda s: int(s, 0), default=0)
    p.add_argument('--stextra-bits', type=lambda s: int(s, 0), default=0)
    a = p.parse_args()
    if min(a.events, a.batch_size, a.depth) <= 0:
        p.error('events, batch-size and depth must be positive')
    with np.load(a.bins, allow_pickle=False) as bins:
        edges = bins['edges']
        if edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0):
            p.error('edges must be finite and strictly increasing')
        analysis = JungfrauAzimuthalIntegration(bins['bin_ids'], len(edges)-1,
            use_offset=a.offset, status_bits=a.status_bits, stextra_bits=a.stextra_bits)
    from psana import DataSource
    from psana.gpu import GpuTask
    task = GpuTask(analysis, inputs=analysis.inputs, calibconst=analysis.calibconst)
    ds = DataSource(exp=a.experiment, run=a.run, dir=a.directory, detectors=['jungfrau'],
                    gpu_det='jungfrau', gpu_fn=task, batch_size=a.batch_size,
                    n_gpu_streams=a.depth, max_events=a.events)
    for run in ds.runs():
        with closing(run.events()) as events:
            for event in events:
                mean, sums, counts = event.gpu.get(analysis.output).on_cpu
                print(int(event.timestamp), 'valid_pixels', int(counts.sum()),
                      'intensity_sum', float(sums.sum()), flush=True)
    if analysis.calls:
        print(f'integration callbacks={analysis.calls}, events={analysis.events}', flush=True)


if __name__ == '__main__':
    main()
