"""Explicit MPI acceptance driver for Stage 1b (run outside pytest).

First run with PS_PARALLEL=none and mode reference, then with four MPI ranks
in exclusive and hybrid modes. All three invocations share the reference JSON.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main():
    mode, reference_path = sys.argv[1:]
    from psana import DataSource
    options = dict(exp='mfx100848724', run=51, dir='/sdf/data/lcls/ds/prj/public01/xtc',
                   detectors=['jungfrau'], max_events=13, batch_size=5)
    if mode == 'reference':
        run = next(DataSource(**options).runs())
        det = run.Detector('jungfrau')
        reference = {}
        for evt in run.events():
            raw = det.raw.raw(evt)
            calib = np.asarray(det.raw.calib(evt, cmpars=None), np.float32)
            assert raw is not None and np.any(calib)
            reference[str(int(evt.timestamp))] = dict(raw=digest(raw), calib=digest(calib))
        assert len(reference) == 13
        Path(reference_path).write_text(json.dumps(reference))
        print('REFERENCE_OK events=13', flush=True)
        return

    from mpi4py import MPI
    from psana.psexp.mpi_ds import RunParallel
    comm = MPI.COMM_WORLD
    assert comm.size == 4
    targets = []
    original = RunParallel._iter_jungfrau_raw

    def checked(self, area_only=False):
        result = original(self, area_only=area_only)
        names = [entry[0] for entry in result]
        assert ('jungfrau' in names) == (mode == 'hybrid'), (mode, names)
        targets.append((area_only, names))
        return result
    RunParallel._iter_jungfrau_raw = checked
    options.update(n_gpu_streams=2, gpu_bulk_read=True)
    options['hybrid_det' if mode == 'hybrid' else 'gpu_det'] = 'jungfrau'
    reference = json.loads(Path(reference_path).read_text())
    seen = []
    for run in DataSource(**options).runs():
        det = run.Detector('jungfrau')
        for evt in run.events():
            stamp = str(int(evt.timestamp))
            assert not evt.gpu._gpu_results
            fields = evt.gpu.detector('jungfrau').field('raw', 'raw').on_cpu
            raw = np.stack([fields[s].reshape(512, 1024) for s in fields.segment_ids])
            assert digest(raw) == reference[stamp]['raw']
            if mode == 'hybrid':
                assert digest(np.asarray(det.raw.calib(evt, cmpars=None), np.float32)) == reference[stamp]['calib']
            seen.append(stamp)
    summaries = comm.gather(dict(rank=comm.rank, seen=seen, targets=targets), root=0)
    if comm.rank == 0:
        all_seen = [stamp for item in summaries for stamp in item['seen']]
        assert len(all_seen) == 13 and set(all_seen) == set(reference)
        assert all(item['targets'] for item in summaries), summaries
        assert sum(bool(item['seen']) for item in summaries) == 2, summaries
        print('MPI_INPUT_OK '+json.dumps(dict(mode=mode, ranks=summaries)), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        if sys.argv[1] != 'reference':
            import traceback
            from mpi4py import MPI
            traceback.print_exc()
            MPI.COMM_WORLD.Abort(1)
        raise
