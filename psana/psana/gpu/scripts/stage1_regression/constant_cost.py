"""Stage 2 constant-store costs on 1/2/4 concurrent BDs sharing one GPU.

No event processing or callback is enabled. CUDA context creation and loading
of the frozen calibration dictionary are outside timing. First and repeated
allocations are retained separately; correctness checks are outside timing.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import pickle
import resource
import time

import numpy as np
from mpi4py import MPI


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--constants', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repetitions', type=int, default=5)
    a = p.parse_args()
    if a.repetitions < 1:
        p.error('--repetitions must be positive')
    comm = MPI.COMM_WORLD
    import cupy as cp
    import psana
    from psana.gpu.gpu_task import RequestedConstants
    from psana.gpu.gpu_budget import _GpuBudget

    assert cp.cuda.runtime.getDeviceCount() == 1
    cp.empty(1)
    cp.cuda.get_current_stream().synchronize()
    with gzip.open(a.constants, 'rb') as stream:
        dictionary = pickle.load(stream)
    gain = dictionary['jungfrau']['pixel_gain'][0].copy(order='C')
    del dictionary
    source = {'jungfrau': {'pixel_gain': (gain, {})}}  # no pedestals
    budget = _GpuBudget.auto(n_bd_ranks=comm.size)
    rows = []
    metrics = dict(peak=0, calls=0, bytes=0)
    from psana.gpu import gpu_allocation
    original_upload = gpu_allocation.upload_owned
    original_hold = budget.hold

    def hold(*args, **kwargs):
        result = original_hold(*args, **kwargs)
        metrics['peak'] = max(metrics['peak'], budget.committed() + budget._held)
        return result

    def upload(cp, arrays, *args, **kwargs):
        arrays = tuple(arrays)
        metrics['calls'] += 1
        metrics['bytes'] += sum(x.nbytes for x in arrays)
        return original_upload(cp, arrays, *args, **kwargs)

    budget.hold = hold
    gpu_allocation.upload_owned = upload

    def measure(case, repetition, store, expected_change, expected_uploads):
        metrics.update(peak=budget.committed(), calls=0, bytes=0)
        comm.Barrier()
        start = time.perf_counter_ns()
        changed = store.refresh(source)
        elapsed = time.perf_counter_ns() - start
        assert changed == expected_change
        assert metrics['calls'] == expected_uploads
        assert metrics['bytes'] == expected_uploads * gain.nbytes
        assert budget._held == 0 and budget.committed() <= budget.limit()
        rows.append(dict(case=case, repetition=repetition, wall_ns=elapsed,
                         committed=budget.committed(), peak_owned_and_held=metrics['peak'],
                         upload_calls=metrics['calls'], upload_bytes=metrics['bytes'],
                         rss_bytes=int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE'),
                         max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        if expected_uploads:
            assert store.get('jungfrau', 'pixel_gain').shape == gain.shape
            assert store.get('jungfrau', 'pixel_gain').dtype == gain.dtype
            assert float(cp.asnumpy(store.get('jungfrau', 'pixel_gain').reshape(-1)[0])) == float(gain.flat[0])

    for repetition in range(1, a.repetitions + 1):
        empty = RequestedConstants((), budget)
        measure('empty_setup', repetition, empty, False, 0)
        empty.close()
        store = RequestedConstants([('jungfrau', 'pixel_gain')], budget)
        measure('gain_setup', repetition, store, True, 1)
        measure('unchanged_refresh', repetition, store, False, 0)
        assert np.isfinite(gain.flat[0])
        gain.flat[0] = np.nextafter(gain.flat[0], np.float32(np.inf))
        measure('changed_refresh', repetition, store, True, 1)
        store.close()
        assert budget.committed() == budget._held == 0

    record = dict(rank=comm.rank, rows=rows, shape=gain.shape, dtype=str(gain.dtype),
                  bytes=gain.nbytes, budget_limit=budget.limit(),
                  cupy=cp.__version__, cuda=cp.cuda.runtime.runtimeGetVersion(),
                  device_bus=cp.cuda.runtime.deviceGetPCIBusId(0), psana=psana.__file__)
    if isinstance(record['device_bus'], bytes):
        record['device_bus'] = record['device_bus'].decode()
    records = comm.gather(record, root=0)
    if comm.rank == 0:
        assert len({r['device_bus'] for r in records}) == 1
        result = dict(complete=True, bds=comm.size, repetitions=a.repetitions, ranks=records,
                      scope='constant store only; idle/drained users; no I/O, callback or cache trimming')
        a.output.write_text(json.dumps(result, indent=2) + '\n')
        print(f'CONSTANT_COST_OK bds={comm.size} repetitions={a.repetitions}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
        raise
