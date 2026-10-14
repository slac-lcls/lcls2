"""Separate synthetic GPU timing probe; does not alter the measured install."""
import json
import numpy as np
import cupy as cp
from psana.gpu import gpu_detector as gd
from psana.gpu.gpudgram import parser as p

nevents, nsegments, nstreams, pixels = 20, 32, 5, 512 * 1024
ndgrams = nevents * nstreams
tiles = (pixels + 255) // 256
raw = cp.arange(nevents * nsegments * pixels, dtype=cp.uint16).view(cp.uint8)
outputs = [cp.empty(nevents * nsegments * pixels, dtype=cp.uint16) for _ in range(3)]
present = cp.empty(nevents * nsegments, dtype=cp.uint8)
loc = np.zeros((nsegments, ndgrams, p.LOC_NCOLS), dtype=np.uint64)
plan = np.zeros((nsegments, 4), dtype=np.uint64)
for segment in range(nsegments):
    plan[segment] = segment % nstreams, segment, 1, 2
    for event in range(nevents):
        row = loc[segment, event * nstreams + segment % nstreams]
        row[p.LOC_STATUS] = p.STATUS_FOUND
        row[p.LOC_TYPE], row[p.LOC_RANK] = 1, 2
        row[p.LOC_NBYTES] = pixels * 2
        row[p.LOC_OFFSET] = (event * nsegments + segment) * pixels * 2
loc = cp.asarray(loc)
plan = cp.asarray(plan)
rows = cp.arange(ndgrams, dtype=cp.int64)
base = gd._batched_gather_kernel(cp.dtype(cp.uint16))
code = base.code
code32 = code.replace('(unsigned long long)blockIdx.x / tiles', 'blockIdx.x / (unsigned int)tiles')
code32 = code32.replace('(unsigned long long)blockIdx.x % tiles', 'blockIdx.x % (unsigned int)tiles')
code32 = code32.replace('const unsigned long long row =', 'const unsigned int row =')
code32 = code32.replace('row % n_segments', 'row % (unsigned int)n_segments').replace('row / n_segments', 'row / (unsigned int)n_segments')
code3 = code.replace('(unsigned long long)blockIdx.x / tiles', '(unsigned long long)blockIdx.z * n_segments + blockIdx.y')
code3 = code3.replace('((unsigned long long)blockIdx.x % tiles)', '((unsigned long long)blockIdx.x)')
code3 = code3.replace('(row % n_segments)', 'blockIdx.y').replace('(row / n_segments)', 'blockIdx.z')
kernels = [base, cp.RawKernel(code32, 'gather_canonical_u16', options=('--std=c++17',)),
           cp.RawKernel(code3, 'gather_canonical_u16', options=('--std=c++17',))]
grids = [(tiles * nevents * nsegments,), (tiles * nevents * nsegments,), (tiles, nsegments, nevents)]
cp.cuda.Stream.null.synchronize()
measurements = {}
for name, kernel, grid, output in zip(('flat64', 'flat32', 'grid3d'), kernels, grids, outputs):
    args = (raw, np.uint64(raw.nbytes), loc, np.uint64(ndgrams), np.uint64(ndgrams), plan,
            rows, np.uint64(nstreams), np.uint64(nsegments), np.uint64(pixels), np.uint64(tiles), output, present)
    kernel(grid, (256,), args)
    cp.cuda.Stream.null.synchronize()
    elapsed = []
    for _ in range(8):
        start, end = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        kernel(grid, (256,), args)
        end.record()
        end.synchronize()
        elapsed.append(cp.cuda.get_elapsed_time(start, end))
    assert bool(cp.all(output == raw.view(cp.uint16)).get())
    assert bool(cp.all(present == 1).get())
    measurements[name] = elapsed
print('GEOMETRY_GPU_MS ' + json.dumps(measurements), flush=True)
