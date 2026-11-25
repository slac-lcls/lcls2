"""Check GPU-resident ECC banks; transfer only counts and scalar check flags."""
import math
import time
import numpy as np
from support import require

SHAPE = (100, 32, 512, 1024)
CHECK_ELEMENTS = 1 << 20


def array_checks(cp, sums, counts, *, shape=SHAPE):
    """Bound temporary predicates to 1 MiB; inspect every pixel on the device.

    The shape override supports small CPU behavioral tests. Production callers
    always use the full default shape. No cube pixels are copied to the host.
    """
    require(tuple(sums.shape) == shape, 'Cube shape')
    require(sums.dtype == np.dtype('float32'), 'Cube dtype')
    require(sums.nbytes == math.prod(shape) * 4, 'Cube bytes')
    require(sums.flags.c_contiguous, 'Cube contiguous layout')
    require(len(counts) == shape[0], 'Cube count length')
    records = []
    for index, count in enumerate(counts):
        flat = sums[index].reshape(-1)
        nonzero = False
        for offset in range(0, flat.size, CHECK_ELEMENTS):
            chunk = flat[offset:offset + CHECK_ELEMENTS]
            require(bool(cp.isfinite(chunk).all().item()), 'Nonfinite cube bin %d' % index)
            present = bool(cp.any(chunk != 0).item())
            nonzero = nonzero or present
        require(nonzero == (count > 0), 'Cube bin %d zero/nonzero occupancy' % index)
        records.append(dict(bin=index, finite=True, nonzero=nonzero))
    return records


def inspect_cube(cube, delivered):
    import cupy as cp
    started = time.monotonic()
    require(cube is not None and cube.phase == 'open' and cube.bank is not None,
            'Prepared open cube required')
    require(cube.seen == set(delivered) and len(set(delivered)) == len(delivered),
            'Cube delivered identities')
    cube.drain()
    require(not cube._pending and cube._completion is None, 'Cube drained owners')
    bank = cube.bank
    for array in (bank.sums, bank.counts):
        require(isinstance(array, cp.ndarray) and array.device.id == cp.cuda.runtime.getDevice(),
                'Cube array device')
    require(bank.counts.shape == (100,) and bank.counts.dtype == np.dtype('uint64'),
            'Cube count layout')
    expected = [0] * 100
    ordinal = {stamp: i for i, stamp in enumerate(cube.timestamps)}
    for stamp in delivered:
        require(stamp in ordinal, 'Cube unknown timestamp')
        expected[ordinal[stamp] % 99] += 1
    with bank.stream:
        counts = bank.counts.get(stream=bank.stream, blocking=True).tolist()
        require(counts == expected, 'Cube event counts')
        bins = array_checks(cp, bank.sums, counts)
        bank.stream.synchronize()
    return dict(status='pass', shape=list(bank.sums.shape), dtype=str(bank.sums.dtype),
                bytes=bank.sums.nbytes, counts_shape=list(bank.counts.shape),
                counts_dtype=str(bank.counts.dtype), counts=counts, bins=bins,
                drained=True, cube_payload_host_bytes=0, exported=False,
                reference_checked=False, seconds=time.monotonic()-started)
