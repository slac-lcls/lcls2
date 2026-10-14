"""Numerical and stream acceptance for the user-owned batched calibration."""
import numpy as np
import pytest

from test_gpu_allocation_device import available
from psana.gpu.examples.jungfrau_calibration import JungfrauCalibration
from psana.tests.gpu.user_calibration_reference import calibrate

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not available(), reason='no CUDA device')]


class Batch:
    # Only the public producer-context protocol is exposed to the user callable.
    def __init__(self, raw, present, segments, constants):
        self.raw, self.presence, self.segments, self.constants = raw, present, segments, constants
        self.size = len(raw)
        self.outputs = {}
    def input(self, name): return self.raw
    def present(self, name): return self.presence
    def segment_ids(self, name): return self.segments
    def calibconst(self, det, key): return self.constants[key]
    def publish(self, name, value): self.outputs[name] = value


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('extra_bits', [0, (1 << 64) - 1])
@pytest.mark.parametrize('offset,status_bits', [(False, 0), (True, 0), (True, 0xffff), (False, 1 << 40)])
def test_sparse_reordered_segments_modes_masks_offsets_and_tails(dtype, offset, status_bits, extra_bits, monkeypatch):
    import cupy as cp
    rng = np.random.default_rng(5)
    segments = (3, 1)
    raw = rng.integers(0, 65536, (5, 2, 3, 7), dtype=np.uint16)
    raw.flat[:4] = [100, 0x4064, 0x8064, 0xc064]
    presence = np.ones((5, 2), np.uint8); presence[1, 1] = 0; presence[3] = 0
    shape = (3, 4, 3, 7)
    constants = dict(pedestals=rng.uniform(-30, 30, shape).astype(dtype),
                     pixel_gain=rng.uniform(0.1, 10, shape).astype(dtype),
                     pixel_offset=rng.uniform(-1, 1, shape).astype(dtype),
                     pixel_status=np.zeros(shape, np.uint64),
                     status_extra=np.zeros(shape, np.uint64))
    constants['pixel_gain'][:, 3, 0, 0] = 0
    constants['pixel_status'][2, 1, 0, :3] = [1, 2, 1 << 40]
    constants['status_extra'][1, 3, 1, :2] = [4, 1 << 40]
    expected = calibrate(raw, presence, segments, constants, use_offset=offset, status_bits=status_bits, stextra_bits=extra_bits)
    user = JungfrauCalibration(use_offset=offset, status_bits=status_bits, stextra_bits=extra_bits)
    launches = []
    original = cp.RawKernel
    def counted(*args, **kwargs):
        kernel = original(*args, **kwargs)
        def launch(grid, block, argv, **kw):
            launches.append((int(argv[7]), kw['stream'].ptr))
            return kernel(grid, block, argv, **kw)
        return launch
    monkeypatch.setattr(cp, 'RawKernel', counted)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        batch = Batch(cp.asarray(raw), cp.asarray(presence), segments,
                      {k: cp.asarray(v) for k, v in constants.items()})
        user(batch, stream)
    stream.synchronize()
    np.testing.assert_array_equal(batch.outputs[user.output].get(), expected)
    assert launches == [(raw.size, stream.ptr)]
    assert (user.calls, user.events) == (1, 5)


def test_overlapping_streams_new_constants_and_retained_outputs():
    import cupy as cp
    user = JungfrauCalibration()
    delay = cp.RawKernel(r'''extern "C" __global__ void delay(unsigned long long ticks) {
        unsigned long long begin = clock64(); while (clock64() - begin < ticks) {}
    }''', 'delay')
    delay.compile()
    saved = []
    for n, pedestal in ((5, 1), (3, 7), (1, 2)):
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            raw = cp.full((n, 1, 3, 4), 20, cp.uint16)
            constants = dict(pedestals=cp.full((3, 1, 3, 4), pedestal, cp.float32),
                             pixel_gain=cp.full((3, 1, 3, 4), 2, cp.float32))
            batch = Batch(raw, cp.ones((n, 1), cp.uint8), (0,), constants)
            delay((1,), (1,), (np.uint64(20000000),), stream=stream)
            user(batch, stream)
            saved.append((stream, batch, (20-pedestal)/2))
    for stream, batch, expected in reversed(saved):
        stream.synchronize()
        np.testing.assert_array_equal(batch.outputs[user.output].get(), expected)
    assert len(user._kernels) == 1 and user.calls == 3 and user.events == 9


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_matches_actual_cpu_v3_with_rounding_and_nonfinite_constants(dtype):
    """Use the real C++ v3 entry point, with its NumPy-prepared constants."""
    import cupy as cp
    from psana.pycalgos.utilsdetector import calib_jungfrau_v3
    rng = np.random.default_rng(42)
    raw = rng.integers(0, 65536, (3, 1, 2, 17), dtype=np.uint16)
    raw[:, :, :, :8] &= 0x3fff
    shape = (3, 1, 2, 17)
    peds = rng.uniform(1000, 4000, shape).astype(dtype)
    offset = rng.uniform(-1, 1, shape).astype(dtype)
    gain = rng.uniform(0.1, 10, shape).astype(dtype)
    gain[:, :, :, 0] = 0
    gain[:, :, :, 1] = np.inf
    peds[:, :, :, 2] = np.nan
    # Exposes premature float32 conversion in the old kernel.
    gain[:, :, :, 3] = 3.0000001
    peds[:, :, :, 4] = 1000.00003
    offset[:, :, :, 4] = 0.00003
    status = np.zeros(shape, np.uint64)
    status[2, :, :, 2] = 1  # CPU preserves NaN * zero for masked nonfinite pixels.
    extra = np.zeros(shape, np.uint64)
    extra[1, :, :, 5] = 1 << 40
    constants = dict(pedestals=peds, pixel_gain=gain, pixel_offset=offset,
                     pixel_status=status, status_extra=extra)
    poff = peds + offset
    gfac = np.divide(np.ones_like(peds), np.where(gain != 0, gain, 1))
    gfac[gain == 0] = 0
    mask = np.all((status | extra) == 0, axis=0)
    cc = np.zeros((4, raw[0].size, 2), np.float32)
    for mode, code in enumerate((0, 1, 3)):
        cc[code, :, 0] = poff[mode].ravel()
        cc[code, :, 1] = (gfac[mode] * mask).ravel()
    expected = np.empty(raw.shape, np.float32)
    for i in range(len(raw)):
        calib_jungfrau_v3(raw[i], cc.ravel(), raw[i].size, expected[i])
    user = JungfrauCalibration(use_offset=True, status_bits=(1 << 64)-1, stextra_bits=(1 << 64)-1)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        batch = Batch(cp.asarray(raw), cp.ones(raw.shape[:2], cp.uint8), (0,),
                      {k: cp.asarray(v) for k, v in constants.items()})
        user(batch, stream)
    stream.synchronize()
    np.testing.assert_array_equal(batch.outputs[user.output].get(), expected)
