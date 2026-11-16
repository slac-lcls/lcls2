"""Exercise content checks on small arrays, including failures after chunk one."""
import numpy as np
import pytest
import cube_check


def fixture():
    sums = np.zeros((100, 1, 2, 3), np.float32)
    sums[0, 0, 0, 0] = 2
    counts = [1] + [0] * 99
    return sums, counts


def check(sums, counts):
    return cube_check.array_checks(np, sums, counts, shape=(100, 1, 2, 3))


def test_valid_partial_zero_and_empty_bins(monkeypatch):
    monkeypatch.setattr(cube_check, 'CHECK_ELEMENTS', 2)
    sums, counts = fixture()
    records = check(sums, counts)
    assert records == [dict(bin=i, finite=True, nonzero=i == 0) for i in range(100)]


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('bin_index', [0, 99])
def test_nonfinite_after_first_chunk_rejected(monkeypatch, value, bin_index):
    monkeypatch.setattr(cube_check, 'CHECK_ELEMENTS', 2)
    sums, counts = fixture()
    sums[bin_index, 0, 1, 2] = value
    with pytest.raises(ValueError, match='Nonfinite'):
        check(sums, counts)


@pytest.mark.parametrize('bin_index,value', [(0, 0), (99, 3)])
def test_wrong_occupancy_rejected(bin_index, value):
    sums, counts = fixture()
    sums[bin_index, 0, 0, 0] = value
    with pytest.raises(ValueError, match='occupancy'):
        check(sums, counts)


def test_bad_shape_and_dtype():
    sums, counts = fixture()
    with pytest.raises(ValueError, match='shape'):
        cube_check.array_checks(np, sums, counts)
    with pytest.raises(ValueError, match='dtype'):
        check(sums.astype(np.float64), counts)


def test_empty_owner_valid():
    sums, counts = fixture()
    sums.fill(0)
    counts[0] = 0
    assert not any(row['nonzero'] for row in check(sums, counts))


@pytest.mark.parametrize('corruption', [None, 'counts', 'device', 'identity', 'drain', 'phase', 'bank',
                                         'unknown', 'duplicate', 'counts_dtype', 'counts_shape'])
def test_inspection_transfers_only_counts(monkeypatch, corruption):
    import sys
    from types import SimpleNamespace
    class Stream:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def synchronize(self): pass
    class DeviceArray:
        def __init__(self, counts=False):
            self.is_counts = counts
            self.shape = (100,) if counts else cube_check.SHAPE
            self.dtype = np.dtype('uint64' if counts else 'float32')
            self.nbytes = 800 if counts else 100*32*512*1024*4
            self.device = SimpleNamespace(id=0)
        def get(self, **kwargs):
            assert self.is_counts, 'Cube payload must never leave GPU'
            assert kwargs == dict(stream=stream, blocking=True)
            result = np.zeros(100, np.uint64)
            result[0] = 2 if corruption == 'counts' else 1
            return result
    stream = Stream()
    bank = SimpleNamespace(stream=stream, sums=DeviceArray(), counts=DeviceArray(True))
    cube = SimpleNamespace(phase='open', bank=bank, timestamps=[123], seen={123},
                           _pending=(object(),), _completion=object())
    def drain():
        if corruption != 'drain': cube._pending, cube._completion = (), None
    cube.drain = drain
    if corruption == 'device': bank.sums.device.id = 1
    if corruption == 'identity': cube.seen = set()
    if corruption == 'phase': cube.phase = 'failed'
    if corruption == 'bank': cube.bank = None
    if corruption == 'unknown': cube.timestamps = [456]
    if corruption == 'counts_dtype': bank.counts.dtype = np.dtype('float64')
    if corruption == 'counts_shape': bank.counts.shape = (99,)
    delivered = [123, 123] if corruption == 'duplicate' else [123]
    cp = SimpleNamespace(ndarray=DeviceArray, cuda=SimpleNamespace(runtime=SimpleNamespace(getDevice=lambda:0)))
    monkeypatch.setitem(sys.modules, 'cupy', cp)
    calls = []
    def checks(actual_cp, sums, counts):
        assert actual_cp is cp and sums is bank.sums and counts == [1]+[0]*99
        calls.append(True)
        return [dict(bin=i, finite=True, nonzero=i==0) for i in range(100)]
    monkeypatch.setattr(cube_check, 'array_checks', checks)
    if corruption:
        with pytest.raises(ValueError): cube_check.inspect_cube(cube, delivered)
        assert not calls
    else:
        receipt = cube_check.inspect_cube(cube, delivered)
        assert calls == [True]
        from audit import audit_cube
        audit_cube(receipt, [123], [123])


def test_device_like_reductions_are_scalar_and_chunk_bounded(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(cube_check, 'CHECK_ELEMENTS', 2)
    slices, reductions, host_flags = [], [], []
    class Array:
        def __init__(self, data): self.data = data
        @property
        def shape(self): return self.data.shape
        @property
        def dtype(self): return self.data.dtype
        @property
        def nbytes(self): return self.data.nbytes
        @property
        def flags(self): return self.data.flags
        @property
        def size(self): return self.data.size
        def __getitem__(self, key):
            result = Array(self.data[key])
            if isinstance(key, slice): slices.append(result.size)
            return result
        def reshape(self, *args): return Array(self.data.reshape(*args))
        def __ne__(self, value):
            assert self.size <= cube_check.CHECK_ELEMENTS
            return Array(self.data != value)
        def all(self):
            reductions.append(self.size)
            assert self.size <= cube_check.CHECK_ELEMENTS
            return Array(np.asarray(self.data.all()))
        def item(self):
            assert self.shape == ()
            host_flags.append(bool(self.data.item()))
            return self.data.item()
        def get(self, *args, **kwargs): raise AssertionError('No payload get')
        def __array__(self, *args, **kwargs): raise AssertionError('No implicit host array')
    def isfinite(array):
        assert isinstance(array, Array) and array.size <= cube_check.CHECK_ELEMENTS
        return Array(np.isfinite(array.data))
    def any_(array):
        assert isinstance(array, Array) and array.size <= cube_check.CHECK_ELEMENTS
        reductions.append(array.size)
        return Array(np.asarray(array.data.any()))
    def asnumpy(*args, **kwargs): raise AssertionError('No payload asnumpy')
    cp = SimpleNamespace(isfinite=isfinite, any=any_, asnumpy=asnumpy)
    sums, counts = fixture()
    result = cube_check.array_checks(cp, Array(sums), counts, shape=sums.shape)
    assert slices and max(slices) <= cube_check.CHECK_ELEMENTS
    assert sum(slices) == sums.size
    assert len(reductions) == len(host_flags) == 2*len(slices)
    assert result == [dict(bin=i, finite=True, nonzero=i==0) for i in range(100)]
