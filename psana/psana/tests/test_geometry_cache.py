"""Geometry access after startup must never require another rank to call it."""
import numpy as np
import pytest

from psana.detector.shared_geo_cache import SharedGeoCache
from geometry_cache_fixture import GeometryDetector, assert_expected


class Memory:
    is_leader = True

    def __init__(self):
        self.shm_comm = self
        self.arrays = {}
        self.allow_collectives = False

    def bcast(self, value, root=0):
        assert self.allow_collectives, 'unexpected event-loop broadcast'
        return value

    def Barrier(self):
        assert self.allow_collectives, 'unexpected event-loop barrier'

    def has_array(self, name):
        return name in self.arrays

    def get_array(self, name):
        return self.arrays[name]

    def allocate_array(self, name, shape, dtype, **kwargs):
        assert self.allow_collectives, 'unexpected event-loop shared allocation'
        self.arrays[name] = np.empty(shape, dtype)
        return self.arrays[name]


VARIANTS = [
    ('_pixel_coords', {}),
    ('_pixel_coords', {'all_segs': True}),
    ('_pixel_coords', {'cframe': 1}),
    ('_pixel_coords', {'do_tilt': False}),
    ('_pixel_coord_indexes', {}),
    ('_pixel_coord_indexes', {'all_segs': True}),
    ('_pixel_coord_indexes', {'cframe': 1}),
    ('_pixel_coord_indexes', {'do_tilt': False}),
    ('_pixel_coord_indexes', {'xy0_off_pix': (100, 200)}),
    ('_pixel_coord_indexes', {'pix_scale_size_um': 20}),
]


@pytest.mark.parametrize('method,kwargs', VARIANTS)
def test_unseeded_variant_is_local_cached_and_numerically_exact(method, kwargs):
    memory = Memory()
    detector = GeometryDetector()
    detector._shared_geo_cache = SharedGeoCache(memory)
    result = getattr(detector, method)(**kwargs)
    calls = detector.geometry.calls.copy()
    again = getattr(detector, method)(**kwargs)
    assert all(a is b for a, b in zip(result, again))
    assert detector.geometry.calls == calls
    assert not memory.arrays
    assert_expected(method, result, **kwargs)


@pytest.mark.parametrize('method', ['_pixel_coords', '_pixel_coord_indexes'])
def test_startup_shared_hit_and_later_local_miss(method):
    memory = Memory()
    memory.allow_collectives = True
    detector = GeometryDetector()
    detector._shared_geo_cache = SharedGeoCache(memory)
    seeded = getattr(detector, method)(_initialize_shared=True)
    names = set(memory.arrays)
    assert len(names) == len(seeded)
    assert all(any(a is shared for shared in memory.arrays.values()) for a in seeded)
    memory.allow_collectives = False
    calls = detector.geometry.calls.copy()
    hit = getattr(detector, method)()
    assert detector.geometry.calls == calls
    assert all(a is b for a, b in zip(seeded, hit))
    fallback = getattr(detector, method)(all_segs=True)
    assert_expected(method, fallback, all_segs=True)
    assert set(memory.arrays) == names
    assert all(not np.shares_memory(a, b) for a in fallback for b in seeded)


def test_local_variants_do_not_replace_each_other_or_cross_detectors():
    cache = SharedGeoCache(Memory())
    detector = GeometryDetector()
    detector._shared_geo_cache = cache
    default = detector._pixel_coord_indexes()
    full = detector._pixel_coord_indexes(all_segs=True)
    shifted = detector._pixel_coord_indexes(xy0_off_pix=(100, 200))
    assert not np.array_equal(default[0], shifted[0])
    assert detector._pixel_coord_indexes()[0] is default[0]
    assert detector._pixel_coord_indexes(all_segs=True)[0] is full[0]
    other = GeometryDetector()
    other._det_name = 'other'
    other._shared_geo_cache = cache
    assert other._pixel_coord_indexes()[0] is not default[0]
