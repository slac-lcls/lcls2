"""Small real geometry shared by local and MPI cache regression tests."""
import numpy as np

from psana.detector.areadetector import AreaDetector
from psana.pscalib.geometry.GeometryAccess import GeometryAccess


GEOMETRY = """
CAMERA 0 MTRX:V2:3:4:10:10 0 0 30 0 0 0 0 1 2 3
CAMERA 0 MTRX:V2:3:4:10:10 1 0 -30 0 0 0 0 1 2 3
IP 0 CAMERA 0 0 0 1000 90 0 0 0 0 0
"""


class CountedGeometry(GeometryAccess):
    def __init__(self):
        super().__init__()
        self.load_pars_from_str(GEOMETRY)
        self.calls = [0, 0]

    def get_pixel_coords(self, *args, **kwargs):
        self.calls[0] += 1
        return super().get_pixel_coords(*args, **kwargs)

    def get_pixel_coord_indexes(self, *args, **kwargs):
        self.calls[1] += 1
        return super().get_pixel_coord_indexes(*args, **kwargs)


class GeometryDetector(AreaDetector):
    def __new__(cls, *args):
        return object.__new__(cls)

    def __init__(self, *args):
        # Exercise the real AreaDetector accessors without XTC or a DB fetch.
        self._det_name = 'camera'
        self._drp_class_name = 'raw'
        self._segment_numbers = [1]
        self._path_geo_default = None
        self._calibconst = {'geometry': (GEOMETRY, {})}
        self.geometry = CountedGeometry()

    def _det_geo(self):
        return self.geometry

    def _calibconstants(self):
        return None


def assert_expected(method, arrays, **kwargs):
    reference = GeometryDetector()  # No shared cache: independent local path.
    expected = getattr(reference, method)(**kwargs)
    for actual, wanted in zip(arrays, expected):
        np.testing.assert_array_equal(actual, wanted)
        assert actual.dtype == wanted.dtype
    assert arrays[0].shape == ((2, 3, 4) if kwargs.get('all_segs') else (1, 3, 4))
