"""
Utilities for area detector.

from psana.detector.UtilsAreaDetector import dict_from_arr3d, arr3d_from_dict

2020-11-06 created by Mikhail Dubrovin
"""

#from psana.pyalgos.generic.NDArrUtils import shape_as_2d, shape_as_3d, reshape_to_2d, reshape_to_3d

import numpy as np

def dict_from_arr3d(a):
    """Converts 3d array of shape=(n0,n1,n2) to dict {k[0:n0-1] : nd.array.shape(n1,n2)}
       Consumes 25us for epix10ka2m array shape:(16, 352, 384) size:2162688 dtype:float32
    """
    assert isinstance(a, np.ndarray)
    assert a.ndim == 3
    return {k:a[k,:,:] for k in range(a.shape[0])}


def arr3d_from_dict(d, keys=None):
    """Converts dict {k[0:n0-1] : nd.array.shape(n1,n2)} to 3d array of shape=(n0,n1,n2)
       Consumes 7ms for epix10ka2m array shape:(16, 352, 384) size:2162688 dtype:float32
    """
    assert isinstance(d, dict)
    _keys = sorted(d.keys() if keys is None else keys)
    return np.stack([d[k] for k in _keys])


def mean_of_median_for_dictarr(d, keys=None):
    assert isinstance(d, dict)
    _keys = sorted(d.keys() if keys is None else keys)
    arr_of_median = np.array([np.median(v) for v in d.values()])
    return arr_of_median.mean()

#----
