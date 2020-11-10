"""LCLS2 data format assumes that 2d panel data is saved in dict = {<panel-number>:<panel-data-2d>}.
   This module is intended to test consumed time for conversion from 3-d array to diict and back. 
"""

from time import time
import numpy as np
from psana.pyalgos.generic.NDArrUtils import info_ndarr, print_ndarr


def load_ndarray(fname='/reg/g/psdm/detector/data2_test/npy/nda-mfxp17218-r0505-lysozyme-max-10k.npy'):
  return np.load(fname)


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


t0_sec = time()
nda = load_ndarray()
dt_sec = time()-t0_sec
print_ndarr(nda, '\nnda load time = %.6f sec ' % dt_sec)


print('\nnp.ndarray attributes:\n', dir(nda), '\n')


t0_sec = time()
d = dict_from_arr3d(nda)
dt_sec = time()-t0_sec
print('dict_from_3darr time = %.6f sec ' % dt_sec)

for k,v in d.items(): print_ndarr(v, 'k:%02d v:' % k)

print('dict.keys:', d.keys())
print('sorted(dict.keys):', sorted(d.keys()))

t0_sec = time()
nda_new = arr3d_from_dict(d, keys=None)
dt_sec = time()-t0_sec
print_ndarr(nda_new, '\narr3d_from_dict time = %.6f sec ' % dt_sec)

t0_sec = time()
keys = keys=(3,5,7,11)
nda_new = arr3d_from_dict(d, keys)
dt_sec = time()-t0_sec
print('keys:', keys)
print_ndarr(nda_new, '\narr3d_from_dict time = %.6f sec ' % dt_sec)
