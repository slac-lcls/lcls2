import h5py
import numpy as np
import sys
from .vals import testvals

def h5():
    f = h5py.File('data.h5')
    nevent = 2
    for ievt in range(nevent):
        for key in testvals:
            val = testvals[key]
            if type(val) is np.ndarray:
                assert np.array_equal(val,f[key][ievt])
            else:
                assert val == f[key][ievt]
    print('h5 tested',len(testvals),'values')
