import h5py
import numpy as np
from testvals import testvals
f = h5py.File('data.h5')
nevent = 2
for ievt in range(nevent):
    for key in testvals:
        val = testvals[key]
        if type(val) is np.ndarray:
            assert np.array_equal(val,f[key][ievt])
        else:
            assert val==f[key][ievt]
