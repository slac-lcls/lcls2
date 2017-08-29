import h5py
import numpy as np
from testvals import testvals
f = h5py.File('data.h5')
for key in testvals:
    val = testvals[key]
    if type(val) is np.ndarray:
        assert np.array_equal(val,f[key][0])
    else:
        assert val==f[key][0]
