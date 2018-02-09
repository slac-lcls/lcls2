import sys
from psana import DataSource
from psana.detector import Detector
import numpy as np

def det():
    ds = DataSource('data.xtc')
    det = Detector('xppcspad', ds.configs[0].software)

    for evt in ds:
        raw = det.raw(evt.__next__())
        break

    print('Raw values and shape:' )
    print(raw, raw.shape)
    assert(np.sum(raw)==9*17)
    assert(raw.shape==(2,3,3))
