import sys
sys.path.append('../../build/psana')
sys.path.append('../Detector')
sys.path.append('../')
from DataSource import DataSource
from Detector import Detector
import numpy as np

ds = DataSource('data.xtc')
det = Detector('xppcspad', ds.configs[0])

for evt in ds:
    raw = det.raw(evt.__next__())
    break

print('Raw values and shape:' )
print(raw, raw.shape)
assert(np.sum(raw)==9*17)
assert(raw.shape==(2,3,3))
