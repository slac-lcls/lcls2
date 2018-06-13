import sys
from psana import DataSource
import numpy as np

def det():
    ds = DataSource('data.xtc')
    det = ds.Detector('xppcspad')

'''
    for evt in ds.events():
        raw = det.raw(evt.__next__())
        break

    print('Raw values and shape:' )
    print(raw, raw.shape)
    assert(np.sum(raw)==9*17)
    assert(raw.shape==(2,3,3))
    assert(ds._configs()[0].software.xppcspad.dettype == 'cspad')
    assert(ds._configs()[0].software.xppcspad.detid == 'detnum1234')
'''

if __name__ == '__main__':
    det()
