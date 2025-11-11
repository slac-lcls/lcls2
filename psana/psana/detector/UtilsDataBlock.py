
"""
:py:class:`DataBlock`
=====================

Usage::

    from psana.detector.UtilsDataBlock import *
    #OR
    import psana.detector.UtilsDataBlock as udb

This software was developed for the LCLS project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2025-10-08 by Mikhail Dubrovin
"""

import logging
logger = logging.getLogger(__name__)
import os
import sys
import numpy as np
from psana.detector.NDArrUtils import info_ndarr #, divide_protected, reshape_to_2d, save_ndarray_in_textfile


class DataBlock():
    """data block accumulation on event w/o processing,
       block shape=(nrecs, <shape-of-passed-raw>)
    """
    def __init__(self, **kwa):
        self.kwa    = kwa
        self.nrecs  = kwa.get('nrecs', 1000)
        self.irec   = -1
        self.block  = None
        #self.dtype  = kwa.get('dtype', np.uint16)
        #self.fname_block = kwa.get('fname_block', None) # None - not save
        #self.datbits= kwa.get('datbits', 0xffff)

    def event(self, raw, evnum):
        """increment of det.raw.raw array in the block.
           Parameters:
           - raw (np.array) - det.raw.raw(evt) with optional [segind,:][aslice]
           - evnum (int) - event number
        """
        logger.debug('event %d' % evnum)

        if raw is None: return self.is_full()

        if self.block is None :
           self.block=np.zeros((self.nrecs,)+tuple(raw.shape), dtype=raw.dtype)
           self.evnums=np.zeros((self.nrecs,), dtype=np.uint16)
           logger.info(info_ndarr(self.block,'created empty data block') + '\n   '\
                      +info_ndarr(self.evnums,'and array for event numbers'))

        if self.not_full():
            self.irec +=1
            irec = self.irec
            self.block[irec,:] = raw
            self.evnums[irec] = evnum
            print('   add to block irec/nrecs: %d/%s' %(irec, self.nrecs), end=('\r' if irec>3 else '\n'))

        return self.is_full()

    def is_full(self):
        return not self.not_full()

    def not_full(self):
        return self.irec < self.nrecs-1

    def max_min(self):
        return np.max(self.block, axis=0),\
               np.min(self.block, axis=0)

    def save(self, fname=None):
        """save data-block array and other DataBlock attributes in file if fname is not None"""
        # if fname is None: fname = self.fname_block
        s = info_ndarr(self.block, 'data-block array', last=5)
        if fname is None:
            s += '\n    IS NOT SAVED, fname is None'
        else:
            np.savez(fname, block=self.block, evnums=self.evnums, intpars=np.array((self.nrecs, self.irec)))
            s += '\n    saved as %s' % fname
        logger.info(s)

    def load(self, fname=None):
        """restore data-block array and other DataBlock attributes from file if available"""
        logger.info('Load DataBlock attributes from file: %s' % fname)
        assert os.path.exists(fname)
        data = np.load(fname)
        self.block = data['block']   # data['arr_0']
        self.evnums = data['evnums'] # data['arr_1']
        self.nrecs, self.irec = tuple(data['intpars'])

    def info_data_block(self, cmt=''):
        return cmt\
             + info_ndarr(self.block, '  data block')\
             + info_ndarr(self.evnums, '\n  evnums')\
             + '\n  nrecs: %d irec: %d' % (self.nrecs, self.irec)

#    def extra_arrays(self, raw, evnum):
#        self.arr_max = np.zeros(shape_raw, dtype=dtype_raw)
#        self.arr_min = np.ones (shape_raw, dtype=dtype_raw) * self.datbits
#        np.maximum(self.arr_max, raw, out=self.arr_max)
#        np.minimum(self.arr_min, raw, out=self.arr_min)


if __name__ == "__main__":
    sys.exit('Used in UtilsPixelStatus.py etc.')

# EOF
