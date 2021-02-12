"""
Data access BASIC METHODS for opal detector
===========================================

Usage::

  from psana.detector.opal_base import opal_base

  o = opal_base(*args, **kwa) # inherits from AreaDetector
  a = o.raw(evt)
  a = o.calib(evt, dtype=np.float32, **kwa)
  img = o.image(self, evt, **kwa)

2021-02-09 created by Mikhail Dubrovin
"""

from amitypes import Array2d, Array3d

import logging
logger = logging.getLogger(__name__)

from psana.detector.areadetector import AreaDetector, np # DTYPE_MASK, DTYPE_STATUS
from psana.detector.UtilsAreaDetector import arr3d_from_dict #, dict_from_arr3d,...
from psana.pyalgos.generic.NDArrUtils import reshape_to_2d, divide_protected, info_ndarr
#----

class opal_base(AreaDetector):

    def __init__(self, *args, **kwa):
        logger.debug('opal_base.__init__') # self.__class__.__name__
        AreaDetector.__init__(self, *args, **kwa)


    def raw(self,evt) -> Array3d:
        segs = self._segments(evt)
        if segs is None: return None
        return arr3d_from_dict({k:v.image for k,v in segs.items()}) if len(segs.items())>1 else\
               next(iter(segs.values())).image


    def calib(self, evt, **kwa) -> Array3d:
        """
        Returns calibrated data array.
        """
        logger.debug('opal_base.calib')

        dtype = kwa.get('dtype', np.float32)
        raw = self.raw(evt)
        if raw is None:
            logger.debug('det.raw.raw(evt) is None')
            return None

        arr = raw.astype(dtype, copy=True)

        peds = self._pedestals()
        if peds is None:
            logger.debug('det.raw._pedestals() is None - return raw')
            return arr

        arr -= peds
        gain = self._gain()
        if gain is None:
            logger.debug('det.raw._gain() is None - return raw-peds')
            return arr

        return divide_protected(arr, gain)


    def image(self, evt, **kwa) -> Array2d:
        print('YYY opal_base.image')
        #logger.debug('opal_base.image')
        arr = self.calib(self, evt, **kwa)
        print(info_ndarr(arr, 'XXX arr:'))

        return arr if arr.ndim==2 else reshape_to_2d(arr)

#----

if __name__ == "__main__":
    import sys
    sys.exit('See example in test_%s if available...' % sys.argv[0].split('/')[-1])

# EOF
