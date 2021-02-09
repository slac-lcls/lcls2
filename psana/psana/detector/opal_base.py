"""
Data access BASIC METHODS for opal detector
===========================================

Usage::

  from psana.detector.opal_base import opal_base

  o = opal_base(*args, **kwargs) # enherits from AreaDetector
  a = o.raw(evt)
  a = o.calib(evt)

2021-02-09 created by Mikhail Dubrovin
"""

from amitypes import Array3d

import logging
logger = logging.getLogger(__name__)

from psana.detector.areadetector import AreaDetector #, np, DTYPE_MASK, DTYPE_STATUS
from psana.detector.UtilsAreaDetector import arr3d_from_dict #, dict_from_arr3d,...
#from psana.pyalgos.generic.NDArrUtils import info_ndarr
#from psana.detector.UtilsMask import merge_status
#from psana.pscalib.geometry.SegGeometryEpix10kaV1 import epix10ka_one as seg

#----

class opal_base(AreaDetector):

    def __init__(self, *args, **kwa):
        logger.debug('opal_base.__init__') # self.__class__.__name__
        AreaDetector.__init__(self, *args, **kwa)


    def calib(self, evt, **kwa) -> Array3d:
        """
        Returns calibrated data array.
        """
        logger.debug('opal_base.calib')
        return self.raw(evt)
        #return calib_opal_any(self, evt, **kwa)


    def raw(self,evt) -> Array3d:
        segs = self._segments(evt)
        if segs is None: return None
        return arr3d_from_dict({k:v.image for k,v in segs.items()})

#----

if __name__ == "__main__":
    import sys
    sys.exit('See example in test_%s if available...' % sys.argv[0].split('/')[-1])

# EOF
