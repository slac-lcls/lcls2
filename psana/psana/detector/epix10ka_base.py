"""Data access BASIC METHODS for
   composite detectors made of epix10ka panels.
"""

#from psana.pscalib.geometry.SegGeometry import *
from psana.detector.areadetector import AreaDetector
from psana.detector.UtilsEpix10ka import calib_epix10ka_any

from amitypes import Array3d

import logging
logger = logging.getLogger(__name__)

#----

class epix10ka_base(AreaDetector):

    def __init__(self, *args, **kwargs):
        logger.debug('epix10ka_base.__init__') # self.__class__.__name__
        AreaDetector.__init__(self, *args, **kwargs)


    def calib(self,evt) -> Array3d:
        """
        Create calibrated data array.
        """
        logger.info('epix10ka_base.calib')
        #return self.raw(evt)
        return calib_epix10ka_any(self, evt)


    # example of some possible common behavior
    #def _common_mode(self, **kwargs):
    #    pass

    #def raw(self,evt):
    #    data = {}
    #    segments = self._segments(evt)
    #    if segments is None: return None
    #    for segment,val in segments.items():
    #        data[segment]=val.raw
    #    return data

#----

if __name__ == "__main__":
    import sys
    sys.exit('See example in test_%s if available...' % sys.argv[0].split('/')[-1])

#----
