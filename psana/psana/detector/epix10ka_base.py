"""Data access BASIC METHODS for
   composite detectors made of epix10ka panels.
"""

#from psana.pscalib.geometry.SegGeometry import *
from psana.detector.areadetector import AreaDetector, np
from psana.detector.UtilsEpix10ka import calib_epix10ka_any

from amitypes import Array3d

import logging
logger = logging.getLogger(__name__)

from psana.detector.Utils import merge_status

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


    def _mask_from_status(self, **kwa) -> Array3d:
        """
        Parameters **kwa
        ----------------
        ##mode - int 0/1/2 masks zero/four/eight neighbors around each bad pixel
        'indexes', (0,1,2,3,4)) # indexes stand for gain ranges 'FH','FM','FL','AHL-H','AML-M'
        Returns 
        -------
        mask made of status: np.array, ndim=3, shape: as full detector data
        """

        status = self._status() # pixel_status from calibration constants
        statmrg = merge_status(status, **kwa) # grinds=(0,1,2,3,4), dtype=np.uint32
        return np.asarray(np.select((statmrg>0,), (0,), default=1), dtype=np.uint8)
        #logger.info(info_ndarr(status, 'status '))
        #return statmrg

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

# EOF
