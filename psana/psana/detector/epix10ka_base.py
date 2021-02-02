"""Data access BASIC METHODS for
   composite detectors made of epix10ka panels.
"""

#from psana.pscalib.geometry.SegGeometry import *
from psana.detector.areadetector import AreaDetector, np, DTYPE_MASK, DTYPE_STATUS
from psana.detector.UtilsEpix10ka import calib_epix10ka_any

from amitypes import Array3d

import logging
logger = logging.getLogger(__name__)

from psana.detector.UtilsMask import merge_status
from psana.pscalib.geometry.SegGeometryEpix10kaV1 import epix10ka_one as seg

from psana.pyalgos.generic.NDArrUtils import info_ndarr
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


    def _mask_from_status(self, **kwa):
        """
        Parameters **kwa
        ----------------
        'grinds', (0,1,2,3,4)) # gain range indexes for 'FH','FM','FL','AHL-H','AML-M'
        Returns 
        -------
        mask made of status: np.array, ndim=3, shape: as full detector data
        """
        _grinds = kwa.get('grinds',(0,1,2,3,4))
        status = self._status() # pixel_status from calibration constants
        statmrg = merge_status(status, grinds=_grinds, dtype=DTYPE_STATUS) # dtype=np.uint64
        return np.asarray(np.select((statmrg>0,), (0,), default=1), dtype=DTYPE_MASK)
        #logger.info(info_ndarr(status, 'status '))
        #return statmrg


    def _mask_edges(self, edge_rows=1, edge_cols=1, center_rows=0, center_cols=0, dtype=DTYPE_MASK, **kwa):
        mask1 = seg.pixel_mask_array(edge_rows, edge_cols, center_rows, center_cols, dtype)
        nsegs = self._number_of_segments_total()
        if nsegs is None:
            logger.warning('_number_of_segments_total is None')
            return None
        logger.info('_mask_edges for %d-segment epix10ka'%nsegs)
        mask = np.stack([mask1 for i in range(nsegs)])
        logger.info(info_ndarr(mask, '_mask_edges '))
        return mask


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
