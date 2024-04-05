import os
import sys
import numpy as np
from amitypes import Array2d, Array3d
import psana.detector.epix_base as eb
import logging
from psana.detector.detector_impl import DetectorImpl
logger = logging.getLogger(__name__)


# make an empty detector interface for Matt's hardware
# configuration object so that config_dump works - cpo
class epixm320hw_config_0_0_0(DetectorImpl):
    def __init__(self, *args, **kwargs):
        super(epixm320hw_config_0_0_0, self).__init__(*args)

class epixm320_raw_0_0_0(eb.epix_base):
    def __init__(self, *args, **kwargs):
        logger.debug('epixm320_raw_0_0_0.__init__')
        eb.epix_base.__init__(self, *args, **kwargs)
        self._seg_geo = eb.sgs.Create(segname='EPIXMASIC:V1')
        self._data_bit_mask = eb.M14 # for epixm320 data on 2024-03-20 Dawood - M has 14 data bits.
        self._data_gain_bit = eb.B15
        self._gain_bit_shift = 10
        self._gains_def = (41.0, 13.7, 0.512) # Revisit: epixhr2x2 ADU/keV H:M:L = 1 : 1/3 : 1/80
        self._path_geo_default = 'pscalib/geometry/data/geometry-def-epixm320.data'
        self._dataDebug = None
        self._segment_numbers = [0,1,2,3]

    def _array(self, evt) -> Array2d:
        f = None
        segs = self._segments(evt)
        if segs is None:
            pass
        else:
            nx = segs[0].raw.shape[1]
            ny = segs[0].raw.shape[0]
            f = segs[0].raw & self._data_bit_mask # 0x7fff
        return f

    def _cbits_config_segment(self, cob):
        """not used in epixm"""
        return None


    def _segment_ids(self):
        """Re-impliment epix_base._segment_ids for epixm320
        returns list of detector segment ids using ASIC numbers, e.g.
        [00016778240-0176075265-0452984854-4021594881-1962934296-0177446913-0402653206-ASIC-00,
         00016778240-0176075265-0452984854-4021594881-1962934296-0177446913-0402653206-ASIC-01,
         00016778240-0176075265-0452984854-4021594881-1962934296-0177446913-0402653206-ASIC-02,
         00016778240-0176075265-0452984854-4021594881-1962934296-0177446913-0402653206-ASIC-03]
         for det.raw._uniqueid: epixm320_0016778240-0176075265-0452984854-4021594881-1962934296-0177446913-0402653206
         and self._segment_numbers = [0, 1, 2, 3]
        """
        id = self._uniqueid.split('_')[1] # 0016778240-0176075265-0452984854-4021594881-1962934296-0177446913-0402653206
        return ['%s-ASIC-%02d' % (id,i) for i in self._segment_numbers]


#    def raw(self, evt) -> Array3d: # see in areadetector.py
#        if evt is None: return None
#        segs = self._segments(evt)    # dict = {seg_index: seg_obj}
#        if segs is None: return None
#        return segs[0].raw # shape=(4, 192, 384)

    def calib(self, evt) -> Array3d: # already defined in epix_base and AreaDetectorRaw
        """ TBD - when pedestals are availavle..."""
        #logger.debug('%s.%s' % (self.__class__.__name__, sys._getframe().f_code.co_name))
        print('TBD: %s.%s' % (self.__class__.__name__, sys._getframe().f_code.co_name))
        if evt is None: return None
        return self.raw(evt).astype(np.float32)

#    def image(self, evt, **kwargs) -> Array2d: # see in areadetector.py
#        if evt is None: return None
#        return self.raw(evt)[0].reshape(768,384)
