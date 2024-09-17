import os
import sys
#from time import time
#from psana.detector.NDArrUtils import info_ndarr
import numpy as np
from amitypes import Array2d, Array3d
import psana.detector.epix_base as eb
import logging
from psana.detector.detector_impl import DetectorImpl
logger = logging.getLogger(__name__)

is_none = eb.ut.is_none
M15 = 0o77777 # 15-bit mask
B16 = 0o100000 # the 16-th bit (counting from 1)

# make an empty detector interface for Matt's hardware
# configuration object so that config_dump works - cpo
class epixm320hw_config_0_0_0(DetectorImpl):
    def __init__(self, *args, **kwargs):
        super(epixm320hw_config_0_0_0, self).__init__(*args)

class epixm320hw_config_0_1_0(epixm320hw_config_0_0_0):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

class epixm320_raw_0_0_0(eb.epix_base):
    def __init__(self, *args, **kwargs):
        logger.debug('epixm320_raw_0_0_0.__init__')
        eb.epix_base.__init__(self, *args, **kwargs)
        self._seg_geo = eb.sgs.Create(segname='EPIXMASIC:V1')
        self._data_bit_mask = M15 # for epixm320 data on 2024-03-20 Dawood - epixM has 15 data bits.
        self._data_gain_bit = B16 # gain switching bit
        self._gain_bit_shift = 10
        self._gains_def = (-100.7, -21.3, -100.7) # ADU/Pulser
        self._gain_modes = ('SH', 'SL', 'AHL')
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
        #print('TBD: %s.%s' % (self.__class__.__name__, sys._getframe().f_code.co_name))
        if evt is None: return None

        #t0_sec = time()
        raw = self.raw(evt)
        if is_none(raw, 'self.raw(evt) is None - return None'):
            return raw

        # Subtract pedestals
        peds = self._pedestals()
        if is_none(peds, 'det.raw._pedestals() is None - return det.raw.raw(evt)'):
            return raw
        #print(info_ndarr(peds,'XXX peds', first=1000, last=1005))

        gr1 = (raw & self._data_gain_bit) > 0

        #print(info_ndarr(gr1,'XXX gr1', first=1000, last=1005))
        pedgr = np.select((gr1,), (peds[1,:],), default=peds[0,:])
        arrf = np.array(raw & self._data_bit_mask, dtype=np.float32)
        arrf -= pedgr

        #print('XXX time for calib: %.6f sec' % (time()-t0_sec)) # 4ms on drp-neh-cmp001

        return arrf


#    def image(self, evt, **kwargs) -> Array2d: # see in areadetector.py
#        if evt is None: return None
#        return self.raw(evt)[0].reshape(768,384)

# EOF
