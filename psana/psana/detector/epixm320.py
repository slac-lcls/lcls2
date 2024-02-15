import os
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
        self._seg_geo = eb.sgs.Create(segname='EPIXM320:V1')
        self._data_bit_mask = eb.M14 # for epixhr2x2 data on 2023-10-30 Dionisio - HR has 14 data bits.
        self._data_gain_bit = eb.B15
        self._gain_bit_shift = 10
        self._gains_def = (41.0, 13.7, 0.512) # epixhr2x2 ADU/keV H:M:L = 1 : 1/3 : 1/80
        self._path_geo_default = 'pscalib/geometry/data/geometry-def-epixm320.data'
        self._dataDebug = None

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

    def raw(self, evt) -> Array3d:
        if evt is None: return None
        segs = self._segments(evt)    # dict = {seg_index: seg_obj}
        if segs is None: return None

        return np.stack([segs[0].raw])

    def image(self, evt, **kwargs) -> Array2d:
        if evt is None: return None
        segs = self._segments(evt)    # dict = {seg_index: seg_obj}
        if segs is None: return None

        return self.raw(evt)[0]
