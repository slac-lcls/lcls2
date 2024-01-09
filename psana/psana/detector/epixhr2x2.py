"""Data access VERSIONS
   for composite detectors made of epix10kt segments/panels.

   2023-10-30 Dionisio: I realized that I wired both MSB (bits 15 and bits 14) to report gain information.
"""

import os
import numpy as np
from amitypes import Array2d
import psana.detector.epix_base as eb
import logging
from psana.detector.detector_impl import DetectorImpl
logger = logging.getLogger(__name__)

# make an empty detector interface for Matt's hardware
# configuration object so that config_dump works - cpo
class epixhr2x2hw_config_2_0_0(DetectorImpl):
    def __init__(self, *args, **kwargs):
        super(epixhr2x2hw_config_2_0_0, self).__init__(*args)

class epixhr2x2_raw_2_0_1(eb.epix_base):
    def __init__(self, *args, **kwargs):
        logger.debug('epixhr2x2_raw_2_0_1.__init__')
        eb.epix_base.__init__(self, *args, **kwargs)
        self._seg_geo = eb.sgs.Create(segname='EPIXHR2X2:V1')
        self._data_bit_mask = eb.M14 # for epixhr2x2 data on 2023-10-30 Dionisio - HR has 14 data bits.
        self._data_gain_bit = eb.B15
        self._gain_bit_shift = 10
        self._gains_def = (41.0, 13.7, 0.512) # epixhr2x2 ADU/keV H:M:L = 1 : 1/3 : 1/80
        self._path_geo_default = 'pscalib/geometry/data/geometry-def-epixhr2x2.data'


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
        """cob=det.raw._seg_configs()[<seg-ind>].config"""
        logger.debug('epixhr2x2._cbits_config_segment')
        return eb.cbits_config_epixhr2x2(cob, shape=(288, 384))

# EOF
