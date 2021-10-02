"""Data access VERSIONS
   for composite detectors made of epix10kt segments/panels.
"""
import numpy as np
from amitypes import Array2d
#from psana.detector.detector_impl import DetectorImpl
import psana.detector.epix_base as eb
import logging
logger = logging.getLogger(__name__)

class epixhr2x2_raw_2_0_1(eb.epix_base):
    def __init__(self, *args, **kwargs):
        logger.debug('epixhr2x2_raw_2_0_1.__init__')
        eb.epix_base.__init__(self, *args, **kwargs)
        self.seg_geo = eb.sgs.Create(segname='EPIXHR2X2:V1')
        self._data_bit_mask = eb.M15 # for epixhr2x2 data

    def _array(self, evt) -> Array2d:
        f = None
        segs = self._segments(evt)
        if segs is None:
            pass
        else:
            nx = segs[0].raw.shape[1]
            ny = segs[0].raw.shape[0]
            f = segs[0].raw & 0x7fff
        return f


    def _cbits_config_segment(self, cob):
        """cob=det.raw._seg_configs()[<seg-ind>].config"""
        return eb.cbits_config_epixhr2x2(cob, shape=(288, 384))


    def _cbits_config_and_data_detector(self, evt=None):
        return eb.cbits_config_and_data_detector_epixhr2x2(self, evt)

# EOF
