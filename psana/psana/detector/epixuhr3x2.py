import logging

import numpy as np
from amitypes import Array2d, Array3d

import psana.detector.epix_base as eb
from psana.detector.detector_impl import DetectorImpl
import psana.detector.UtilsEpixUHR as ueu
#ndu = ueu.ndu
#cond_msg = eb.ue.cond_msg

logger: logging.Logger = logging.getLogger(__name__)

class epixuhr3x2hw_config_0_1_0(DetectorImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class epixuhr3x2_config_0_1_0(DetectorImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class epixuhr3x2_raw_0_1_0(eb.epix_base):
    def __init__(self, *args, **kwargs):
        eb.epix_base.__init__(self, *args, **kwargs)
        self._gain_modes = ueu.GAIN_MODES # ('FHG', 'FMG', 'FLG1', 'FLG2', 'AHLG1', 'AHLG2', 'AMLG1', 'AMLG2')
        self._gain_states = ueu.GAIN_STATES # GAIN_MODES + ('AHLG1_L', 'AHLG2_L', 'AMLG1_L', 'AMLG2_L') # 12 total
        self._store_ = None
        self._counter_image = 0
        self._seg_geo = eb.sgs.Create(segname='EPIXUHR3X2:V1')
        self._path_geo_default = 'pscalib/geometry/data/geometry-def-epixuhr3x2-02.data'
        self._data_gain_bitnum = 1 # LSB (right-most) is gain bit
        self._data_bit_mask = 0x0FFE # 11-bit data mask (bits 2-12)

    def _cbits_config_segment(self, cob):
        """cob=det.raw._seg_configs()[<seg-ind>].config - segment configuration object, where self=det.raw
           returns segment gain control bits # shape=(336, 576)
        """
        return ueu.cbits_config_segment(cob)

    def raw(self, evt, sh_seg=(336,576)) -> Array3d:
        return ueu.raw_v01(self, evt, sh_seg=sh_seg)

    def calib(self, evt, **kwa) -> Array3d:
        """overrides lcls2/psana/psana/detector/epix_base.py epix_base.calib"""
        return ueu.calib_v02(self, evt, **kwa)

#    def image(self, evt, **kwa) -> Array2d: # see in areadetector.py
#        """temporary re-implement AreaDetector.image
#           NOW HIDDEN: returns raw[0,:] 2-d temporary image for a single panel raw data (1, 336, 576)"""
#        return ueu.image_v01(self, evt, **kwa)

# EOF
