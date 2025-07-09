
import numpy as np
import psana2.detector.areadetector as ad

import os
import sys
logger = ad.logging.getLogger(__name__)
import psana2.detector.UtilsEpix100 as ue100
from psana2.pyalgos.generic.NDArrUtils import divide_protected


class epix100hw_raw_2_0_1(ad.AreaDetectorRaw):

    def __init__(self, *args, **kwa):
        ad.AreaDetectorRaw.__init__(self, *args, **kwa)
        self._seg_geo = ad.sgs.Create(segname='EPIX100:V1')
        self._path_geo_default = 'pscalib/geometry/data/geometry-def-epix100a.data'

    def image(self,evt):
        """substitution for real image."""
        segments = self._segments(evt)
        return segments[0].raw


class epix100_raw_2_0_1(ad.AreaDetectorRaw):

    def __init__(self, *args, **kwa):
        ad.AreaDetectorRaw.__init__(self, *args, **kwa)
        self._seg_geo = ad.sgs.Create(segname='EPIX100:V1')
        self._data_bit_mask = 0xffff
        self._gain_ = None # ADU/eV
        self._gain_factor_ = None # keV/ADU
        self._path_geo_default = 'pscalib/geometry/data/geometry-def-epix100a.data'

    def _gain(self):
        """Returns gain in ADU/eV
           1. returns cached gain (self._gain_) if not None
           2. check if gain is available in calib constants and return it if not None
           3. set default gain factor shaped as pedestals
        """
        if self._gain_ is not None: return self._gain_
        g = ad.AreaDetectorRaw._gain(self)
        if g is not None:
            self._gain_factor_ = divide_protected(np.ones_like(g), g)
            return g
        peds = self._pedestals() # - n-d pedestals
        if peds is None: return None
        self._gain_ = ue100.GAIN_DEFAULT * np.ones_like(peds)
        self._gain_factor_ = ue100.GAIN_FACTOR_DEFAULT * np.ones_like(peds)
        return self._gain_

    def _gain_factor(self):
        if self._gain_factor_ is None: _ = self._gain()
        return self._gain_factor_

    def _common_mode_increment(self, evt, cmpars=(0,7,100,10), **kwa):
        return ue100.common_mode_increment(self, evt, cmpars=cmpars, **kwa)

    def calib(self, evt, cmpars=None, **kwa): #cmpars=(0,7,100,10)):
        return ue100.calib_epix100(self, evt, cmpars=cmpars, **kwa)
        #return ue100.common_mode_increment(self, evt, cmpars=cmpars, **kwa)

# EOF
