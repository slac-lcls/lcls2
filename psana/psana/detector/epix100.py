
import numpy as np
import psana.detector.areadetector as ad

import os
import sys
logger = ad.logging.getLogger(__name__)
import psana.detector.UtilsEpix100 as ue100


class epix100hw_raw_2_0_1(ad.AreaDetector):

    def __init__(self, *args, **kwa):
        ad.AreaDetector.__init__(self, *args, **kwa)
        self._seg_geo = ad.sgs.Create(segname='EPIX100:V1')

    def image(self,evt):
        """substitution for real image."""
        segments = self._segments(evt)
        return segments[0].raw


class epix100_raw_2_0_1(ad.AreaDetector):

    def __init__(self, *args, **kwa):
        ad.AreaDetector.__init__(self, *args, **kwa)
        self._seg_geo = ad.sgs.Create(segname='EPIX100:V1')
        self._data_bit_mask = 0xffff


    def _gain(self):
        """Returns gain in ADU/eV
           1. returns cached gain (self._gain_) if not None
           2. check if gain is available in calib constants and return it if not None
           3. set default gain factor shaped as pedestals
        """
        if self._gain_ is not None: return self._gain_
        g = ad.AreaDetector._gain(self)
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


    def _det_geotxt_default(self):
        """returns (str) default geometry constants from lcls2/psana/psana/pscalib/geometry/data/geometry-def-*.data
        """
        dir_detector = os.path.abspath(os.path.dirname(__file__))
        fname = '%s/../pscalib/geometry/data/geometry-def-epix100a.data' % dir_detector
        logger.debug('_det_geotxt_default from file: %s' % fname)
        return ad.ut.load_textfile(fname)


    def calib(self, evt, cmpars=None, **kwa): #cmpars=(0,7,100,10)):
        return ue100.calib_epix100(self, evt, cmpars=cmpars, **kwa)
        #return ue100.common_mode_increment(self, evt, cmpars=cmpars, **kwa)

# EOF
