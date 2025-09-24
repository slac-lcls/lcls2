
import numpy as np
import psana.detector.areadetector as ad

import os
import sys
logger = ad.logging.getLogger(__name__)
import psana.detector.UtilsEpix100 as ue100
import psana.detector.NDArrUtils as ndau

is_none, is_true = ad.is_none, ad.is_true
#divide_protected = ndau.divide_protected


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
        """Returns gain in ADU/keV
           1. returns cached gain (self._gain_) if not None
           2. check if gain is available in calib constants and return it if not None
           3. set default gain factor shaped as pedestals
        """
        if self._gain_ is not None: return self._gain_
        g = ad.AreaDetectorRaw._gain(self)
        if g is not None:
            self._gain_factor_ = ndau.divide_protected(np.ones_like(g), g)
            metag = self._calibconst['pixel_gain'][1]
            s = 'pixel_gain constants from exp:%s run:%d' % (metag['experiment'], metag['run'])
            s += ndau.info_ndarr(self._gain_factor_, '\n    gain factors set from calib constants', last=5)
            s += '\n    %s\n' % self._info_calibconst()
            self._logmet_init(s) #logger.info(s)
            return g
        logger.warning('gain is missing in calib constants, try to set default')
        peds = self._pedestals() # - n-d pedestals
        if is_none(peds, 'pedestals are None, _gain_ IS NOT SET', logger_method=logger.warning):
            return None
        self._gain_ = ue100.GAIN_DEFAULT * np.ones_like(peds)
        self._gain_factor_ = ue100.GAIN_FACTOR_DEFAULT * np.ones_like(peds)
        logger.warning(ndau.info_ndarr(self._gain_factor_, 'USE DEFAULT GAIN FACTORS', last=5))
        return self._gain_

    def _gain_factor(self):
        """Returns gain factors in keV/ADU"""
        if self._gain_factor_ is None: _ = self._gain()
        return self._gain_factor_

    def _common_mode_increment(self, evt, cmpars=(0,7,100,10), **kwa):
        return ue100.common_mode_increment(self, evt, cmpars=cmpars, **kwa)

    def calib(self, evt, cmpars=None, **kwa): #cmpars=(0,7,100,10)):
        return ue100.calib_epix100(self, evt, cmpars=cmpars, **kwa)

# EOF
