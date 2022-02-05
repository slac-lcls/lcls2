
import numpy as np
import psana.detector.areadetector as ad

import sys
from time import time
from psana.detector.NDArrUtils import info_ndarr
logger = ad.logging.getLogger(__name__)

from psana.detector.UtilsCommonMode import common_mode_cols,\
  common_mode_rows_hsplit_nbanks, common_mode_2d_hsplit_nbanks

GAIN_FACTOR_DEFAULT = 0.06 # keV/ADU on 2022/0/03 gain factor(Philip) = 60 eV/ADU, gain(Conny) = 16.4 ADU/keV
GAIN_DEFAULT = 1./GAIN_FACTOR_DEFAULT # ADU/keV

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
        self._gain_ = GAIN_DEFAULT * np.ones_like(peds)
        self._gain_factor_ = GAIN_FACTOR_DEFAULT * np.ones_like(peds)
        return self._gain_


    def _gain_factor(self):
        if self._gain_factor_ is None: _ = self._gain()
        return self._gain_factor_


    def calib(self, evt, cmpars=None, **kwa): #cmpars=(7,2,100)):
        logger.debug('In calib_epix10ka_any')

        t0_sec = time()

        det_raw = self
        nda_raw = kwa.get('nda_raw', None)
        raw = det_raw.raw(evt) if nda_raw is None else nda_raw
        if raw is None: return None

        gain = det_raw._gain() # - 3d gains shape:(1, 704, 768)
        gfac = det_raw._gain_factor()
        peds = det_raw._pedestals()
        mask = det_raw._mask()
        _cmpars  = None #(1,7,100) #None #cmpars # det_raw._common_mode() if cmpars is None else cmpars


        logger.debug('  %s\n  %s\n  %s\n  %s\n  %s\n' %(\
          info_ndarr(raw,'raw'),
          info_ndarr(gfac,'gfac'),
          info_ndarr(peds,'peds'),
          info_ndarr(mask,'mask'),
          info_ndarr(_cmpars,'cmpars')))

        if gfac is None: return None # gain = np.ones_like(peds)
        if peds is None: return None # peds = np.zeros_like(peds)

        arrf = raw - peds
        #arrf[0,352:,:]=5

        if _cmpars is not None:
          alg, mode, cormax = int(_cmpars[0]), int(_cmpars[1]), _cmpars[2]
          npixmin = _cmpars[3] if len(_cmpars)>3 else 10
          if mode>0:

            gmask = mask

            #shape:(1, 704, 768)
            hrows = 352 # int(704/2)

            for s in range(arrf.shape[0]):

              if mode & 4: # in banks: (352/2,384/8)=(176,48) pixels
                common_mode_2d_hsplit_nbanks(arrf[s,:hrows,:], mask=gmask[s,:hrows,:], nbanks=8, cormax=cormax, npix_min=npixmin)
                common_mode_2d_hsplit_nbanks(arrf[s,hrows:,:], mask=gmask[s,hrows:,:], nbanks=8, cormax=cormax, npix_min=npixmin)

              if mode & 1: # in rows per bank: 384/8 = 48 pixels # 190ms
                common_mode_rows_hsplit_nbanks(arrf[s,], mask=gmask[s,], nbanks=8, cormax=cormax, npix_min=npixmin)

              if mode & 2: # in cols per bank: 352/2 = 176 pixels # 150ms
                common_mode_cols(arrf[s,:hrows,:], mask=gmask[s,:hrows,:], cormax=cormax, npix_min=npixmin)
                common_mode_cols(arrf[s,hrows:,:], mask=gmask[s,hrows:,:], cormax=cormax, npix_min=npixmin)


        logger.debug('consumed time (sec) = %.6f' % (time()-t0_sec))
        #sys.exit('TEST EXIT')

        return arrf * gfac

# EOF
