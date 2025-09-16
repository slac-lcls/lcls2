
"""
:py:class:`UtilsEpix100` - utilities for epix... detectors
===========================================================

Usage::
    import psana.detector.UtilsEpix100 as ue100
    kwa = {'nda_raw', raw-peds} # optional
    ue100.calib_epix100(det_raw, evt, cmpars=(0,7,100.,10), **kwa)


This software was developed for the LCLS project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2022-02-07 by Mikhail Dubrovin
"""

from time import time

import psana.detector.UtilsCommonMode as ucm
import psana.detector.Utils as ut
logger = ucm.logging.getLogger(__name__)

is_none = ut.is_none

info_ndarr = ucm.info_ndarr
GAIN_FACTOR_DEFAULT = 0.06 # keV/ADU on 2022/0/03 gain factor(Philip) = 60 eV/ADU, gain(Conny) = 16.4 ADU/keV
GAIN_DEFAULT = 1./GAIN_FACTOR_DEFAULT # ADU/keV


def calib_epix100(det_raw, evt, cmpars=None, **kwa): #cmpars=(0,7,100,10)):

    t0_sec = time()

    nda_raw = kwa.get('nda_raw', None)
    raw = det_raw.raw(evt) if nda_raw is None else nda_raw
    if raw is None: return None

    gfac = det_raw._gain_factor()# - 3d gain factors shape:(1, 704, 768)
    peds = det_raw._pedestals()
    mask = det_raw._mask() # **kwa
    _cmpars  = cmpars #(0,7,100,10) #None #cmpars # det_raw._common_mode() if cmpars is None else cmpars

    if peds is None: return raw

    arrf = raw - peds
    #arrf[0,352:,:]=5 # for debug

    if _cmpars is not None: ucm.common_mode_apply(arrf, mask, cmpars=_cmpars)

#    logger.debug('calib_epix100 consumed time (sec) = %.6f  \n  %s\n  %s\n  %s\n  %s\n  %s\n' %(\
#      (time()-t0_sec),
#      info_ndarr(raw,'raw'),
#      info_ndarr(gfac,'gfac'),
#      info_ndarr(peds,'peds'),
#      info_ndarr(mask,'mask'),
#      info_ndarr(_cmpars,'cmpars')))

    return arrf * gfac if mask is None else arrf * gfac * mask


def common_mode_increment(det_raw, evt, cmpars=None, **kwa):
    logger.debug('In common_mode_increment')
    if cmpars is None: return None

    nda_raw = kwa.get('nda_raw', None)
    raw = det_raw.raw(evt) if nda_raw is None else nda_raw
    if raw is None: return None

    peds = det_raw._pedestals()
    mask = det_raw._mask()

    if peds is None: return None # peds = np.zeros_like(peds)

    arrf = raw - peds
    ucm.common_mode_apply(arrf, mask, cmpars=cmpars)

    return arrf - raw + peds

# EOF
