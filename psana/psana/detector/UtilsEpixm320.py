
"""
:py:class:`UtilsEpixm320` contains utilities for epixm320
=========================================================

Usage::
    import psana.detector.UtilsEpixm320 as uem

EPIXM (my): https://confluence.slac.stanford.edu/spaces/PSDM/pages/436570012/EPIXM

@author Mikhail Diubrovin
Created on 2026-10-07
"""
#import os
import sys
import numpy as np
from time import time
import psana.detector.NDArrUtils as ndu
import psana.detector.Utils as ut # info_dict, is_true, is_none

is_none = ut.is_none

#def calib_v01(det_raw, evt, **kwa):
#    print('UtilsEpixm320: calib_v01 - returns raw')
#    return det_raw.raw(evt)

def calib_v01(det_raw, evt) -> Array3d: # already defined in epix_base and AreaDetectorRaw
    """  """
    #logger.debug('%s.%s' % (det_raw.__class__.__name__, sys._getframe().f_code.co_name))
    #print('TBD: %s.%s' % (det_raw.__class__.__name__, sys._getframe().f_code.co_name))

    print('UtilsEpixm320: calib_v01 - returns raw - peds')

    if is_none(evt, 'evt is None - return None', logger.debug): return None

    #t0_sec = time()
    raw = det_raw.raw(evt)
    if is_none(raw, 'det_raw.raw(evt) is None - return None'): return None

    # Subtract pedestals
    peds = det_raw._pedestals()
    if is_none(peds, 'det.raw._pedestals() is None - return det.raw.raw(evt)', logger.debug):
        return raw
    #print(info_ndarr(peds,'XXX peds', first=1000, last=1005))

    gr1 = (raw & det_raw._data_gain_bit) > 0

    #print(info_ndarr(gr1,'XXX gr1', first=1000, last=1005))
    pedgr = np.select((gr1,), (peds[1,:],), default=peds[0,:])
    arrf = np.array(raw & det_raw._data_bit_mask, dtype=np.float32)
    arrf -= pedgr

    #print('XXX time for calib: %.6f sec' % (time()-t0_sec)) # 4ms on drp-neh-cmp001
    mask = det_raw._mask()

    #print(info_ndarr(mask,'XXX mask', first=1000, last=1005)) # IT WORKS mask is available

    return arrf if is_none(mask, 'det.raw._mask() is None - return raw-peds', logger.info) else\
           arrf * mask

# EOF
