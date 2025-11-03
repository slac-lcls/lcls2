#!/usr/bin/env python

"""
  python wrapper for C++/cython algorithms.
  See: UtilsDetector.cc/hh, utilsdetector_ext.pyx

Usage::

  import psana.pycalgos.utilsdetector as ud
  out  = np.empty(sh, dtype=np.float32)
  ud.calib_std(raw, peds, gain, mask, out)

"""
import utilsdetector_ext as udext # !!! NAME utilsdetector_ext is defined in lcls2/psana/setup.py
from time import time

def calib_std(raw, peds, gain, mask, databits, out):
    """assume that all numpy arrays have the same shape"""
    t0_sec = time()
    dt_us_cpp = udext.cy_calib_std(raw.ravel(), peds.ravel(), gain.ravel(), mask.ravel(), raw.size, databits, out.ravel())
    return dt_us_cpp, time()-t0_sec
#    print('in %s' % sys._getframe().f_code.co_name)

if __name__ == "__main__" :
    print("""self-test for calib_std""")
    from time import time
    import sys
    import numpy as np
    from psana.detector.NDArrUtils import info_ndarr
    import psana.pyalgos.generic.NDArrGenerators as ag
    databits = 0x3fff
    sh = (16, 352, 384)
    mask = np.ones(sh, dtype=np.uint8)
    raw  = np.ones(sh, dtype=np.uint16) * 10
    peds = np.ones(sh, dtype=np.float32) * 8
    gain = np.ones(sh, dtype=np.float32) * 2
    out  = np.empty(sh, dtype=np.float32)
    print(info_ndarr(raw,  'raw :'))
    print(info_ndarr(mask, 'mask:'))
    print(info_ndarr(peds, 'peds:'))
    print(info_ndarr(gain, 'gain:'))

    t0_sec = time()
    dt_cpp, dt_cy = calib_std(raw, peds, gain, mask, databits, out)

    print('dt_py: %.3f msec' % ((time()-t0_sec)*1000),\
          'cython: %.3f msec' % (dt_cy*1000),\
          'cpp: %.3f msec' % (dt_cpp*1e-3))

    print(info_ndarr(out, 'out :'))

# EOF
