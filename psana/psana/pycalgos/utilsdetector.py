#!/usr/bin/env python

"""
  python wrapper for C++/cython algorithms.
  See: UtilsDetector.cc/hh, utilsdetector_ext.pyx

Usage::

  import psana.pycalgos.utilsdetector as ud
  out  = np.empty(sh, dtype=np.float32)
  ud.calib_std(raw, peds, gain, mask, out)
"""
from psana.detector.NDArrUtils import info_ndarr

import psana.utilsdetector_ext as udext # !!! NAME utilsdetector_ext is defined in lcls2/psana/setup.py
from time import time

MSK =  0x3fff # 16383 or (1<<14)-1 - 14-bit mask

def calib_std(raw, peds, gain, mask, databits, out):
    """assuming that all numpy arrays have the same shape"""
    t0_sec = time()
    dt_us_cpp = udext.cy_calib_std(raw.ravel(), peds.ravel(), gain.ravel(), mask.ravel(), raw.size, databits, out.ravel())
    return dt_us_cpp, time()-t0_sec
#    print('in %s' % sys._getframe().f_code.co_name)

def calib_jungfrau_v0(raw, peds, gain, mask, out):
    """assuming that raw, mask and out numpy arrays have the same size/shape,
       while peds, gain have x3 size for 3 gain ranges
       databits is a mask for databits = 0x3fff
    """
    #t0_sec = time()
    dt_us_cpp = udext.cy_calib_jungfrau_v0(raw.ravel(), peds.ravel(), gain.ravel(), mask.ravel(), raw.size, out.ravel())
    out.shape = raw.shape
    return out #, dt_us_cpp, time()-t0_sec

def calib_jungfrau_v1(raw, cc, size_blk, out):
    """cc.shape = (<number-of-pixels-in detector>, <2-for-peds-and-gains>, <4-gain-ranges>) = (npix, 2, 4)
       size = size_blk * (int)number_of_blocks
       jungfrau:
         * databits = 0x3fff, gain bits 0x6000 0o140000
         * moving two gain bits to the right raw>>14
    """
    #t0_sec = time()
    assert raw.size % size_blk == 0, 'array size of raw data %d should be split for any number of equal blocks, current size_blk: %d' % (raw.size, size_blk)
    dt_us_cpp = udext.cy_calib_jungfrau_v1(raw.ravel(), cc, raw.size, size_blk, out.ravel())
    out.shape = raw.shape
    return out #, dt_us_cpp, time()-t0_sec

def calib_jungfrau_v2(raw, cc, size_blk, out):
    """See v1, shape of cc is .T
       cc.shape = (<2-for-peds-and-gains>, <4-gain-ranges>, <number-of-pixels-in detector>) = (2, 4, npix)
       size_blk - IS NOT USED
    """
    #t0_sec = time()
    #assert raw.size % size_blk == 0, 'array size of raw data %d should be split for any number of equal blocks, current size_blk: %d' % (raw.size, size_blk)
    dt_us_cpp = udext.cy_calib_jungfrau_v2(raw.ravel(), cc, raw.size, size_blk, out.ravel())
    out.shape = raw.shape
    return out #, dt_us_cpp, time()-t0_sec

def calib_jungfrau_v3(raw, cc, size_blk, out):
    """v3 different shape for constants
       cc.shape = (<4-gain-ranges>, <number-of-pixels-in detector>, <2-for-peds-and-gains>) = (4, npix,2 )
       size_blk - IS NOT USED
    """
    #t0_sec = time()
    #assert raw.size % size_blk == 0, 'array size of raw data %d should be split for any number of equal blocks, current size_blk: %d' % (raw.size, size_blk)
    dt_us_cpp = udext.cy_calib_jungfrau_v3(raw.ravel(), cc, raw.size, size_blk, out.ravel())
    out.shape = raw.shape
    return out #, dt_us_cpp, time()-t0_sec

def calib_jungfrau_v4_empty():
    """v4 empty test WITHOUT parameters for cython-c++ overhead"""
    t0_sec = time()
    dt_us_cpp = udext.cy_calib_jungfrau_v4_empty()
    return dt_us_cpp, time()-t0_sec

def calib_jungfrau_v5_empty(raw, cc, size_blk, out):
    """v5 empty test WITH parameters for cython-c++ overhead"""
    t0_sec = time()
    #print(info_ndarr(raw, '  raw :'))
    #print(info_ndarr(cc,  '  cc  :'))
    #print(info_ndarr(out, '  out :'))
    #t0_sec_ravel = time()
    #_raw = raw.ravel()
    #_cc = cc.ravel()
    #_out = out.ravel()
    #print('dt_sec_ravel', time()-t0_sec_ravel)
    #dt_us_cpp = udext.cy_calib_jungfrau_v5_empty(_raw, _cc, raw.size, size_blk, _out)
    #dt_us_cpp = udext.cy_calib_jungfrau_v5_empty(raw.ravel(), cc.ravel(), raw.size, size_blk, out.ravel())
    dt_us_cpp = udext.cy_calib_jungfrau_v5_empty(raw.ravel(), cc, raw.size, size_blk, out.ravel())
    out.shape = raw.shape
    return dt_us_cpp, time()-t0_sec





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
