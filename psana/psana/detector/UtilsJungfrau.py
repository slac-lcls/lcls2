
"""
:py:class:`UtilsJungfrau`
==============================

Usage::
    from psana.detector.UtilsJungfrau import *
    import psana.detector.UtilsJungfrau as uj

Jungfrau gain range coding
bit: 15,14,...,0   Gain range, ind
      0, 0         Normal,       0
      0, 1         ForcedGain1,  1
      1, 1         FixedGain2,   2

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2021-04-05 by Mikhail Dubrovin
2025-03-05 - adopted to lcls2
"""

import logging
logger = logging.getLogger(__name__)

import os
import sys

import numpy as np
from time import time
import psana.detector.NDArrUtils as ndau
import psana.detector.UtilsCalib as uc
import psana.detector.utils_psana as up
import psana.detector.UtilsCommonMode as ucm

info_ndarr = ndau.info_ndarr

BW1 =  0o40000 # 16384 or 1<<14 (15-th bit starting from 1)
BW2 = 0o100000 # 32768 or 2<<14 or 1<<15
BW3 = 0o140000 # 49152 or 3<<14
MSK =  0x3fff # 16383 or (1<<14)-1 - 14-bit mask

MAX_DETNAME_SIZE = 20

#import psana.detector.Utils as ut
#is_true = ut.is_true

def is_true(cond, msg, logger_method=logger.debug):
    if cond: logger_method(msg)
    return cond

def jungfrau_segments_tot(segnum_max):
    """Returns total number of segments in the detector 1,2,8,32 depending on segnum_max."""
    return 1 if segnum_max<1 else\
           2 if segnum_max<2 else\
           8 if segnum_max<8 else\
           32

class Cache():
    """Wrapper around dict {detname:DetCache} for per-detector cache of calibration constants."""
    def __init__(self):
        self.calibcons = {}

    def add_detcache(self, det, evt, **kwa):
        detname = det._det_name # i.e.: jungfrau
        assert isinstance(detname, str)
        logger.debug('add_detcache for detector name: %s' % detname)
        o = self.calibcons[detname] = DetCache(det, evt, **kwa)
        return o

    def detcache_for_detname(self, detname):
        return self.calibcons.get(detname, None)

    def detcache_for_detobject(self, det):
        return self.detcache_for_detname(det._det_name)

cache = Cache() # singleton


class DetCache():
    """Cash of calibration constants for jungfrau."""
    def __init__(self, det, evt, **kwa):
        self.kwa = kwa
        self.isset = False
        self.poff = None # peds + offs
        self.gfac = None # 1/gain, keV/ADU
        self.cmps = None # common mode parameters
        self.mask = None # combined mask
        self.inds = None # panel indices in daq
        self.loop_banks = True
        self.add_calibcons(det, evt)

    def kwargs_are_the_same(self, **kwa):
        return self.kwa == kwa

    def _calibcons_for_ctype(self, ctype):
        nda_and_meta = self.calibc.get(ctype, None)
        if nda_and_meta is None:
            logger.warning('---> calibcons for ctype: %s are non-available, use default' % ctype)
            return None, None
        return nda_and_meta # - 4d shape:(3, <nsegs>, 512, 1024)

    def add_calibcons(self, det, evt):

        self.detname = det._det_name
        self.inds    = det._sorted_segment_inds # det._segment_numbers
        self.calibc  = det._calibconst

        logger.info('%s add_calibcons for _det_name: %s %s' % (30*'_', self.detname, 30*'_'))
        logger.info('\n  _sorted_segment_inds: %s' % str(self.inds)\
                   + ndau.info_ndarr(det.raw(evt), '\n  raw(evt)')
                    )
        if is_true(self.calibc is None, 'det._calibconst is None > CALIB CONSTANTS ARE NOT AVAILABLE FOR %s' % self.detname,\
                   logger_method=logger.warning): return

        keys = self.calibc.keys()
        logger.info('det.raw._calibconst.keys: %s' % (', '.join(keys)))
        if is_true(not('pedestals' in keys), 'PEDESTALS ARE NOT AVAILABLE det.raw.calib(evt) will return det.raw.raw(evt)',\
                   logger_method = logger.warning): return

        peds, meta_peds = self._calibcons_for_ctype('pedestals') # shape:(3, <nsegs>, 512, 1024) dtype:float32
        if is_true(peds is None, 'peds is None, det.raw.calib(evt) will return det.raw.raw(evt)',\
                   logger_method = logger.warning): return

        d = up.dict_filter(meta_peds, list_keys=('ctype', 'experiment', 'run', 'run_orig', 'run_beg', 'run_end', 'time_stamp',\
                                                 'tstamp_orig', 'detname', 'longname', 'shortname',\
                                                 'data_dtype', 'data_shape', 'version', 'uid'))
        logger.info('partial metadata for pedestals:\n  %s' % '\n  '.join(['%15s: %s' % (k,v) for k,v in d.items()]))
        logger.info(ndau.info_ndarr(peds, 'pedestals'))

        gain, meta_gain = self._calibcons_for_ctype('pixel_gain')
        offs, meta_offs = self._calibcons_for_ctype('pixel_offset')

        self.gfac = np.ones_like(peds) if is_true(gain is None, 'pixel_gain constants missing, use default ones',\
                                                  logger_method = logger.warning) else\
                    ndau.divide_protected(np.ones_like(peds), gain)

        logger.info(ndau.info_ndarr(self.gfac, 'gain factors'))

        self.poff = peds if is_true(offs is None, 'pixel_offset constants missing, use default zeros',\
                                    logger_method = logger.warning) else\
                    peds + offs

        self.cmps = self.kwa.get('cmpars', None)
        self.loop_banks = self.kwa.get('loop_banks', True)
        self.mask = det.mask_total(evt, **self.kwa)
        logger.info('cached constants:\n  %s\n  %s\n  %s\n  %s' % (\
                      ndau.info_ndarr(self.mask, 'mask'),\
                      ndau.info_ndarr(self.cmps, 'cmps'),\
                      ndau.info_ndarr(self.inds, 'inds'),\
                      'loop over banks %s' % self.loop_banks))

        self.isset = True


def calib_jungfrau(det, evt, **kwa): # cmpars=(7,3,200,10),
    """
    improved performance, reduce time and memory consumption, use peds-offset constants
    Returns calibrated jungfrau data

    - gets constants
    - gets raw data
    - evaluates (code - pedestal - offset)
    - applys common mode correction if turned on
    - apply gain factor

    Parameters

    - det (psana.Detector) - Detector object
    - evt (psana.Event)    - Event object
    - cmpars (tuple) - common mode parameters
        - cmpars[0] - algorithm # 7-for jungfrau
        - cmpars[1] - control bit-word 1-in rows, 2-in columns
        - cmpars[2] - maximal applied correction
    - **kwa - used here and passed to det.mask_v2 or det.mask_comb
      - nda_raw - if not None, substitutes evt.raw()
      - mbits - DEPRECATED parameter of the det.mask_comb(...)
      - mask - user defined mask passed as optional parameter
    """

    nda_raw = kwa.get('nda_raw', None)

    arr = det.raw(evt) if nda_raw is None else nda_raw # shape:(<npanels>, 512, 1024) dtype:uint16

    if is_true(arr is None, 'det.raw(evt) and nda_raw are None, return None',\
               logger_method = logger.warning): return None

    odc = cache.detcache_for_detname(det._det_name)
    first_entry = odc is None

    if first_entry:
        odc = cache.add_detcache(det, evt, **kwa)

    if odc.poff is None: return arr

    if kwa != odc.kwa:
        logger.warning('IGNORED ATTEMPT to call det.calib/image with different **kwargs (due to caching)'\
                       + '\n  **kwargs at first entry: %s' % str(odc.kwa)\
                       + '\n  **kwargs at this entry: %s' % str(kwa)\
                       + '\n  MUST BE FIXED - please consider to use the same **kwargs during the run in all calls to det.calib/image.')
    # 4d pedestals + offset shape:(3, 1, 512, 1024) dtype:float32

    poff, gfac, mask, cmps, inds =\
        odc.poff, odc.gfac, odc.mask, odc.cmps, odc.inds

    if first_entry:
        logger.debug('\n  ====================== det.name: %s' % det._det_name\
                   +info_ndarr(arr,  '\n  calib_jungfrau first entry:\n    arr ')\
                   +info_ndarr(poff, '\n    peds+off')\
                   +info_ndarr(gfac, '\n    gfac')\
                   +info_ndarr(mask, '\n    mask')\
                   +'\n    inds: segment indices: %s' % str(inds)\
                   +'\n    common mode parameters: %s' % str(cmps)\
                   +'\n    loop over segments: %s' % odc.loop_banks)

    #nsegs = arr.shape[0]
    shseg = arr.shape[-2:] # (512, 1024)
    outa = np.zeros_like(arr, dtype=np.float32)

    for iraw,i in enumerate(inds):
        arr1  = arr[iraw,:]
        mask1 = None if mask is None else mask[i,:]
        gfac1 = None if gfac is None else gfac[:,i,:,:]
        poff1 = None if poff is None else poff[:,i,:,:]
        arr1.shape  = (1,) + shseg
        if mask1 is not None: mask1.shape = (1,) + shseg
        if gfac1 is not None: gfac1.shape = (3,1,) + shseg
        if poff1 is not None: poff1.shape = (3,1,) + shseg
        out1 = calib_jungfrau_single_panel(arr1, gfac1, poff1, mask1, cmps)

        logger.debug('segment index %d arrays:' % i\
            + info_ndarr(arr1,  '\n  arr1 ')\
            + info_ndarr(poff1, '\n  poff1')\
            + info_ndarr(out1,  '\n  out1 '))
        outa[iraw,:] = out1[0,:]
    logger.debug(info_ndarr(outa, 'XXX  outa '))
    return outa


def info_gainbits(arr):
    gr0 = arr <  BW1
    gr1 =(arr >= BW1) & (arr<BW2)
    gr2 = arr >= BW3
    arr1 = np.ones_like(arr, dtype=np.uint64)
    arr_sta_gr0 = np.select((gr0,), (arr1,), 0)
    arr_sta_gr1 = np.select((gr1,), (arr1,), 0)
    arr_sta_gr2 = np.select((gr2,), (arr1,), 0)
    return 'statistics of gainbits 00: %5d  01: %5d  11: %5d' %\
        (arr_sta_gr0.sum(), arr_sta_gr1.sum(), arr_sta_gr2.sum())


def calib_jungfrau_single_panel(arr, gfac, poff, mask, cmps):
    """ example for 8-panel detector
    arr:  shape:(1, 512, 1024) size:524288 dtype:uint16 [2906 2945 2813 2861 3093...]
    poff: shape:(3, 1, 512, 1024) size:1572864 dtype:float32 [2922.283 2938.098 2827.207 2855.296 3080.415...]
    gfac: shape:(3, 1, 512, 1024) size:1572864 dtype:float32 [0.02490437 0.02543429 0.02541406 0.02539831 0.02544083...]
    mask: shape:(1, 512, 1024) size:524288 dtype:uint8 [1 1 1 1 1...]
    cmps: shape:(16,) size:16 dtype:float64 [  7.   1. 100.   0.   0....]
    """

    # Define bool arrays of ranges
    gr0 = arr <  BW1              # 490 us
    gr1 =(arr >= BW1) & (arr<BW2) # 714 us
    gr2 = arr >= BW3              # 400 us

    factor = np.select((gr0, gr1, gr2), (gfac[0,:], gfac[1,:], gfac[2,:]), default=1) # 2msec
    pedoff = np.select((gr0, gr1, gr2), (poff[0,:], poff[1,:], poff[2,:]), default=0)

    # Subtract offset-corrected pedestals
    arrf = np.array(arr & MSK, dtype=np.float32)
    arrf -= pedoff

    if cmps is not None:
      mode, cormax = int(cmps[1]), cmps[2]
      npixmin = cmps[3] if len(cmps)>3 else 10
      if mode>0:
        #arr1 = store.arr1
        #grhg = np.select((gr0,  gr1), (arr1, arr1), default=0)
        logger.debug(info_ndarr(gr0, 'gain group0'))
        logger.debug(info_ndarr(mask, 'mask'))
        t0_sec_cm = time()
        gmask = np.bitwise_and(gr0, mask) if mask is not None else gr0
        #sh = (nsegs, 512, 1024)
        hrows = 256 #512/2
        s = 0 # SINGLE SEGMENT ONLY, deprecated: for s in range(arrf.shape[0]):
        if True:
          if mode & 4: # in banks: (512/2,1024/16) = (256,64) pixels # 100 ms
            ucm.common_mode_2d_hsplit_nbanks(arrf[s,:hrows,:], mask=gmask[s,:hrows,:], nbanks=16, cormax=cormax, npix_min=npixmin)
            ucm.common_mode_2d_hsplit_nbanks(arrf[s,hrows:,:], mask=gmask[s,hrows:,:], nbanks=16, cormax=cormax, npix_min=npixmin)

          if mode & 1: # in rows per bank: 1024/16 = 64 pixels # 275 ms
            ucm.common_mode_rows_hsplit_nbanks(arrf[s,], mask=gmask[s,], nbanks=16, cormax=cormax, npix_min=npixmin)

          if mode & 2: # in cols per bank: 512/2 = 256 pixels  # 290 ms
            ucm.common_mode_cols(arrf[s,:hrows,:], mask=gmask[s,:hrows,:], cormax=cormax, npix_min=npixmin)
            ucm.common_mode_cols(arrf[s,hrows:,:], mask=gmask[s,hrows:,:], cormax=cormax, npix_min=npixmin)

        logger.debug('TIME: common-mode correction time = %.6f sec' % (time()-t0_sec_cm))

    arrf *= factor
    return arrf if mask is None else arrf * mask

#EOF
