
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
      1, 0         bad switch pixel status 64 catched in UtilsJungfrauCalib.py

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

try:
    from psana.detector import _jungfrau_calib
except ImportError:
    _jungfrau_calib = None

info_ndarr = ndau.info_ndarr

BW1 =  0o40000 # 16384 or 1<<14 (15-th bit starting from 1)
BW2 = 0o100000 # 32768 or 2<<14 or 1<<15
BW3 = 0o140000 # 49152 or 3<<14
MSK =  0x3fff # 16383 or (1<<14)-1 - 14-bit mask
BSH = 14

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

#class Cache():
#    """Wrapper around dict {detname:DetCache} for per-detector cache of calibration constants."""
#    def __init__(self):
#        self.calibcons = {}
#
#    def add_detcache(self, det, evt, **kwa):
#        detname = det._det_name # i.e.: jungfrau
#        assert isinstance(detname, str)
#        logger.debug('add_detcache for detector name: %s' % detname)
#        o = self.calibcons[detname] = DetCache(det, evt, **kwa)
#        return o
#
#    def detcache_for_detname(self, detname):
#        return self.calibcons.get(detname, None)
#
#    def detcache_for_detobject(self, det):
#        return self.detcache_for_detname(det._det_name)
#
#cache = Cache() # singleton

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
        self.outa = None # panel indices in daqoutput array, shaped as raw
        self.gmask = None # gfac * mask
        self.ccons = None # combined calibration constants, content controlled by cversion
        self.loop_banks = True
        self.poff_cpp = None
        self.gfac_cpp = None
        self.mask_cpp = None
        self._cpp_ready = False
        self._logmet_init = kwa.get('logmet_init', logger.debug)
        self.cversion = kwa.get('cversion', 0) # numerated version of cached constants
        self.add_calibcons(det, evt)

    def kwargs_are_the_same(self, **kwa):
        return self.kwa == kwa

    def _calibcons_for_ctype(self, ctype):
        nda_and_meta = self.calibc.get(ctype, None)
        if nda_and_meta is None:
            logger.debug('calibcons for ctype: %s are NON-AVAILABLE, use default' % ctype)
            return None, None
        return nda_and_meta # - 4d shape:(3, <nsegs>, 512, 1024)

    def add_calibcons(self, det, evt):

        self.detname = det._det_name
        self.inds    = det._sorted_segment_inds # det._segment_numbers
        self.npanels = len(self.inds) if self.inds is not None else 0
        self.calibc  = det._calibconst
        logmet_init = self._logmet_init

        logmet_init('%s add_calibcons for _det_name: %s %s' % (30*'_', self.detname, 30*'_'))
        raw_data = det.raw(evt)
        logmet_init('\n  _sorted_segment_inds: %s' % str(self.inds)\
                   + ndau.info_ndarr(raw_data, '\n  raw(evt)')
                    )
        if is_true(self.calibc is None, 'det._calibconst is None > CALIB CONSTANTS ARE NOT AVAILABLE FOR %s' % self.detname,\
                   logger_method=logger.warning): return

        keys = [k for k in self.calibc.keys()]     # because self.calibc is WikiDict.... and self.calibc.keys() is a generator...

        #print('XXXX keys', keys, type(keys))

        logmet_init('det.raw._calibconst.keys: %s' % (', '.join(keys)))
        if is_true(not('pedestals' in keys), 'PEDESTALS ARE NOT AVAILABLE det.raw.calib(evt) will return det.raw.raw(evt)',\
                   logger_method = logger.warning): return

        peds, meta_peds = self._calibcons_for_ctype('pedestals') # shape:(3, <nsegs>, 512, 1024) dtype:float32
        if is_true(peds is None, 'peds is None, det.raw.calib(evt) will return det.raw.raw(evt)',\
                   logger_method = logger.warning): return

        d = up.dict_filter(meta_peds, list_keys=('ctype', 'experiment', 'run', 'run_orig', 'run_beg', 'run_end', 'time_stamp',\
                                                 'tstamp_orig', 'detname', 'longname', 'shortname',\
                                                 'data_dtype', 'data_shape', 'version', 'uid'))
        logmet_init('partial metadata for pedestals:\n  %s' % '\n  '.join(['%15s: %s' % (k,v) for k,v in d.items()]))
        logmet_init(ndau.info_ndarr(peds, 'pedestals'))

        gain, meta_gain = self._calibcons_for_ctype('pixel_gain')
        offs, meta_offs = self._calibcons_for_ctype('pixel_offset')

        self.gfac = np.ones_like(peds) if is_true(gain is None, 'pixel_gain constants missing, use default ones',\
                                                  logger_method = logger.warning) else\
                    ndau.divide_protected(np.ones_like(peds), gain)

        if raw_data is not None:
            self.outa = np.zeros_like(raw_data, dtype=np.float32)
        else:
            self.outa = np.zeros(peds.shape[-3:], dtype=np.float32)

        logmet_init(ndau.info_ndarr(self.gfac, 'gain factors'))

        self.poff = peds if is_true(offs is None, 'pixel_offset constants missing, use default zeros',\
                                    logger_method = logger.debug) else\
                    peds + offs

        self.cmps = self.kwa.get('cmpars', None)
        self.loop_banks = self.kwa.get('loop_banks', True)

        logger.debug('before call det._mask(**self.kwa) from UtilsJungfrau DetCache.add_calibcons self.kwa: %s' % str(self.kwa))
        self.mask = det._mask(**self.kwa)
        logmet_init('cached constants:\n  %s\n  %s\n  %s\n  %s\n  %s' % (\
                      ndau.info_ndarr(self.mask, 'mask'),\
                      ndau.info_ndarr(self.cmps, 'cmps'),\
                      ndau.info_ndarr(self.inds, 'inds'),\
                      ndau.info_ndarr(self.outa, 'outa'),\
                      'loop over banks %s' % self.loop_banks))

        if self.cversion > 0:
            self.add_ccons()

        self.isset = True


    def add_gain_mask(self):
        """adds product of gain factor and mask: self.gmask = gfac*mask"""
        if self.mask is None:
            self.gmask = np.copy(self.gfac)
            return
        self.gmask = np.empty_like(self.gfac)
        for i in range(3):
            self.gmask[i,:] = self.gfac[i,:] * self.mask

    def _panel_stack_axis1(self, arr):
        """Return array reordered to match raw panel order along axis 1."""
        if arr is None or not self.inds:
            return arr
        slices = []
        axis_len = arr.shape[1]
        for seg in self.inds:
            iseg = int(seg)
            if iseg >= axis_len:
                iseg = 0
            slices.append(arr[:, iseg, ...])
        return np.stack(slices, axis=1)

    def _panel_stack_axis0(self, arr):
        """Return array reordered to match raw panel order along axis 0."""
        if arr is None or not self.inds:
            return arr
        slices = []
        axis_len = arr.shape[0]
        for seg in self.inds:
            iseg = int(seg)
            if iseg >= axis_len:
                iseg = 0
            slices.append(arr[iseg, ...])
        return np.stack(slices, axis=0)

    def prepare_cpp_constants(self):
        """Prepare contiguously ordered arrays for the compiled calibrator."""
        if self._cpp_ready or self.poff is None or self.gfac is None:
            return
        po = self._panel_stack_axis1(self.poff)
        gf = self._panel_stack_axis1(self.gfac)
        if po is None or gf is None:
            return
        self.poff_cpp = np.ascontiguousarray(po.astype(np.float32, copy=False))
        self.gfac_cpp = np.ascontiguousarray(gf.astype(np.float32, copy=False))
        mask_sel = self._panel_stack_axis0(self.mask)
        if mask_sel is not None:
            self.mask_cpp = np.ascontiguousarray(mask_sel.astype(np.float32, copy=False))
        else:
            self.mask_cpp = None
        self._cpp_ready = True


    def add_ccons(self):
        """make combined calibration constants
           ** of V1, ccons.shape = (<number-of-pixels-in detector>, <2-for-peds-and-gains>, <4-gain-ranges>) = (npix, 2, 4),
           ** of V2, (2, 4, npix),
           ** of V3, (4, npix, 2),
           ** peds = peds + offset, gain = gain * mask
        """
        self.add_gain_mask()
        po = self._panel_stack_axis1(self.poff)
        gm = self._panel_stack_axis1(self.gmask)
        self._logmet_init('DetCache.add_ccons combine cached constants for cversion %d:\n  %s\n  %s' % (\
                      self.cversion,\
                      ndau.info_ndarr(self.poff, 'poff', vfmt='%0.1f'),\
                      ndau.info_ndarr(self.gmask, 'gmask', vfmt='%0.4f')))
        arr0 = np.zeros(self.outa.size)

        if self.cversion in (1,2):
            self.ccons = np.vstack((po[0,:].ravel(), po[1,:].ravel(), arr0, po[2,:].ravel(),\
                                    gm[0,:].ravel(), gm[1,:].ravel(), arr0, gm[2,:].ravel()),\
                                    dtype=np.float32)  # .astype(np.float32)
            if self.cversion == 1:
                self.ccons = self.ccons.T

        elif self.cversion == 3:
            # test: lcls2/psana/psana/detector]$ testman/test-scaling-mpi-jungfrau.py -t6
            npix = po[0,:].size
            print('npix:', npix)
            self.ccons = np.vstack((
                            np.vstack((po[0,:].ravel(), gm[0,:].ravel())).T,
                            np.vstack((po[1,:].ravel(), gm[1,:].ravel())).T,
                            np.vstack((arr0, arr0)).T,
                            np.vstack((po[2,:].ravel(), gm[2,:].ravel())).T),
                            dtype=np.float32)
            self.ccons.shape = (4, npix, 2)

            self.check_cversion3_validity()
        #logger.info(ndau.info_ndarr(self.ccons, 'XXX ccons', last=100, vfmt='%0.3f'))

        #sys.exit('TEST EXIT')


    def check_cversion3_validity(self):
        po = self.poff
        gm = self.gmask
        cc = self.ccons

        logger.info(ndau.info_ndarr(cc, 'ccons      ', last=10, vfmt='%0.3f'))       # shape:(4, 16777216, 2)
        logger.info(ndau.info_ndarr(po, 'peds-offset', last=10, vfmt='%0.3f')) # shape:(3, 32, 512, 1024)
        logger.info(ndau.info_ndarr(gm, 'gain * mask', last=10, vfmt='%0.4f')) # shape:(3, 32, 512, 1024)

        assert np.array_equal(cc[0,:,0], po[0,:].ravel())
        assert np.array_equal(cc[1,:,0], po[1,:].ravel())
        assert np.array_equal(cc[3,:,0], po[2,:].ravel())

        assert np.array_equal(cc[0,:,1], gm[0,:].ravel())
        assert np.array_equal(cc[1,:,1], gm[1,:].ravel())
        assert np.array_equal(cc[3,:,1], gm[2,:].ravel())

        logger.info('passed check_cversion3_validity')


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
      - use_cpp_calib - when True (default False) applies a cython-based fast path
        that performs pedestal subtraction and gain application. The compiled
        implementation currently skips common-mode corrections, so it is only
        active when ``cmpars``/``cmps`` is ``None``.
    """

    kwa = dict(kwa)
    _t0_calib = time()
    logger.debug('calib_jungfrau **kwa: %s' % str(kwa))

    use_cpp_requested = kwa.get('use_cpp_calib', False)
    use_cpp = use_cpp_requested
    if use_cpp_requested and _jungfrau_calib is None:
        logger.warning('use_cpp_calib requested but _jungfrau_calib module is unavailable; falling back to python implementation')
        use_cpp = False
    kwa['use_cpp_calib'] = use_cpp

    nda_raw = kwa.get('nda_raw', None)

    arr = det.raw(evt) if nda_raw is None else nda_raw # shape:(<npanels>, 512, 1024) dtype:uint16

    if is_true(arr is None, 'det.raw(evt) and nda_raw are None, return None',\
               logger_method = logger.warning):
        print(f"[{det._det_name}] calib_jungfrau time (no data): {time() - _t0_calib:.6f} s")
        return None

    odc = det._odc # cache.detcache_for_detname(det._det_name)
    first_entry = odc is None

    if first_entry:
        det._odc = odc = DetCache(det, evt, **kwa) # cache.add_detcache(det, evt, **kwa)
        #logger.info(det._info_calibconst()) # is called in AreaDetector

    if odc.poff is None:
        print(f"[{det._det_name}] calib_jungfrau time (no pedestals): {time() - _t0_calib:.6f} s")
        return arr

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

    cpp_enabled = use_cpp and cmps is None and _jungfrau_calib is not None
    if use_cpp and cmps is not None:
        logger.debug('Common-mode parameters detected; disabling compiled Jungfrau calibration')

    if cpp_enabled:
        odc.prepare_cpp_constants()
        if odc.poff_cpp is None or odc.gfac_cpp is None:
            logger.debug('Compiled Jungfrau calibration not ready (missing constants); falling back to python implementation')
            cpp_enabled = False
        else:
            if odc.outa is None or odc.outa.shape != arr.shape:
                odc.outa = np.zeros_like(arr, dtype=np.float32)
            arr_c = np.ascontiguousarray(arr)
            _jungfrau_calib.calibrate_panels(arr_c,
                                             odc.poff_cpp,
                                             odc.gfac_cpp,
                                             odc.mask_cpp,
                                             odc.outa)
            print(f"[{det._det_name}] calib_jungfrau time (cpp path): {time() - _t0_calib:.6f} s")
            return odc.outa

    #nsegs = arr.shape[0]
    shseg = arr.shape[-2:] # (512, 1024)
    outa = np.zeros_like(arr, dtype=np.float32)

    #print('XXX inds:', inds)
    #print('XXX _sorted..., _segment_numbers:', det._sorted_segment_inds , det._segment_numbers)
    for iraw,i in enumerate(inds):
        arr1  = arr[iraw,:]

        #print('XXX i:', i)
        #print(info_ndarr(mask, 'XXX mask:'))

        mask1 = None if mask is None else mask[i,:] if i<mask.shape[0] else mask[0,:]
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
    logger.debug(info_ndarr(outa, '     outa '))
    print(f"[{det._det_name}] calib_jungfrau time (python path): {time() - _t0_calib:.6f} s")
    return outa


def gainbits_statistics(arr):
    #gb00 = arr & BW3 == 0                                   # 00
    #gb01 = np.logical_and(arr & BW1 == BW1, arr & BW2 == 0) # 01
    #gb10 = np.logical_and(arr & BW2 == BW2, arr & BW1 == 0) # 10
    #gb11 = arr & BW3 == BW3                                 # 11

    gbits = np.array(arr>>14, dtype=np.uint8)
    gb00, gb01, gb10, gb11 = gbits==0, gbits==1, gbits==2, gbits==3

    arr1 = np.ones_like(arr, dtype=np.uint32)
    arr_sta_gb00 = np.select((gb00,), (arr1,), 0)
    arr_sta_gb01 = np.select((gb01,), (arr1,), 0)
    arr_sta_gb10 = np.select((gb10,), (arr1,), 0)
    arr_sta_gb11 = np.select((gb11,), (arr1,), 0)
    ngb00, ngb01, ngb10, ngb11 =\
        arr_sta_gb00.sum(), arr_sta_gb01.sum(), arr_sta_gb10.sum(), arr_sta_gb11.sum()
    assert (ngb00 + ngb01 + ngb10 + ngb11) == arr.size
    total = ngb00 + ngb01 + ngb10 + ngb11
    return ngb00, ngb01, ngb10, ngb11, total, arr1.size


def info_gainbits_statistics(arr, fmt='gainbits statistics 00:%05d  01:%05d  10:%05d  11:%05d  total/arr.size:%6d/%6d'):
    #ngb00, ngb01, ngb10, ngb11, total, size = gainbits_statistics(arr)
    return fmt % gainbits_statistics(arr)


def gainrange_statistics(arr):
    #gr0 = arr <  BW1              # 00
    #gr1 =(arr >= BW1) & (arr<BW2) # 01
    #gr2 = arr >= BW3              # 11
    #bad =(arr >= BW2) & (arr<BW3) # 10 - badly frozen pixel

    gbits = np.array(arr>>14, dtype=np.uint8)
    gr0, gr1, gr2, bad = gbits==0, gbits==1, gbits==3, gbits==2

    arr1 = np.ones_like(arr, dtype=np.uint32)
    arr_sta_gr0 = np.select((gr0,), (arr1,), 0)
    arr_sta_gr1 = np.select((gr1,), (arr1,), 0)
    arr_sta_gr2 = np.select((gr2,), (arr1,), 0)
    arr_sta_bad = np.select((bad,), (arr1,), 0)
    return arr_sta_gr0.sum(), arr_sta_gr1.sum(), arr_sta_gr2.sum(), arr_sta_bad.sum()


def info_gainrange_statistics(arr, fmt='gainrange statistics 0:%d  1:%d  2:%d  bad:%d  total/arr.size:%d/%d'):
    ngr0, ngr1, ngr2, nbad = gainrange_statistics(arr)
    return fmt % (ngr0, ngr1, ngr2, nbad, ngr0+ngr1+ngr2+nbad, arr.size)


def gainrange_fractions(arr):
    ngr0, ngr1, ngr2, nbad = gainrange_statistics(arr)
    total = float(ngr0 + ngr1 + ngr2 + nbad)
    return ngr0/total, ngr1/total, ngr2/total, nbad/total, total


def info_gainrange_fractions(arr, fmt='gainrange fractions 0:%0.4f  1:%0.4f  2:%0.4f  bad:%0.4f  of total:%d'):
    fgr0, fgr1, fgr2, fbad, total = gainrange_fractions(arr)
    return fmt % gainrange_fractions(arr)


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
    #gbits = np.array(arr>>BSH, dtype=np.uint8)
    #gr0, gr1, gr2, bad = gbits==0, gbits==1, gbits==3, gbits==2
    #factor = np.select((gr0, gr1, gr2), (gfac[0,:], gfac[1,:], gfac[2,:]), default=0) # 2msec
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
