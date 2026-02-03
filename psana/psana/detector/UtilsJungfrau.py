
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
import re
import hashlib

import numpy as np
from time import time, perf_counter
import psana.detector.NDArrUtils as ndau
import psana.detector.UtilsCalib as uc
import psana.detector.utils_psana as up
import psana.detector.UtilsCommonMode as ucm
import psana.pycalgos.utilsdetector as ud

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

info_ndarr = ndau.info_ndarr

BW1 =  0o40000 # 16384 or 1<<14 (15-th bit starting from 1)
BW2 = 0o100000 # 32768 or 2<<14 or 1<<15
BW3 = 0o140000 # 49152 or 3<<14
MSK =  0x3fff # 16383 or (1<<14)-1 - 14-bit mask
BSH = 14

MAX_DETNAME_SIZE = 20

# CALIBMET options
CALIB_PYT_V0    = 0
CALIB_CPP_V1    = 1
CALIB_CPP_V2    = 2
CALIB_CPP_V3    = 3
CALIB_CPP_V4    = 4
CALIB_CPP_V5    = 5

dic_calibmet = {CALIB_PYT_V0:    'CALIB_PYT_V0',\
                CALIB_CPP_V1:    'CALIB_CPP_V1',\
                CALIB_CPP_V2:    'CALIB_CPP_V2',\
                CALIB_CPP_V3:    'CALIB_CPP_V3',\
                CALIB_CPP_V4:    'CALIB_CPP_V4',\
                CALIB_CPP_V5:    'CALIB_CPP_V5'}

#import psana.detector.Utils as ut
#is_true = ut.is_true

def is_true(cond, msg, logger_method=logger.debug):
    if cond: logger_method(msg)
    return cond

def _jf_shared_prefix(det_name, runnum, cversion):
    safe = re.sub(r"[^0-9A-Za-z_]", "_", det_name)
    if runnum is None:
        return f"jf_{safe}_v{cversion}"
    return f"jf_{safe}_r{runnum}_v{cversion}"

_MASK_KWA_KEYS = {
    "status",
    "neighbors",
    "edges",
    "center",
    "calib",
    "umask",
    "dtype",
    "status_bits",
    "stextra_bits",
    "stci_bits",
    "gain_range_inds",
    "rad",
    "ptrn",
    "width",
    "edge_rows",
    "edge_cols",
    "wcenter",
    "center_rows",
    "center_cols",
    "force_update",
}


def _jf_mask_kwargs(det):
    kwa = getattr(det, "_kwargs", None) or {}
    return {k: kwa[k] for k in _MASK_KWA_KEYS if k in kwa}


def _jf_mask_key(mask_kwa):
    if not mask_kwa:
        return "default"
    if "umask" in mask_kwa and mask_kwa.get("umask") is not None:
        return None
    items = []
    for key in sorted(mask_kwa.keys()):
        val = mask_kwa[key]
        if isinstance(val, np.ndarray):
            items.append((key, ("ndarray", val.shape, str(val.dtype))))
        else:
            items.append((key, val))
    payload = repr(items)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _get_shared_jf(det, peds_shape, cversion):
    shared = getattr(det, "_jf_shared", None)
    if not shared:
        return None
    if shared.get("cversion") != cversion:
        return None
    if shared.get("shape") != peds_shape:
        return None
    return shared

def build_shared_jungfrau_calib(det, shared_mem, runnum=None, cversion=CALIB_CPP_V3):
    """Allocate and populate shared Jungfrau calibration arrays on the node leader."""
    if shared_mem is None or det is None:
        return None
    if cversion != CALIB_CPP_V3:
        return None

    calibc = getattr(det, "_calibconst", None)
    if not calibc:
        return None
    peds_entry = calibc.get("pedestals")
    if not peds_entry:
        return None
    peds, _meta = peds_entry
    if peds is None:
        return None

    prefix = _jf_shared_prefix(det._det_name, runnum, cversion)
    gfac_name = f"{prefix}_gfac"
    poff_name = f"{prefix}_poff"
    gmask_name = f"{prefix}_gmask"
    ccons_name = f"{prefix}_ccons"

    t_total_start = perf_counter()
    if (
        shared_mem.has_array(gfac_name)
        and shared_mem.has_array(poff_name)
        and shared_mem.has_array(gmask_name)
        and shared_mem.has_array(ccons_name)
    ):
        t_reuse_start = perf_counter()
        gfac = shared_mem.get_array(gfac_name)
        poff = shared_mem.get_array(poff_name)
        gmask = shared_mem.get_array(gmask_name)
        ccons = shared_mem.get_array(ccons_name)
        t_reuse_end = perf_counter()
        t_barrier_start = perf_counter()
        shared_mem.barrier()
        t_barrier_end = perf_counter()
        if shared_mem.is_leader:
            logger.debug(
                "Jungfrau shared calib reuse det=%s lookup=%.3fs barrier=%.3fs total=%.3fs"
                % (
                    det._det_name,
                    t_reuse_end - t_reuse_start,
                    t_barrier_end - t_barrier_start,
                    t_barrier_end - t_total_start,
                )
            )
        return {
            "gfac": gfac,
            "poff": poff,
            "gmask": gmask,
            "ccons": ccons.ravel(),
            "cversion": cversion,
            "shape": peds.shape,
        }

    t_alloc_start = perf_counter()
    dtype = peds.dtype
    gfac = shared_mem.allocate_array(gfac_name, peds.shape, dtype, zero_init=False)
    poff = shared_mem.allocate_array(poff_name, peds.shape, dtype, zero_init=False)
    gmask = shared_mem.allocate_array(gmask_name, peds.shape, dtype, zero_init=False)

    npix = int(np.prod(peds.shape[-3:]))
    ccons = shared_mem.allocate_array(ccons_name, (4, npix, 2), np.float32, zero_init=False)
    t_alloc_end = perf_counter()

    ok = True
    t_gfac = t_poff = t_mask = t_gmask = t_ccons = 0.0
    if shared_mem.is_leader:
        try:
            t_pop_start = perf_counter()
            gain_entry = calibc.get("pixel_gain")
            offs_entry = calibc.get("pixel_offset")
            gain = gain_entry[0] if gain_entry else None
            offs = offs_entry[0] if offs_entry else None

            t_gfac_start = perf_counter()
            if gain is None:
                gfac.fill(1.0)
            else:
                gfac[:] = ndau.divide_protected(np.ones_like(peds), gain)
            t_gfac = perf_counter() - t_gfac_start

            t_poff_start = perf_counter()
            if offs is None:
                np.copyto(poff, peds)
            else:
                np.add(peds, offs, out=poff)
            t_poff = perf_counter() - t_poff_start

            t_mask_start = perf_counter()
            mask = det._mask()
            if mask is None:
                mask = 1
            t_mask = perf_counter() - t_mask_start

            t_gmask_start = perf_counter()
            for i in range(3):
                np.multiply(gfac[i], mask, out=gmask[i], casting="unsafe")
            t_gmask = perf_counter() - t_gmask_start

            t_ccons_start = perf_counter()
            ccons_view = ccons
            ccons_view[0, :, 0] = poff[0].ravel()
            ccons_view[0, :, 1] = gmask[0].ravel()
            ccons_view[1, :, 0] = poff[1].ravel()
            ccons_view[1, :, 1] = gmask[1].ravel()
            ccons_view[2, :, 0].fill(0)
            ccons_view[2, :, 1].fill(0)
            ccons_view[3, :, 0] = poff[2].ravel()
            ccons_view[3, :, 1] = gmask[2].ravel()
            t_ccons = perf_counter() - t_ccons_start
            t_pop_end = perf_counter()
            t_populate = t_pop_end - t_pop_start
        except Exception as exc:
            ok = False
            logger.error("Failed to populate shared Jungfrau calibration constants: %s" % exc)

    t_bcast_start = perf_counter()
    shm_comm = getattr(shared_mem, "shm_comm", None)
    if shm_comm is not None:
        ok = shm_comm.bcast(ok, root=0)
    t_bcast_end = perf_counter()
    t_barrier_start = perf_counter()
    shared_mem.barrier()
    t_barrier_end = perf_counter()
    if not ok:
        return None
    if shared_mem.is_leader:
        logger.debug(
            "Jungfrau shared calib build det=%s alloc=%.3fs populate=%.3fs gfac=%.3fs poff=%.3fs "
            "mask=%.3fs gmask=%.3fs ccons=%.3fs bcast=%.3fs barrier=%.3fs total=%.3fs"
            % (
                det._det_name,
                t_alloc_end - t_alloc_start,
                t_populate if ok else 0.0,
                t_gfac,
                t_poff,
                t_mask,
                t_gmask,
                t_ccons,
                t_bcast_end - t_bcast_start,
                t_barrier_end - t_barrier_start,
                t_barrier_end - t_total_start,
            )
        )

    return {
        "gfac": gfac,
        "poff": poff,
        "gmask": gmask,
        "ccons": ccons.ravel(),
        "cversion": cversion,
        "shape": peds.shape,
    }


def build_shared_jungfrau_mask(det, shared_mem, runnum=None, cversion=CALIB_CPP_V3):
    """Allocate and populate shared Jungfrau mask array on the node leader."""
    if shared_mem is None or det is None:
        return None
    if cversion != CALIB_CPP_V3:
        return None

    mask_kwa = _jf_mask_kwargs(det)
    mask_key = _jf_mask_key(mask_kwa)
    if mask_key is None:
        return None

    prefix = _jf_shared_prefix(det._det_name, runnum, cversion)
    mask_name = f"{prefix}_mask_{mask_key}"

    if shared_mem.has_array(mask_name):
        mask_arr = shared_mem.get_array(mask_name)
        if hasattr(shared_mem, "barrier"):
            shared_mem.barrier()
        return {
            "mask": mask_arr,
            "mask_key": mask_key,
            "mask_shape": mask_arr.shape,
        }

    shm_comm = getattr(shared_mem, "shm_comm", None)
    ok = True
    shape = None
    dtype_str = None
    mask = None
    t_total_start = perf_counter()

    if shared_mem.is_leader:
        try:
            t_mask_start = perf_counter()
            mask = det._mask(**mask_kwa)
            t_mask = perf_counter() - t_mask_start
            if mask is None:
                ok = False
            else:
                shape = mask.shape
                dtype_str = mask.dtype.str
        except Exception:
            ok = False
            t_mask = 0.0
    else:
        t_mask = 0.0

    if shm_comm is not None:
        ok = shm_comm.bcast(ok, root=0)
        shape = shm_comm.bcast(shape, root=0)
        dtype_str = shm_comm.bcast(dtype_str, root=0)

    if not ok or shape is None or dtype_str is None:
        return None

    mask_arr = shared_mem.allocate_array(
        mask_name, shape, np.dtype(dtype_str), zero_init=False
    )
    if shared_mem.is_leader and mask is not None:
        np.copyto(mask_arr, mask, casting="unsafe")

    if hasattr(shared_mem, "barrier"):
        shared_mem.barrier()

    t_total = perf_counter() - t_total_start
    if shared_mem.is_leader:
        logger.debug(
            "Jungfrau shared mask build det=%s mask=%.3fs total=%.3fs",
            det._det_name,
            t_mask,
            t_total,
        )

    return {
        "mask": mask_arr,
        "mask_key": mask_key,
        "mask_shape": mask_arr.shape,
    }

def jungfrau_segments_tot(segnum_max):
    """Returns total number of segments in the detector 1,2,8,32 depending on segnum_max."""
    return 1 if segnum_max<1 else\
           2 if segnum_max<2 else\
           8 if segnum_max<8 else\
           32

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
        self._logmet_init = kwa.get('logmet_init', logger.debug)
        self.cversion = kwa.get('cversion', 3) # numerated version of cached constants
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
        self.calibc  = det._calibconst
        logmet_init = self._logmet_init

        logmet_init('%s add_calibcons for _det_name: %s %s' % (30*'_', self.detname, 30*'_'))
        logmet_init('\n  _sorted_segment_inds: %s' % str(self.inds)\
                   + ndau.info_ndarr(det.raw(evt), '\n  raw(evt)')
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

        t_start = perf_counter()
        t_shared_lookup_start = perf_counter()
        shared = _get_shared_jf(det, peds.shape, self.cversion)
        t_shared_lookup = perf_counter() - t_shared_lookup_start
        t_gfac = 0.0
        t_poff = 0.0
        t_ccons = 0.0
        if shared:
            t_assign_start = perf_counter()
            self.gfac = shared.get("gfac")
            self.poff = shared.get("poff")
            self.gmask = shared.get("gmask")
            self.ccons = shared.get("ccons")
            t_assign = perf_counter() - t_assign_start
        else:
            t_gfac_start = perf_counter()
            self.gfac = np.ones_like(peds) if is_true(gain is None, 'pixel_gain constants missing, use default ones',\
                                                      logger_method = logger.warning) else\
                        ndau.divide_protected(np.ones_like(peds), gain)
            t_gfac = perf_counter() - t_gfac_start

        self.outa = np.zeros(peds.shape[-3:], dtype=np.float32, order='C').ravel()
        self.outa = np.ascontiguousarray(self.outa).ravel()

        logmet_init(ndau.info_ndarr(self.gfac, 'gain factors'))

        if not shared:
            t_poff_start = perf_counter()
            self.poff = peds if is_true(offs is None, 'pixel_offset constants missing, use default zeros',\
                                        logger_method = logger.debug) else\
                        peds + offs
            t_poff = perf_counter() - t_poff_start

        self.cmps = self.kwa.get('cmpars', None)
        self.loop_banks = self.kwa.get('loop_banks', True)

        logger.debug('before call det._mask(**self.kwa) from UtilsJungfrau DetCache.add_calibcons self.kwa: %s' % str(self.kwa))
        t_mask_start = perf_counter()
        mask_shared = False
        mask_key = _jf_mask_key(_jf_mask_kwargs(det))
        if shared and mask_key is not None:
            shared_mask = shared.get("mask")
            if (
                shared_mask is not None
                and shared.get("mask_key") == mask_key
                and shared.get("mask_shape") == shared_mask.shape
            ):
                self.mask = shared_mask
                mask_shared = True
        if not mask_shared:
            self.mask = det._mask(**self.kwa)
        t_mask = perf_counter() - t_mask_start
        if mask_shared:
            logger.debug("DetCache.add_calibcons mask shared det=%s key=%s", self.detname, mask_key)
        else:
            logger.debug("DetCache.add_calibcons mask local det=%s key=%s", self.detname, mask_key)
        logmet_init('cached constants:\n  %s\n  %s\n  %s\n  %s\n  %s' % (\
                      ndau.info_ndarr(self.mask, 'mask'),\
                      ndau.info_ndarr(self.cmps, 'cmps'),\
                      ndau.info_ndarr(self.inds, 'inds'),\
                      ndau.info_ndarr(self.outa, 'outa'),\
                      'loop over banks %s' % self.loop_banks))

        if self.cversion > 0 and not shared:
            t_ccons_start = perf_counter()
            self.add_ccons()
            t_ccons = perf_counter() - t_ccons_start

        total_time = perf_counter() - t_start
        if shared:
            logger.debug(
                "DetCache.add_calibcons shared det=%s lookup=%.3fs assign=%.3fs mask=%.3fs total=%.3fs",
                self.detname,
                t_shared_lookup,
                t_assign,
                t_mask,
                total_time,
            )
        else:
            logger.debug(
                "DetCache.add_calibcons local det=%s lookup=%.3fs gfac=%.3fs poff=%.3fs mask=%.3fs ccons=%.3fs total=%.3fs",
                self.detname,
                t_shared_lookup,
                t_gfac,
                t_poff,
                t_mask,
                t_ccons,
                total_time,
            )

        self.isset = True


    def add_gain_mask(self):
        """adds product of gain factor and mask: self.gmask = gfac*mask"""
        self.gmask = np.empty_like(self.gfac)
        for i in range(3):
            self.gmask[i,:] = self.gfac[i,:] * self.mask


    def add_ccons(self):
        """make combined calibration constants for self.cversion = 1/2/3
           ** of V1, (npix, 2, 4), ccons.shape = (<number-of-pixels-in detector>, <2-for-peds-and-gains>, <4-gain-ranges>)
           ** of V2, (2, 4, npix),
           ** of V3, (4, npix, 2),
           ** po = peds + offset, gm = gain * mask
        """
        self.add_gain_mask()
        po = self.poff
        gm = self.gmask
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

        elif self.cversion > 2:
            # test: lcls2/psana/psana/detector]$ testman/test-scaling-mpi-jungfrau.py -t6
            npix = po[0,:].size
            #print('npix:', npix)
            self.ccons = np.vstack((
                            np.vstack((po[0,:].ravel(), gm[0,:].ravel())).T,
                            np.vstack((po[1,:].ravel(), gm[1,:].ravel())).T,
                            np.vstack((arr0, arr0)).T,
                            np.vstack((po[2,:].ravel(), gm[2,:].ravel())).T),
                            dtype=np.float32)
            self.ccons.shape = (4, npix, 2)
            self.check_cversion3_validity()

        logger.debug(ndau.info_ndarr(self.ccons, 'DetCache.add_ccons ccons:', last=100, vfmt='%0.3f'))
        self.ccons = np.ascontiguousarray(self.ccons).ravel()


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
    """

    logger.debug('calib_jungfrau **kwa: %s' % str(kwa))

    nda_raw = kwa.get('nda_raw', None)

    arr = det.raw(evt) if nda_raw is None else nda_raw # shape:(<npanels>, 512, 1024) dtype:uint16

    if is_true(arr is None, 'det.raw(evt) and nda_raw are None, return None',\
               logger_method = logger.warning): return None

    odc = det._odc # cache.detcache_for_detname(det._det_name)
    first_entry = odc is None

    if first_entry:
        det._odc = odc = DetCache(det, evt, **kwa) # cache.add_detcache(det, evt, **kwa)
        #logger.info(det._info_calibconst()) # is called in AreaDetector

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
        #out1 = calib_jungfrau_single_panel_v0(arr1, gfac1, poff1, mask1, cmps)

        logger.debug('segment index %d arrays:' % i\
            + info_ndarr(arr1,  '\n  arr1 ')\
            + info_ndarr(poff1, '\n  poff1')\
            + info_ndarr(out1,  '\n  out1 '))
        outa[iraw,:] = out1[0,:]
    logger.debug(info_ndarr(outa, '     outa '))
    return outa


def gainbits_statistics(arr):
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


#def calib_jungfrau_single_panel_v0(arr, gfac, poff, mask, cmps):
#    """ example for 8-panel detector
#    arr:  shape:(1, 512, 1024) size:524288 dtype:uint16 [2906 2945 2813 2861 3093...]
#    poff: shape:(3, 1, 512, 1024) size:1572864 dtype:float32 [2922.283 2938.098 2827.207 2855.296 3080.415...]
#    gfac: shape:(3, 1, 512, 1024) size:1572864 dtype:float32 [0.02490437 0.02543429 0.02541406 0.02539831 0.02544083...]
#    mask: shape:(1, 512, 1024) size:524288 dtype:uint8 [1 1 1 1 1...]
#    cmps: shape:(16,) size:16 dtype:float64 [  7.   1. 100.   0.   0....]
#    """
#    # Define bool arrays of ranges
#    #t0_sec = time()
#    gr0 = arr <  BW1              # 490 us
#    gr1 =(arr >= BW1) & (arr<BW2) # 714 us
#    gr2 = arr >= BW3              # 400 us
#    #print('XXX V0 make gain bit arrays time = %.6f sec' % (time()-t0_sec)) # 190 us
#    factor = np.select((gr0, gr1, gr2), (gfac[0,:], gfac[1,:], gfac[2,:]), default=0) # 2msec
#    pedoff = np.select((gr0, gr1, gr2), (poff[0,:], poff[1,:], poff[2,:]), default=0)
#
#    # Subtract offset-corrected pedestals
#    arrf = np.array(arr & MSK, dtype=np.float32)
#    arrf -= pedoff
#
#    if cmps is not None:
#      mode, cormax = int(cmps[1]), cmps[2]
#      npixmin = cmps[3] if len(cmps)>3 else 10
#      if mode>0:
#        #arr1 = store.arr1
#        #grhg = np.select((gr0,  gr1), (arr1, arr1), default=0)
#        logger.debug(info_ndarr(gr0, 'gain group0'))
#        logger.debug(info_ndarr(mask, 'mask'))
#        t0_sec_cm = time()
#        gmask = np.bitwise_and(gr0, mask) if mask is not None else gr0
#        #sh = (nsegs, 512, 1024)
#        hrows = 256 #512/2
#        s = 0 # SINGLE SEGMENT ONLY, deprecated: for s in range(arrf.shape[0]):
#        if True:
#          if mode & 4: # in banks: (512/2,1024/16) = (256,64) pixels # 100 ms
#            ucm.common_mode_2d_hsplit_nbanks(arrf[s,:hrows,:], mask=gmask[s,:hrows,:], nbanks=16, cormax=cormax, npix_min=npixmin)
#            ucm.common_mode_2d_hsplit_nbanks(arrf[s,hrows:,:], mask=gmask[s,hrows:,:], nbanks=16, cormax=cormax, npix_min=npixmin)
#
#          if mode & 1: # in rows per bank: 1024/16 = 64 pixels # 275 ms
#            ucm.common_mode_rows_hsplit_nbanks(arrf[s,], mask=gmask[s,], nbanks=16, cormax=cormax, npix_min=npixmin)
#
#          if mode & 2: # in cols per bank: 512/2 = 256 pixels  # 290 ms
#            ucm.common_mode_cols(arrf[s,:hrows,:], mask=gmask[s,:hrows,:], cormax=cormax, npix_min=npixmin)
#            ucm.common_mode_cols(arrf[s,hrows:,:], mask=gmask[s,hrows:,:], cormax=cormax, npix_min=npixmin)
#
#        logger.debug('TIME: common-mode correction time = %.6f sec' % (time()-t0_sec_cm))
#
#    arrf *= factor
#    return arrf if mask is None else arrf * mask


def calib_jungfrau_single_panel(arr, gfac, poff, mask, cmps):
    """ The same as calib_jungfrau_single_panel_v0,
        CHANGED: defenition of gr0, gr1, gr2, bad
    """
    # Define bool arrays of ranges
    #t0_sec = time()
    gbits = np.array(arr>>14, dtype=np.uint8) # 00/01/11 - gain bits for mode 0,1,2
    gr0, gr1, gr2 = gbits==0, gbits==1, gbits==3
    #print('XXX V1 make gain bit arrays time = %.6f sec' % (time()-t0_sec)) # 180 us
    factor = np.select((gr0, gr1, gr2), (gfac[0,:], gfac[1,:], gfac[2,:]), default=0) # 2msec
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




def calib_jungfrau_versions(det_raw, evt, **kwa): # cmpars=(7,3,200,10), self.cversion = kwa.get('cversion', 0)
    """
       - switch for calib versions C++
    """
    nda_raw = kwa.get('nda_raw', None)
    size_blk = kwa.get('size_blk', 512*1024) # single panel size

    # Avoid extra raw() copy here; calib immediately consumes raw per-event data.
    # This keeps perf while raw() defaults to copy=True for safety elsewhere.
    arr = det_raw.raw(evt, copy=False) if nda_raw is None else nda_raw # shape:(<npanels>, 512, 1024) dtype:uint16

    if is_true(arr is None, 'det_raw.raw(evt) and nda_raw are None, return None',\
               logger_method = logger.warning): return None

    odc = det_raw._odc # cache.detcache_for_detname(det_raw._det_name)
    first_entry = odc is None

    if first_entry:
        #kwa.setdefault('cversion', 3)
        det_raw._odc = odc = DetCache(det_raw, evt, **kwa) # cache.add_detcache(det_raw, evt, **kwa)
        logger.info('calib_jungfrau **kwa: %s' % str(kwa))
        logger.info(det_raw._info_calibconst()) # is called in AreaDetector

    if odc.poff is None: return arr

    if kwa != odc.kwa:
        logger.warning('IGNORED ATTEMPT to call det_raw.calib/image with different **kwargs (due to caching)'\
                       + '\n  **kwargs at first entry: %s' % str(odc.kwa)\
                       + '\n  **kwargs at this entry: %s' % str(kwa)\
                       + '\n  MUST BE FIXED - please consider to use the same **kwargs during the run in all calls to det.calib/image.')
    # 4d pedestals + offset shape:(3, 1, 512, 1024) dtype:float32

    ccons, poff, gfac, mask, cmps, inds, outa, cversion =\
        odc.ccons, odc.poff, odc.gfac, odc.mask, odc.cmps, odc.inds, odc.outa, odc.cversion

    if first_entry:
        # NOTE: This block is intentionally disabled to avoid expensive full-array
        # stats/log construction that can add seconds on event 0. Re-enable only
        # for deep debugging.
        #         logger.info('\n  ====================== det.name: %s calibmet: %s' % (det_raw._det_name, dic_calibmet[cversion])\
        #                    +info_ndarr(arr,  '\n  calib_jungfrau first entry:\n    arr ')\
        #                    +info_ndarr(poff, '\n    peds+off')\
        #                    +info_ndarr(gfac, '\n    gfac')\
        #                    +info_ndarr(mask, '\n    mask')\
        #                    +info_ndarr(outa, '\n    outa')\
        #                    +info_ndarr(ccons, '\n   ccons (peds+off, gain*mask)', last=8, vfmt='%0.3f')\
        #                    +'\n    inds: segment indices: %s' % str(inds)\
        #                    +'\n    common mode parameters: %s' % str(cmps)\
        #                    +'\n    loop over segments: %s' % odc.loop_banks)
        #         logger.debug(info_gainbits_statistics(arr))
        #         logger.debug(info_gainrange_statistics(arr))
        #         logger.debug(info_gainrange_fractions(arr))
        pass

    if cversion == CALIB_CPP_V3:   # ccons.shape = (4, <NPIXELS>, 2)
        ud.calib_jungfrau_v3(arr, ccons, size_blk, outa)

    elif cversion == CALIB_CPP_V1: # ccons.shape = (<NPIXELS>,8)
        ud.calib_jungfrau_v1(arr, ccons, size_blk, outa)

    elif cversion == CALIB_CPP_V2: # ccons.shape = (8,<NPIXELS>)
        ud.calib_jungfrau_v2(arr, ccons, size_blk, outa)

    elif cversion == CALIB_PYT_V0:
        return calib_jungfrau(det_raw, evt, **kwa)

    elif cversion == CALIB_CPP_V4: # constants shape = IS NOT USED
        return ud.calib_jungfrau_v4_empty()

    elif cversion == CALIB_CPP_V5: # constants shape = IS NOT USED
        return ud.calib_jungfrau_v5_empty(arr, ccons, size_blk, outa)
    return outa

#EOF
