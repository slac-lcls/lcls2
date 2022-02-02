
"""
:py:class:`UtilsEpix10ka` contains utilities for epix10ka and its composite detectors
=====================================================================================

Usage::
    from psana.detector.UtilsEpix10ka import ...

    #inds = segment_indices_det(det)
    #long_name = fullname_det(det)
    #ids = segment_ids_det(det)
    o = config_object_det(det, detname=None)
    #o = config_object_det_raw(det_raw)
    cbits = cbits_config_epix10ka(cob)  # used in det.raw._cbits_config_segment(cob)
    cbits = cbits_config_epixhr2x2(cob) # used in det.raw._cbits_config_segment(cob)
    cbits = cbits_config_and_data_detector_epix10ka(det_raw, evt=None)  # used in det.raw._cbits_config_and_data_detector(evt)
    cbits = cbits_config_and_data_detector_epixhr2x2(det_raw, evt=None) # used in det.raw._cbits_config_and_data_detector(evt)
    maps = gain_maps_epix10ka_any(det_raw, evt=None)
    s = def info_gain_mode_arrays(gmaps, first=0, last=5)
    gmstatist = pixel_gain_mode_statistics(gmaps)
    s = info_pixel_gain_mode_statistics(gmaps)
    s = info_pixel_gain_mode_statistics_for_raw(det_raw, evt=None, msg='pixel gain mode statistics: ')
    gmfs = pixel_gain_mode_fractions(det_raw, evt=None)
    s = info_pixel_gain_mode_for_fractions(grp_prob, msg='pixel gain mode fractions: ')
    s = info_pixel_gain_mode_fractions(det_raw, evt=None, msg='pixel gain mode fractions: ')
    gmind = gain_mode_index_from_fractions(gmfs)
    gmind = find_gain_mode_index(det_raw, evt=None)
    gmode = gain_mode_name_for_index(ind)
    gmode = find_gain_mode(det_raw, evt=None)
    calib = calib_epix10ka_any(det_raw, evt, cmpars=None, **kwa)
    calib = calib_epix10ka_any(det_raw, evt, cmpars=(7,2,100,10),\
                            mbits=0o7, mask=None, edge_rows=10, edge_cols=10, center_rows=5, center_cols=5)

This software was developed for the LCLS project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2020-12-03 by Mikhail Dubrovin for LCLS2 from LCLS1
"""

import os
import sys
import numpy as np
from time import time

import logging
logger = logging.getLogger(__name__)

from psana.detector.NDArrUtils import info_ndarr, divide_protected
from psana.detector.UtilsMask import merge_masks, DTYPE_MASK
from psana.detector.UtilsCommonMode import common_mode_cols,\
  common_mode_rows_hsplit_nbanks, common_mode_2d_hsplit_nbanks

GAIN_MODES    = ['FH','FM','FL','AHL-H','AML-M','AHL-L','AML-L']
GAIN_MODES_IN = ['FH','FM','FL','AHL-H','AML-M']

B04 =    0o20 #    16 or 1<<4   (5-th bit starting from 1)
B05 =    0o40 #    32 or 1<<5   (6-th bit starting from 1)

# epix10ka data gainbit and mask
B14 = 0o40000 # 16384 or 1<<14 (15-th bit starting from 1)
M14 =  0x3fff # 16383 or (1<<14)-1 - 14-bit mask

# epixhr data gainbit and mask
B15 = 0o100000 # 32768 or 1<<15 (16-th bit starting from 1)
M15 =  0x7fff  # 32767 or (1<<15)-1 - 15-bit mask


class Storage:
    def __init__(self):
        self.arr1 = None
        self.gfac = None
        self.mask = None
        self.dcfg = None
        self.counter = -1

dic_store = {} # {det.name:Storage()} in stead of singleton


def config_object_det(det, detname=None):
    """Returns [dict]={<seg-index>:<cob>} of configuration objects for detector with optional name.
    """
    _detname = det.raw._det_name if detname is None else detname
    for config in det._configs:
        if not _detname in config.__dict__:
            logger.debug('Skipping config {:}'.format(config.__dict__))
            continue
        return getattr(config,_detname)
    return None


def cbits_config_epix10ka(cob, shape=(352, 384)):
    """Creates array of the segment control bits for epix10ka shape=(352, 384)
    from cob=det.raw._seg_configs()[<seg-ind>].config object.
    Returns per panel 4-bit pixel config array with bit assignment]
          0001 = 1<<0 = 1 - T test bit
          0010 = 1<<1 = 2 - M mask bit
          0100 = 1<<2 = 4 - g  gain bit
          1000 = 1<<3 = 8 - ga gain bit
          # add trbit
          010000 = 1<<4 = 16 - trbit

    Parameters
    ----------
    cob : container.Container object
        segment configuration object det.raw._seg_configs()[<seg-ind>].config
        Contains:
        cob.asicPixelConfig: shape:(4, 178, 192) size:136704 dtype:uint8 [12 12 12 12 12...]
        cob.trbit: [1 1 1 1]

    Returns
    -------
    xxxx: np.array, dtype:uint8, ndim=2, shape=(352, 384)
    """
    trbits = cob.trbit # [1 1 1 1]
    pca = cob.asicPixelConfig # [:,:176,:] - fixed in daq # shape:(4, 176, 192) size:135168 dtype:uint8 [8 8 8 8 8...]
    logger.debug(info_ndarr(cob.asicPixelConfig, 'trbits: %s asicPixelConfig:'%str(trbits)))
    rowsh, colsh = int(shape[0]/2), int(shape[1]/2) # should be 176, 192 for epix10ka

    #t0_sec = time()

    # begin to create array of control bits
    # Origin of ASICs in bottom-right corner, so
    # stack them in upside-down matrix and rotete it by 180 deg.

    cbits = np.flipud(np.fliplr(np.vstack((np.hstack((pca[2],pca[1])),
                                           np.hstack((pca[3],pca[0])))))) # 0.000090 sec

    #cbits = np.bitwise_and(cbits,12) # 0o14 (bin:1100) # 0.000202 sec
    np.bitwise_and(cbits,12,out=cbits) # 0o14 (bin:1100) # 0.000135 sec

    #logger.debug('TIME for cbits composition = %.6f sec' % (time()-t0_sec))
    #logger.debug(info_ndarr(cbits,'cbits:'))
    #exit('TEST EXIT')

    if all(trbits): cbits = np.bitwise_or(cbits, B04) # add trbit for all pixels (352, 384)
    elif not any(trbits): return cbits
    else: # set trbit per ASIC
        if trbits[2]: np.bitwise_or(cbits[:rowsh,:colsh], B04, out=cbits[:rowsh,:colsh])
        if trbits[3]: np.bitwise_or(cbits[rowsh:,:colsh], B04, out=cbits[rowsh:,:colsh])
        if trbits[0]: np.bitwise_or(cbits[rowsh:,colsh:], B04, out=cbits[rowsh:,colsh:])
        if trbits[1]: np.bitwise_or(cbits[:rowsh,colsh:], B04, out=cbits[:rowsh,colsh:]) #0.000189 sec
    return cbits


def cbits_config_epixhr2x2(cob, shape=(288, 384)):
    """Creates array of the segment control bits for epixhr2x2 shape=(288, 384)
    from cob=det.raw._seg_configs()[<seg-ind>].config object.
    Returns per panel 4-bit pixel config array with bit assignment]
          0001 = 1<<0 = 1 - T test bit
          0010 = 1<<1 = 2 - M mask bit
          0100 = 1<<2 = 4 - g  gain bit
          1000 = 1<<3 = 8 - ga gain bit
          # add trbit
          010000 = 1<<4 = 16 - trbit

    Parameters
    ----------
    cob : container.Container object
        segment configuration object det.raw._seg_configs()[<seg-ind>].config
        Contains:
        cob.asicPixelConfig shape:(110592,) size:110592 dtype:uint8 [0 0 0 0 0...]
        cob.trbit: [1 1 1 1]

    ASIC map of epixhr2x2 (Matt)
      A1 | A3
     ----+----
      A0 | A2

    Returns
    -------
    xxxx: np.array, dtype:uint8, ndim=2, shape=(288, 384)
    """
    #t0_sec = time()
    trbits = cob.trbit # [1 1 1 1]
    pca = cob.asicPixelConfig # shape:(110592,)
    rowsh, colsh = int(shape[0]/2), int(shape[1]/2) # should be 144, 192 for epixhr2x2
    logger.debug(info_ndarr(cob.asicPixelConfig, 'shape: %s trbits: %s asicPixelConfig:'%(str(shape), str(trbits))))

    cbits = np.bitwise_and(pca,12,out=None) # copy and mask non-essential bits 0o14 (bin:1100)
    cbits.shape = shape

    #logger.info('TIME1 in cbits_config_epixhr2x2 = %.6f sec' % (time()-t0_sec)) # 0.000206 sec

    if all(trbits): cbits = np.bitwise_or(cbits, B04) # add trbit for all pixels (288, 384)
    elif not any(trbits): return cbits
    else: # set trbit per ASIC
        if trbits[1]: np.bitwise_or(cbits[:rowsh,:colsh], B04, out=cbits[:rowsh,:colsh])
        if trbits[0]: np.bitwise_or(cbits[rowsh:,:colsh], B04, out=cbits[rowsh:,:colsh])
        if trbits[3]: np.bitwise_or(cbits[:rowsh,colsh:], B04, out=cbits[:rowsh,colsh:])
        if trbits[2]: np.bitwise_or(cbits[rowsh:,colsh:], B04, out=cbits[rowsh:,colsh:])

    #logger.info('TIME2 in cbits_config_epixhr2x2 = %.6f sec' % (time()-t0_sec))
    return cbits


def cbits_config_and_data_detector(det_raw, evt=None):
    """Returns array of control bits shape=(<number-of-segments>, 352(or 288), 384)
    from any config object and data array.

    get 5-bit pixel config array with bit assignments
      0001 = 1<<0 = 1 - T test bit
      0010 = 1<<1 = 2 - M mask bit
      0100 = 1<<2 = 4 - g  gain bit
      1000 = 1<<3 = 8 - ga gain bit
    010000 = 1<<4 = 16 - trbit 1/0 for H/M
    add data bit
    100000 = 1<<5 = 32 - data bit 14/15 for epix10ka/epixhr2x2 panel
    """
    data = det_raw.raw(evt)
    cbits = det_raw._cbits_config_detector()
    #logger.info(info_ndarr(cbits, 'cbits', first=0, last=5))
    if cbits is None: return None

    if data is not None:
        #logger.debug(info_ndarr(data, 'data', first, last))
        # get array of data bit 15 and add it as a bit 5 to cbits
        datagainbit = np.bitwise_and(data, det_raw._data_gain_bit)
        databit05 = np.right_shift(datagainbit, det_raw._gain_bit_shift) # 0o100000 -> 0o40
        np.bitwise_or(cbits, databit05, out=cbits) # 109us

    return cbits


def gain_maps_epix10ka_any(det_raw, evt=None):
    """Returns maps of gain groups shape=(<number-of-segments>, <2-d-panel-shape>)
       works for both epix10ka (352, 384) and epixhr2x2 (288, 384)

      cbits - pixel control bit array

        data bit 14 is moved here 1/0 for H,M/L
       / trbit  1/0 for H/M
      V / bit3  1/0 for F/A
       V / bit2 1/0 for H,M/L
        V / M   mask
         V / T  test       gain range index
          V /             /  in calib files
           V             V
      x111xx =28 -  FH_H 0
      x011xx =12 -  FM_M 1
      xx10xx = 8 -  FL_L 2
      0100xx =16 - AHL_H 3
      0000xx = 0 - AML_M 4
      1100xx =48 - AHL_L 5
      1000xx =32 - AML_L 6
      ---
      111100 =60 - cbitsM60 - mask
      011100 =28 - cbitsM28 - mask
      001100 =12 - cbitsM12 - mask
    """

    cbits = det_raw._cbits_config_and_data_detector(evt)
    if cbits is None: return None

    cbitsM60 = cbits & 60 # control bits masked by configuration 3-bit-mask
    cbitsM28 = cbits & 28 # control bits masked by configuration 3-bit-mask
    cbitsM12 = cbits & 12 # control bits masked by configuration 2-bit-mask
    #logger.debug(info_ndarr(cbitsMCB, 'cbitsMCB', first, last))

    #return gr0, gr1, gr2, gr3, gr4, gr5, gr6 # per-pixel bool for 7 gain ranges
    return (cbitsM28 == 28),\
           (cbitsM28 == 12),\
           (cbitsM12 ==  8),\
           (cbitsM60 == 16),\
           (cbitsM60 ==  0),\
           (cbitsM60 == 48),\
           (cbitsM60 == 32)


def info_gain_mode_arrays(gmaps, first=0, last=5):
    """ gr0, gr1, gr2, gr3, gr4, gr5, gr6 = gmaps
    """
    recs = [info_ndarr(gr, 'gr%d'%i, first, last) for i,gr in enumerate(gmaps)]
    return 'gain range arrays:\n  %s' % ('  %s\n'.join(recs))


def pixel_gain_mode_statistics(gmaps):
    """returns statistics of pixels in defferent gain modes in gain maps
       gr0, gr1, gr2, gr3, gr4, gr5, gr6 = gmaps
    """
    arr1 = np.ones_like(gmaps[0], dtype=np.int32)
    return [np.sum(np.select((gr,), (arr1,), default=0)) for gr in gmaps]


def info_pixel_gain_mode_statistics(gmaps):
    """returns (str) with statistics of pixels in defferent gain modes in gain maps
    """
    grp_stat = pixel_gain_mode_statistics(gmaps)
    return ', '.join(['%7d' % npix for npix in grp_stat])


def info_pixel_gain_mode_statistics_for_raw(det_raw, evt=None, msg='pixel gain mode statistics: '):
    """DOES ANYONE USE IT?
       returns (str) with statistics of pixels in defferent gain modes in raw data
    """
    gmaps = gain_maps_epix10ka_any(det_raw, evt)
    if gmaps is None: return None
    return '%s%s' % (msg, info_pixel_gain_mode_statistics(gmaps))


def pixel_gain_mode_fractions(det_raw, evt=None):
    """returns fraction of pixels in defferent gain modes in gain maps
    """
    gmaps = gain_maps_epix10ka_any(det_raw, evt)
    if gmaps is None: return None
    pix_stat = pixel_gain_mode_statistics(gmaps)
    f = 1.0/gmaps[0].size
    return [npix*f for npix in pix_stat]


def info_pixel_gain_mode_for_fractions(grp_prob, msg='pixel gain mode fractions: '):
    return '%s%s' % (msg, ', '.join(['%.5f'%p for p in grp_prob]))


def info_pixel_gain_mode_fractions(det_raw, evt=None, msg='pixel gain mode fractions: '):
    """returns (str) with fraction of pixels in defferent gain modes in gain maps
    """
    grp_prob = pixel_gain_mode_fractions(det_raw, evt)
    return info_pixel_gain_mode_for_fractions(grp_prob, msg=msg)


def gain_mode_index_from_fractions(grp_prob):
    """Returns int gain mode index or None from list of gain group fractions."""
    return next((i for i,p in enumerate(grp_prob) if p>0.5), None)


def find_gain_mode_index(det_raw, evt=None):
    """Returns int gain mode index or None.
       if data=None: distinguish 5-modes w/o data
    """
    grp_prob = pixel_gain_mode_fractions(det_raw, evt)
    return gain_mode_index_from_fractions(grp_prob)


def gain_mode_name_for_index(ind):
    """Returns str gain mode name for int index in the list GAIN_MODES or None.
    """
    return GAIN_MODES[ind] if ind<len(GAIN_MODES) else None


def find_gain_mode(det_raw, evt=None):
    """Returns str gain mode from the list GAIN_MODES or None.
       if data=None: distinguish 5-modes w/o data
    """
    grp_prob = pixel_gain_mode_fractions(det_raw, evt)
    ind = gain_mode_index_from_fractions(grp_prob)
    if ind is None: return None
    return gain_mode_name_for_index(ind)
    #return GAIN_MODES[ind] if ind<len(grp_prob) else None


def event_constants_for_gmaps(gmaps, cons, default=0):
    """ 6 msec
    Parameters
    ----------
    - gmaps - tuple of 7 boolean maps ndarray(<nsegs>, 352, 384)
    - cons - 4d constants  (7, <nsegs>, 352, 384)
    - default value for constants

    Returns
    -------
    np.ndarray (<nsegs>, 352, 384) - per event constants
    """
    return np.select(gmaps, (cons[0,:], cons[1,:], cons[2,:], cons[3,:],\
                             cons[4,:], cons[5,:], cons[6,:]), default=default)


def event_constants(det_raw, evt, cons, default=0):
    gmaps = gain_maps_epix10ka_any(det_raw, evt) #tuple: 7 x shape:(4, 352, 384)
    if gmaps is None: return None
    return event_constants_for_gmaps(gmaps, cons, default=default)


def event_constants_for_grinds(grinds, cons):
    """ 12  msec
    FOR TEST PURPOSE ONLY - x2 slower than event_constants_for_gmaps

    Parameters
    ----------
    - grinds - ndarray(<nsegs>, 352, 384) array of the gain range indices [0,6]
    - cons - 4d constants  (7, <nsegs>, 352, 384)
    - default value for constants

    Returns
    -------
    np.ndarray (<nsegs>, 352, 384) - per event constants
    """
    #shape0 = grinds.shape
    #grinds.shape = (1,) + tuple(grinds.shape) #(1, 4, 352, 384) # add dimension for take_along_axis
    #nda = np.take_along_axis(cons, grinds, 0)
    #grinds.shape = shape0 # restore original shape
    #return nda

    shapei = grinds.shape #(<nsegs>, 352, 384)
    shapec = cons.shape #(7, <nsegs>, 352, 384)
    cons.shape = (7,grinds.size)
    grinds.shape = (1,grinds.size)
    nda = np.take_along_axis(cons, grinds, 0)
    nda.shape = shapei
    cons.shape = shapec
    grinds.shape = shapei
    return nda


def test_event_constants_for_grinds(det_raw, evt, gfac, peds):
    """factor, pedest = test_event_constants_for_grinds(det_raw, evt, gfac, peds)
       12msec for epixquad ueddaq02 r557
    """
    t0_sec = time()
    grinds = map_gain_range_index(det_raw, evt) #.ravel()
    factor = event_constants_for_grinds(grinds, gfac)
    pedest = event_constants_for_grinds(grinds, peds)
    print('XXX test_event_constants_for_grinds consumed time = %.6f sec' % (time()-t0_sec)) # 12msec for epixquad ueddaq02 r557
    print(info_ndarr(grinds, 'evt grinds'))
    print(info_ndarr(pedest, 'evt pedest'))
    print(info_ndarr(factor, 'evt factor'))
    return factor, pedest


def test_event_constants_for_gmaps(det_raw, evt, gfac, peds):
    """factor, pedest = test_event_constants_for_gmaps(det_raw, evt, gfac, peds)
       6msec for epixquad ueddaq02 r557
    """
    t0_sec = time()
    gmaps = gain_maps_epix10ka_any(det_raw, evt) #tuple: 7 x shape:(4, 352, 384)
    if gmaps is None: return None
    factor = event_constants_for_gmaps(gmaps, gfac, default=1)
    pedest = event_constants_for_gmaps(gmaps, peds, default=0) # 6 msec total versus 5.5 using select directly
    print('XXX test_event_constants_for_gmaps consumed time = %.6f sec' % (time()-t0_sec)) # 6msec for epixquad ueddaq02 r557
    print(info_ndarr(gmaps,  'evt gmaps'))
    print(info_ndarr(pedest, 'evt pedest'))
    print(info_ndarr(factor, 'evt factor'))
    return factor, pedest


def calib_epix10ka_any(det_raw, evt, cmpars=None, **kwa): #cmpars=(7,2,100)):
    """
    Algorithm
    ---------
    - gets constants
    - gets raw data
    - evaluates (code - pedestal - offset)
    - applys common mode correction if turned on
    - apply gain factor

    Parameters
    ----------
    - det_raw (psana.Detector.raw) - Detector.raw object
    - evt (psana.Event)    - Event object
    - cmpars (tuple) - common mode parameters
          = None - use pars from calib directory
          = cmpars=(<alg>, <mode>, <maxcorr>)
            alg is not used
            mode =0-correction is not applied, =1-in rows, =2-in cols-WORKS THE BEST
            i.e: cmpars=(7,0,100) or (7,2,100)
    - **kwa - used here and passed to det_raw.mask_comb
      - nda_raw - substitute for det_raw.raw(evt)
      - mbits - parameter of the det_raw.mask_comb(...)
      - mask - user defined mask passed as optional parameter

    Returns
    -------
      - calibrated epix10ka data
    """

    logger.debug('In calib_epix10ka_any')

    t0_sec_tot = time()

    nda_raw = kwa.get('nda_raw', None)
    raw = det_raw.raw(evt) if nda_raw is None else nda_raw # shape:(352, 384) or suppose to be later (<nsegs>, 352, 384) dtype:uint16
    if raw is None: return None

    _cmpars  = det_raw._common_mode() if cmpars is None else cmpars

    gain = det_raw._gain()      # - 4d gains  (7, <nsegs>, 352, 384)
    peds = det_raw._pedestals() # - 4d pedestals
    if gain is None: return None # gain = np.ones_like(peds)  # - 4d gains
    if peds is None: return None # peds = np.zeros_like(peds) # - 4d gains

    store = dic_store.get(det_raw._det_name, None)

    if store is None:

        logger.info('create new store for %s' % det_raw._det_name)
        store = dic_store[det_raw._det_name] = Storage()

        # do ONCE this initialization
        logger.debug(info_ndarr(raw,  '\n  raw ')\
                    +info_ndarr(gain, '\n  gain')\
                    +info_ndarr(peds, '\n  peds'))

        store.gfac = divide_protected(np.ones_like(gain), gain)
        store.arr1 = np.ones_like(raw, dtype=np.int8)

        logger.debug(info_ndarr(store.gfac,  '\n  gfac '))

        # 'FH','FM','FL','AHL-H','AML-M','AHL-L','AML-L'
        #store.gf4 = np.ones_like(raw, dtype=np.int32) * 0.25 # 0.3333 # M - perefierial
        #store.gf6 = np.ones_like(raw, dtype=np.int32) * 1    # L - center

    gfac = store.gfac

    #if store.dcfg is None: store.dcfg = det_raw._config_object() #config_object_det_raw(det_raw)

    gmaps = gain_maps_epix10ka_any(det_raw, evt) #tuple: 7 x shape:(4, 352, 384)
    if gmaps is None: return None

    factor = np.select(gmaps,\
                       (gfac[0,:], gfac[1,:], gfac[2,:], gfac[3,:],\
                        gfac[4,:], gfac[5,:], gfac[6,:]), default=1) # 2msec

    pedest = np.select(gmaps,\
                       (peds[0,:], peds[1,:], peds[2,:], peds[3,:],\
                        peds[4,:], peds[5,:], peds[6,:]), default=0)

    #factor, pedest = test_event_constants_for_gmaps(det_raw, evt, gfac, peds) # 6msec
    #factor, pedest = test_event_constants_for_grinds(det_raw, evt, gfac, peds) # 12msec

    store.counter += 1
    if not store.counter%100:
        logger.debug(info_gain_mode_arrays(gmaps))
        logger.debug(info_pixel_gain_mode_statistics(gmaps))

    logger.debug('TOTAL consumed time (sec = %.6f' % (time()-t0_sec_tot))

    arrf = np.array(raw & det_raw._data_bit_mask, dtype=np.float32) - pedest

    logger.debug('common-mode correction pars cmp: %s' % str(_cmpars))

    if store.mask is None:
        mbits = kwa.pop('mbits',1) # 1-mask from status, etc.
        mask = det_raw._mask_comb(mbits=mbits, **kwa) if mbits > 0 else None
        mask_opt = kwa.get('mask',None) # mask optional parameter in det_raw.calib(...,mask=...)
        store.mask = mask if mask_opt is None else mask_opt if mask is None else merge_masks(mask,mask_opt)

    mask = store.mask if store.mask is not None else np.ones_like(raw, dtype=DTYPE_MASK)

    if _cmpars is not None:
      alg, mode, cormax = int(_cmpars[0]), int(_cmpars[1]), _cmpars[2]
      npixmin = _cmpars[3] if len(_cmpars)>3 else 10
      if mode>0:
        t0_sec_cm = time()
        arr1 = store.arr1 # np.ones_like(mask, dtype=np.uint8)
        gr0, gr1, gr2, gr3, gr4, gr5, gr6 = gmaps
        grhm = np.select((gr0,  gr1,  gr3,  gr4), (arr1, arr1, arr1, arr1), default=0) if alg==7 else arr1
        gmask = np.bitwise_and(grhm, mask) if mask is not None else grhm
        #logger.debug(info_ndarr(arr1, '\n  arr1'))
        #logger.debug(info_ndarr(grhm, 'XXXX grhm'))
        #logger.debug(info_ndarr(gmask, 'XXXX gmask'))
        #logger.debug('common-mode mask massaging (sec) = %.6f' % (time()-t2_sec_cm)) # 5msec
        logger.debug(info_ndarr(gmask, 'gmask')\
                     + '\n  per panel statistics of cm-corrected pixels: %s' % str(np.sum(gmask, axis=(1,2), dtype=np.uint32)))

        #sh = (nsegs, 352, 384)
        hrows = 176 # int(352/2)
        for s in range(arrf.shape[0]):

          if mode & 4: # in banks: (352/2,384/8)=(176,48) pixels
            common_mode_2d_hsplit_nbanks(arrf[s,:hrows,:], mask=gmask[s,:hrows,:], nbanks=8, cormax=cormax, npix_min=npixmin)
            common_mode_2d_hsplit_nbanks(arrf[s,hrows:,:], mask=gmask[s,hrows:,:], nbanks=8, cormax=cormax, npix_min=npixmin)

          if mode & 1: # in rows per bank: 384/8 = 48 pixels # 190ms
            common_mode_rows_hsplit_nbanks(arrf[s,], mask=gmask[s,], nbanks=8, cormax=cormax, npix_min=npixmin)

          if mode & 2: # in cols per bank: 352/2 = 176 pixels # 150ms
            common_mode_cols(arrf[s,:hrows,:], mask=gmask[s,:hrows,:], cormax=cormax, npix_min=npixmin)
            common_mode_cols(arrf[s,hrows:,:], mask=gmask[s,hrows:,:], cormax=cormax, npix_min=npixmin)

        logger.debug('TIME common-mode correction = %.6f sec for cmp=%s' % (time()-t0_sec_cm, str(_cmpars)))

    return arrf * factor if mask is None else arrf * factor * mask # gain correction


def map_gain_range_index(det_raw, evt, **kwa):
    """Returns array of epix10ka per pixel gain range indices [0:6] shaped as raw (<nsegs>, 352, 384) dtype:uint16
    """
    nda_raw = kwa.get('nda_raw', None)
    raw = det_raw.raw(evt) if nda_raw is None else nda_raw
    if raw is None: return None

    gmaps = gain_maps_epix10ka_any(det_raw, evt)
    if gmaps is None: return None
    #gr0, gr1, gr2, gr3, gr4, gr5, gr6 = gmaps
    return np.select(gmaps, (0, 1, 2, 3, 4, 5, 6), default=10)#.astype(np.uint16) # int64 -> uint16


calib_epix10ka = calib_epix10ka_any

# EOF

