"""
:py:class:`UtilsEpix10ka` contains utilities for epix10ka and its composite detectors
=====================================================================================

Usage::
    from psana.detector.UtilsEpix10ka import ...

    t_sec = seconds(ts, epoch_offset_sec=631152000) #Converts LCLS2 timestamp to unix epoch time

    inds = segment_indices_epix10ka_detector(det)
    long_name = fullname_epix10ka_detector(det)
    ids = segment_ids_epix10ka_detector(det)
    o = config_object_epix10ka(det, detname=None)
    o = config_object_epix10ka_raw(det_raw)
    cbits = cbits_config_epix10ka(cob)
    cbits = cbits_config_epix10ka_any(dcfg)
    cbits = cbits_total_epix10ka_any(dcfg, data=None)
    maps = gain_maps_epix10ka_any(dcfg, data=None)
    s = def info_gain_mode_arrays(gmaps, first=0, last=5)
    gmstatist = pixel_gain_mode_statistics(gmaps)
    s = info_pixel_gain_mode_statistics(gmaps)
    s = info_pixel_gain_mode_statistics_for_raw(dcfg, data=None, msg='pixel gain mode statistics: ')
    gmfs = pixel_gain_mode_fractions(dcfg, data=None)
    s = info_pixel_gain_mode_fractions(dcfg, data=None, msg='pixel gain mode fractions: ')
    gmode = find_gain_mode(dcfg, data=None)
    calib = calib_epix10ka_any(det_raw, evt, cmpars=None, **kwa)

This software was developed for the LCLS project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2020-12-03 by Mikhail Dubrovin for LCLS2 from LCLS1 
"""

import os
import numpy as np
from time import time


import logging
logger = logging.getLogger(__name__)

from psana.pyalgos.generic.NDArrUtils import info_ndarr, divide_protected

#----

GAIN_MODES    = ['FH','FM','FL','AHL-H','AML-M','AHL-L','AML-L']
GAIN_MODES_IN = ['FH','FM','FL','AHL-H','AML-M']

B14 = 0o40000 # 16384 or 1<<14 (15-th bit starting from 1)
B04 =    0o20 #    16 or 1<<4   (5-th bit starting from 1)
B05 =    0o40 #    32 or 1<<5   (6-th bit starting from 1)
M14 =  0x3fff # 16383 or (1<<14)-1 - 14-bit mask

#----

class Storage:
    def __init__(self):
        self.arr1 = None
        self.gfac = None
        self.mask = None
        self.dcfg = None
        self.counter = -1

store = Storage() # singleton

#----

def seconds(ts, epoch_offset_sec=631152000) -> float:
    """
    Converts LCLS2 timestamp to unix epoch time.
    The epoch used is 1-Jan-1990 rather than 1970. -Matt

    Receives  ts = orun.timestamp  # 4193682596073796843 relative to 1990-01-01
    Returns unix epoch time in sec # 1607569818.532117 sec

    import datetime
    epoch_offset_sec=(datetime.datetime(1990, 1, 1)-datetime.datetime(1970, 1, 1)) / datetime.timedelta(seconds=1)
    """
    return float(ts>>32) + float(ts&0xffffffff)*1.e-9 + epoch_offset_sec


def segment_indices_epix10ka_detector(det):
    """Returns list det.raw._sorted_segment_ids, e.g. [0, 1, 2, 3]""" 
    return det.raw._sorted_segment_ids


def fullname_epix10ka_detector(det):
    """Returns epix10ka detector full name, e.g. 
       epix_3926196238-0175152897-1157627926-0000000000-0000000000-0000000000-0000000000\
           _3926196238-0174824449-0268435478-0000000000-0000000000-0000000000-0000000000\
           _3926196238-0175552257-3456106518-0000000000-0000000000-0000000000-0000000000\
           _3926196238-0176373505-4043309078-0000000000-0000000000-0000000000-0000000000
    """    
    return det.raw._uniqueid


def segment_ids_epix10ka_detector(det):
    """Returns list of epix10ka detector segment ids, e.g.
    [3926196238-0175152897-1157627926-0000000000-0000000000-0000000000-0000000000,
     3926196238-0174824449-0268435478-0000000000-0000000000-0000000000-0000000000,
     3926196238-0175552257-3456106518-0000000000-0000000000-0000000000-0000000000,
     3926196238-0176373505-4043309078-0000000000-0000000000-0000000000-0000000000]
    """
    return det.raw._uniqueid.split('_')[1:]


def config_object_epix10ka(det, detname=None):
    """Returns [dict]={<seg-index>:<cob>} of configuration objects for detector with optional name.
    """    
    _detname = det.raw._det_name if detname is None else detname
    for config in det._configs:
        if not _detname in config.__dict__:
            logger.debug('Skipping config {:}'.format(config.__dict__))
            continue
        return getattr(config,_detname)
    return None


def config_object_epix10ka_raw(det_raw):
    """Returns [dict]={<seg-index>:<cob>} of configuration objects for det.raw
    """    
    logger.debug('det_raw._seg_configs(): ' + str(det_raw._seg_configs()))
    return det_raw._seg_configs()


def cbits_config_epix10ka(cob):
    """
    Creates array of control bits shape=(352, 384) from det.raw._seg_configs()[<seg-ind>].config object
    get epix10ka per panel 4-bit pixel config array with bit assignment]
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
    pca = cob.asicPixelConfig[:,:176,:] # shape:(4, 176, 192) size:135168 dtype:uint8 [8 8 8 8 8...]
    #logger.debug(info_ndarr(pca, 'trbits: %s asicPixelConfig:'%str(trbits)))

    #t0_sec = time()

    # begin to create array of control bits 
    #nasics, narows, nacols = pca.shape
    #seg_shape = (narows*2, nacols*2) # (352, 384)

    #cbits = np.empty(seg_shape, dtype=np.int16)
    #cbits[176:,192:] = np.flipud(np.fliplr(pca[0]))
    #cbits[:176,192:] = np.flipud(np.fliplr(pca[1]))
    #cbits[:176,:192] = np.flipud(np.fliplr(pca[2]))
    #cbits[176:,:192] = np.flipud(np.fliplr(pca[3])) # 0.000278 sec

    #cbits = np.empty(seg_shape, dtype=np.int16)
    #cbits[176:,192:] = pca[0,::-1,::-1]
    #cbits[:176,192:] = pca[1,::-1,::-1]
    #cbits[:176,:192] = pca[2,::-1,::-1]
    #cbits[176:,:192] = pca[3,::-1,::-1] # 0.000248-264 sec

    # Origin of ASICs in bottom-right corner, so
    # stack them in upside-down matrix and rotete it by 180 deg.
    cbits = np.flipud(np.fliplr(np.vstack((np.hstack((pca[2],pca[1])),
                                           np.hstack((pca[3],pca[0])))))) # 0.000090 sec

    #cbits = np.bitwise_and(cbits,12) # 0o14 (bin:1100) # 0.000202 sec
    np.bitwise_and(cbits,12,out=cbits) # 0o14 (bin:1100) # 0.000135 sec

    #logger.debug('TIME for cbits composition = %.6f sec' % (time()-t0_sec))
    #logger.debug(info_ndarr(cbits,'cbits:'))    
    #exit('TEST EXIT')

    # add trbit
    if all(trbits): cbits = np.bitwise_or(cbits, B04) # for all pixels (352, 384)
    elif not any(trbits): return cbits
    else: # set trbit per ASIC
        if trbits[2]: np.bitwise_or(cbits[:176,:192], B04, out=cbits[:176,:192])
        if trbits[3]: np.bitwise_or(cbits[176:,:192], B04, out=cbits[176:,:192])
        if trbits[0]: np.bitwise_or(cbits[176:,192:], B04, out=cbits[176:,192:])
        if trbits[1]: np.bitwise_or(cbits[:176,192:], B04, out=cbits[:176,192:]) #0.000189 sec
    return cbits


def cbits_config_epix10ka_any(dcfg):
    """Returns array of control bits shape=(<number-of-segments>, 352, 384) from any config object
    """
    #for k,v in dcfg.items():
    #      scob = v.config
    #      logger.debug('YYY dcfg[0].config.trbit: %s' % (k,str(dcfg[0].config.trbit))) # [1 1 1 1]
    #      logger.debug(info_ndarr(scob.asicPixelConfig, '...[%d].config.asicPixelConfig: '%k))
 
    #for k,v in dcfg.items(): 
    #    print('AAAA dir(k)', dir(k))
    #    print('AAAA dir(v)', dir(v))

    lst_cbits = [cbits_config_epix10ka(v.config) for k,v in dcfg.items()]
    cbits = np.stack(tuple(lst_cbits))
    return cbits


def cbits_total_epix10ka_any(dcfg, data=None):
    """Returns array of control bits shape=(<number-of-segments>, 352, 384) 
       from any config object and data array.
    """
    cbits = cbits_config_epix10ka_any(dcfg)
    #logger.debug(info_ndarr(cbits, 'cbits', first, last))
    
    if cbits is None: return None

    #----
    # get 5-bit pixel config array with bit assignments
    #   0001 = 1<<0 = 1 - T test bit
    #   0010 = 1<<1 = 2 - M mask bit
    #   0100 = 1<<2 = 4 - g  gain bit
    #   1000 = 1<<3 = 8 - ga gain bit
    # 010000 = 1<<4 = 16 - trbit 1/0 for H/M
    # add data bit
    # 100000 = 1<<5 = 32 - data bit 14
    #----
    if data is not None:
        #logger.debug(info_ndarr(data, 'data', first, last))
        # get array of data bit 14 and add it as a bit 5 to cbits
        databit14 = np.bitwise_and(data, B14)
        databit05 = np.right_shift(databit14,9) # 040000 -> 040
        np.bitwise_or(cbits, databit05, out=cbits) # 109us
        #cbits[databit14>0] += 040              # 138us

    return cbits


def gain_maps_epix10ka_any(dcfg, data=None):
    """Returns maps of gain groups shape=(<number-of-segments>, 352, 384) 
    """
    cbits = cbits_total_epix10ka_any(dcfg, data)
    if cbits is None: return None

    #----
    # cbits - pixel control bit array
    #----
    #   data bit 14 is moved here 1/0 for H,M/L
    #  / trbit  1/0 for H/M
    # V / bit3  1/0 for F/A
    #  V / bit2 1/0 for H,M/L
    #   V / M   mask
    #    V / T  test       gain range index
    #     V /             /  in calib files
    #      V             V 
    # x111xx =28 -  FH_H 0 
    # x011xx =12 -  FM_M 1 
    # xx10xx = 8 -  FL_L 2
    # 0100xx =16 - AHL_H 3
    # 0000xx = 0 - AML_M 4
    # 1100xx =48 - AHL_L 5
    # 1000xx =32 - AML_L 6
    #------
    # 111100 =60 - cbitsM60 - mask 
    # 011100 =28 - cbitsM28 - mask 
    # 001100 =12 - cbitsM12 - mask 
    #------

    cbitsM60 = cbits & 60 # control bits masked by configuration 3-bit-mask
    cbitsM28 = cbits & 28 # control bits masked by configuration 3-bit-mask
    cbitsM12 = cbits & 12 # control bits masked by configuration 2-bit-mask
    #logger.debug(info_ndarr(cbitsMCB, 'cbitsMCB', first, last))

    gr0 = (cbitsM28 == 28)
    gr1 = (cbitsM28 == 12)
    gr2 = (cbitsM12 ==  8)
    gr3 = (cbitsM60 == 16)
    gr4 = (cbitsM60 ==  0)
    gr5 = (cbitsM60 == 48)
    gr6 = (cbitsM60 == 32)

    #first = 10000; logger.debug(info_gain_mode_arrays((gr0, gr1, gr2, gr3, gr4, gr5, gr6), first, first+5))
        
    return gr0, gr1, gr2, gr3, gr4, gr5, gr6


def info_gain_mode_arrays(gmaps, first=0, last=5):
    gr0, gr1, gr2, gr3, gr4, gr5, gr6 = gmaps
    return 'gain range arrays:\n  %s\n  %s\n  %s\n  %s\n  %s\n  %s\n  %s'%(\
        info_ndarr(gr0, 'gr0', first, last),\
        info_ndarr(gr1, 'gr1', first, last),\
        info_ndarr(gr2, 'gr2', first, last),\
        info_ndarr(gr3, 'gr3', first, last),\
        info_ndarr(gr4, 'gr4', first, last),\
        info_ndarr(gr5, 'gr5', first, last),\
        info_ndarr(gr6, 'gr6', first, last))


def pixel_gain_mode_statistics(gmaps):
    """returns statistics of pixels in defferent gain modes in gain maps
    """
    gr0, gr1, gr2, gr3, gr4, gr5, gr6 = gmaps
    arr1 = np.ones_like(gr0, dtype=np.int32)
    return\
      np.sum(np.select((gr0,), (arr1,), default=0)),\
      np.sum(np.select((gr1,), (arr1,), default=0)),\
      np.sum(np.select((gr2,), (arr1,), default=0)),\
      np.sum(np.select((gr3,), (arr1,), default=0)),\
      np.sum(np.select((gr4,), (arr1,), default=0)),\
      np.sum(np.select((gr5,), (arr1,), default=0)),\
      np.sum(np.select((gr6,), (arr1,), default=0))


def info_pixel_gain_mode_statistics(gmaps):
    """returns (str) with statistics of pixels in defferent gain modes in gain maps
    """
    grp_stat = pixel_gain_mode_statistics(gmaps)
    return ', '.join(['%7d' % npix for npix in grp_stat])


def info_pixel_gain_mode_statistics_for_raw(dcfg, data=None, msg='pixel gain mode statistics: '):
    """returns (str) with statistics of pixels in defferent gain modes in raw data
    """
    gmaps = gain_maps_epix10ka_any(dcfg, data)
    if gmaps is None: return None
    return '%s%s' % (msg, info_pixel_gain_mode_statistics(gmaps))


def pixel_gain_mode_fractions(dcfg, data=None):
    """returns fraction of pixels in defferent gain modes in gain maps
    """
    gmaps = gain_maps_epix10ka_any(dcfg, data)
    if gmaps is None: return None
    pix_stat = pixel_gain_mode_statistics(gmaps)
    f = 1.0/gmaps[0].size
    return [npix*f for npix in pix_stat]


def info_pixel_gain_mode_fractions(dcfg, data=None, msg='pixel gain mode fractions: '):
    """returns (str) with fraction of pixels in defferent gain modes in gain maps
    """
    grp_prob = pixel_gain_mode_fractions(dcfg, data)
    return '%s%s' % (msg, ', '.join(['%.5f'%p for p in grp_prob]))


def find_gain_mode(dcfg, data=None):
    """Returns str gain mode from the list GAIN_MODES or None.
       if data=None: distinguish 5-modes w/o data
    """
    grp_prob = pixel_gain_mode_fractions(dcfg, data)

    ind = next((i for i,p in enumerate(grp_prob) if p>0.5), None)
    if ind is None: return None
    gain_mode = GAIN_MODES[ind] if ind<len(grp_prob) else None 
    #logger.debug('Gain mode %s is selected from %s' % (gain_mode, ', '.join(GAIN_MODES)))

    return gain_mode


def calib_epix10ka_any(det_raw, evt, cmpars=None, **kwa): #cmpars=(7,2,100)):
    """
    Returns calibrated epix10ka data

    - gets constants
    - gets raw data
    - evaluates (code - pedestal - offset)
    - applys common mode correction if turned on
    - apply gain factor

    Parameters

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

    if store.gfac is None: 
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

    if store.dcfg is None: store.dcfg = config_object_epix10ka_raw(det_raw)

    gmaps = gain_maps_epix10ka_any(store.dcfg, raw)
    if gmaps is None: return None
    gr0, gr1, gr2, gr3, gr4, gr5, gr6 = gmaps

    factor = np.select(gmaps,\
                       (gfac[0,:], gfac[1,:], gfac[2,:], gfac[3,:],\
                        gfac[4,:], gfac[5,:], gfac[6,:]), default=1) # 2msec

    pedest = np.select(gmaps,\
                       (peds[0,:], peds[1,:], peds[2,:], peds[3,:],\
                        peds[4,:], peds[5,:], peds[6,:]), default=0)

    store.counter += 1
    if not store.counter%100:
        logger.debug(info_gain_mode_arrays(gmaps))
        logger.debug(info_pixel_gain_mode_statistics(gmaps))

    logger.debug('TOTAL consumed time (sec) = %.6f' % (time()-t0_sec_tot))
    logger.debug(info_ndarr(factor, 'calib_epix10ka factor'))
    logger.debug(info_ndarr(pedest, 'calib_epix10ka pedest'))

    arrf = np.array(raw & M14, dtype=np.float32) - pedest

    logger.debug('common-mode correction pars cmp: %s' % str(_cmpars))

    #if store.mask is None: 
    #    mbits = kwa.pop('mbits',1) # 1-mask from status, etc.
    #    mask = det_raw.mask_comb(evt, mbits, **kwa) if mbits > 0 else None
    #    mask_opt = kwa.get('mask',None) # mask optional parameter in det_raw.calib(...,mask=...)
    #    if mask_opt is not None:
    #       store.mask = mask_opt if mask is None else merge_masks(mask,mask_opt)
    #mask = store.mask        

    mask = np.ones_like(raw, dtype=np.int8)

    if _cmpars is not None:
      mode, cormax = int(_cmpars[1]), _cmpars[2]
      npixmin = _cmpars[3] if len(_cmpars)>3 else 10
      if mode>0:
        t0_sec_cm = time()
        #t2_sec_cm = time()
        arr1 = store.arr1 # np.ones_like(mask, dtype=np.uint8)
        grhm = np.select((gr0,  gr1,  gr3,  gr4), (arr1, arr1, arr1, arr1), default=0)
        gmask = np.bitwise_and(grhm, mask) if mask is not None else grhm
        #logger.debug(info_ndarr(arr1, '\n  arr1'))
        #logger.debug(info_ndarr(grhm, 'XXXX grhm'))
        #logger.debug(info_ndarr(gmask, 'XXXX gmask'))
        #logger.debug('common-mode mask massaging (sec) = %.6f' % (time()-t2_sec_cm)) # 5msec
        logger.debug('per panel statistics of good pixels: %s' % str(np.sum(gmask, axis=(1,2), dtype=np.uint32)))

        #sh = (nsegs, 352, 384)
        hrows = 352/2
        for s in range(arrf.shape[0]):

          if mode & 4: # in banks: (352/2,384/8)=(176,48) pixels
            common_mode_2d_hsplit_nbanks(arrf[s,:hrows,:], mask=gmask[s,:hrows,:], nbanks=8, cormax=cormax, npix_min=npixmin)
            common_mode_2d_hsplit_nbanks(arrf[s,hrows:,:], mask=gmask[s,hrows:,:], nbanks=8, cormax=cormax, npix_min=npixmin)

          if mode & 1: # in rows per bank: 384/8 = 48 pixels # 190ms
            common_mode_rows_hsplit_nbanks(arrf[s,], mask=gmask[s,], nbanks=8, cormax=cormax, npix_min=npixmin)

          if mode & 2: # in cols per bank: 352/2 = 176 pixels # 150ms
            common_mode_cols(arrf[s,:hrows,:], mask=gmask[s,:hrows,:], cormax=cormax, npix_min=npixmin)
            common_mode_cols(arrf[s,hrows:,:], mask=gmask[s,hrows:,:], cormax=cormax, npix_min=npixmin)

        logger.debug('TIME common-mode correction = %.6f sec' % (time()-t0_sec_cm))

    return arrf * factor if mask is None else arrf * factor * mask # gain correction

#--------------------

calib_epix10ka = calib_epix10ka_any


# EOF

