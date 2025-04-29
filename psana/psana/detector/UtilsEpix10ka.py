
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

GAIN_MODES    = ['FH','FM','FL','AHL_H','AML_M','AHL_L','AML_L']
GAIN_MODES_IN = ['FH','FM','FL','AHL_H','AML_M']

B04 =    0o20 #    16 or 1<<4   (5-th bit starting from 1)
B05 =    0o40 #    32 or 1<<5   (6-th bit starting from 1)

# epix10ka data gainbit and mask
B14 = 0o40000 # 16384 or 1<<14 (15-th bit starting from 1)
M14 = 0x3fff  # 16383 or (1<<14)-1 - 14-bit mask

# epixhr data gainbit and mask
B15 = 0o100000 # 32768 or 1<<15 (16-th bit starting from 1)
M15 = 0x7fff   # 32767 or (1<<15)-1 - 15-bit mask

def gain_bitshift(dettype):
    return {'epix10ka':9, 'epixhr':10, 'epixhr2x2':10, 'epixhremu':10}.get(dettype, None)

def gain_bitword(dettype):
    return {'epix10ka':B14, 'epixhr':B15, 'epixhr2x2':B15, 'epixhremu':B15}.get(dettype, None)

def data_bitword(dettype):
    """ 2023-10-30 Dionisio: I realized that I wired both MSB (bits 15 and bits 14) to report gain information.
        HR has 14 data bits, gain bit is 15th (counting from 0)
    """
    return {'epix10ka':M14, 'epixhr':M14, 'epixhr2x2':M14, 'epixhremu':M14}.get(dettype, None)


class Storage:
    def __init__(self, det_raw, **kwa):
        """Holds cached calibration parameters for the epix multi-gain getector.

        **kwa
        ------
        cmpars (tuple) - common mode parameters, e.g. (7,2,100,10)
        perpix (bool) - if True, preserves peds and gfac arrays shaped per pixel, as (<nsegs>, 352, 384, 7)

        Parameters
        ----------
        - counter (int) - event counter
        - gain (ndarray (7, <nsegs>, 352, 384)) - gains from calibration constants
        - peds (ndarray (7, <nsegs>, 352, 384)) - pedestals from calibration constants
        - shape_as_daq (tuple) - shape (<nsegs>, 352, 384) from calibration constants
        - mask - (ndarray (<nsegs>, 352, 384)) - mask retreived from calibration constants
        - arr1 - (ndarray (<nsegs>, 352, 384)) - ones with shape_as_daq
        - cmpars (int or tuple) - user defined or from calibration constants if None, 0 - cm correction is turrned off
        """

        #logger.info('create store with cached parameters for %s' % det_raw._det_name)

        self.arr1 = None
        self.peds = None
        self.gfac = None
        self.mask = None
        self.counter = -1

        cmpars = kwa.get('cmpars', None)
        perpix = kwa.get('perpix', False)

        #logger.info('det_raw._calibconst.keys: %s' % str(det_raw._calibconst.keys()))

        gain = det_raw._gain()      # - 4d gains  (7, <nsegs>, 352, 384)
        peds = det_raw._pedestals() # - 4d pedestals

        # 'FH','FM','FL','AHL_H','AML_M','AHL_L','AML_L'
        #self.gf4 = np.ones_like(raw, dtype=np.int32) * 0.25 # 0.3333 # M - perefierial
        #self.gf6 = np.ones_like(raw, dtype=np.int32) * 1    # L - center
        #if self.dcfg is None: self.dcfg = det_raw._config_object() #config_object_det_raw(det_raw)

        #raw = det_raw.raw(evt) # need it for raw shape only...
        #self.arr1 = np.ones_like(raw, dtype=np.int8)
        self.shape_as_daq = det_raw._shape_as_daq()
        self.arr1 = np.ones(self.shape_as_daq, dtype=np.int8)
        gfac = divide_protected(np.ones_like(gain), gain)

        self.mask = det_raw._mask(**kwa)
        if self.mask is None: self.mask = det_raw._mask_from_status(**kwa)
        if self.mask is None: self.mask = np.ones(self.shape_as_daq, dtype=DTYPE_MASK)

        t0_sec = time()
        self.peds = arr7grToPerPixelCons(peds) if perpix else peds
        self.gfac = arr7grToPerPixelCons(gfac) if perpix else gfac
        dt_sec = (time()-t0_sec)*1000

        self.cmpars = det_raw._common_mode() if cmpars is None else cmpars

        logger.info('\n  det_name: %s' % det_raw._det_name\
                    +'\n  peds and gfac reshape time: %.3f msec' % dt_sec\
                    +'\n  shape_as_daq %s' % str(self.shape_as_daq)\
                    +info_ndarr(gain, '\n  gain')\
                    +info_ndarr(self.peds, '\n  peds')\
                    +info_ndarr(self.gfac, '\n  gfac')\
                    +info_ndarr(self.mask, '\n  mask')\
                    +'\n  common-mode correction parameters cmpars: %s' % str(self.cmpars))


def arr7grToPerPixelCons(arr7gr):
    """Converts array shaped as (7, <number-of-segments>, 352, 384), dt=12us"""
    sh = arr7gr.shape
    assert arr7gr.ndim == 4
    assert sh[-2:] == (352, 384)
    ngr, nsegs, rows, cols = sh0 = arr7gr.shape # (7, <number-of-segments>, 352, 384)
    arr7gr.shape = (ngr, nsegs * rows * cols)
    arrperpix = arr7gr.T
    arrperpix.shape = (nsegs, rows, cols, ngr)
    arr7gr.shape = sh0 # preserve shape of input array
    return arrperpix


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
    trbits = cob.trbit # [1 1 1 1] < per ASIC trbit in the panel, consisting off 4 ASICs
    pca = cob.asicPixelConfig # [:,:176,:] - fixed in daq # shape:(4, 176, 192) size:135168 dtype:uint8 [8 8 8 8 8...]
    logger.debug(info_ndarr(cob.asicPixelConfig, 'trbits: %s asicPixelConfig:'%str(trbits)))
    #print(info_ndarr(cob.asicPixelConfig, 'trbits: %s asicPixelConfig:'%str(trbits)))
    rowsh, colsh = int(shape[0]/2), int(shape[1]/2) # should be 176, 192 for epix10ka

    #t0_sec = time()

    # begin to create array of control bits
    # Origin of ASICs in bottom-right corner, so
    # stack them in upside-down matrix and rotete it by 180 deg.

    #cbits = np.flipud(np.fliplr(np.vstack((np.hstack((pca[2],pca[1])),
    #                                       np.hstack((pca[3],pca[0])))))) # 0.000090 sec

    cbits = np.vstack((np.hstack((np.flipud(np.fliplr(pca[2])),
                                  np.flipud(np.fliplr(pca[1])))),
                       np.hstack((pca[3],pca[0]))))

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


def cbits_config_epixhr1x4(cob, shape=(144, 768)):
    """ see cbits_config_epixhr2x2

      ASIC map of epixhr1x4
      A0 | A1 | A2 | A3 ???
    Returns
    -------
    xxxx: np.array, dtype:uint8, ndim=2, shape=(144, 768)
    """
    trbits = cob.trbit # [1 1 1 1]
    pca = cob.asicPixelConfig # shape:(110592,)
    rows, colsa = shape[0], int(shape[1]/4) # should be 144, 192 for epixhr1x4
    logger.debug(info_ndarr(cob.asicPixelConfig, 'shape: %s trbits: %s asicPixelConfig:'%(str(shape), str(trbits))))

    cbits = np.bitwise_and(pca,12,out=None) # copy and mask non-essential bits 0o14 (bin:1100)
    cbits.shape = shape

    if all(trbits): cbits = np.bitwise_or(cbits, B04) # add trbit for all pixels (144, 768)
    elif not any(trbits): return cbits

    else: # set trbit per ASIC
        if trbits[0]: np.bitwise_or(cbits[:,colsa*0:colsa*1], B04, out=cbits[:,colsa*0:colsa*1])
        if trbits[1]: np.bitwise_or(cbits[:,colsa*1:colsa*2], B04, out=cbits[:,colsa*1:colsa*2])
        if trbits[2]: np.bitwise_or(cbits[:,colsa*2:colsa*3], B04, out=cbits[:,colsa*2:colsa*3])
        if trbits[3]: np.bitwise_or(cbits[:,colsa*3:colsa*4], B04, out=cbits[:,colsa*3:colsa*4])

    #logger.info('TIME2 in cbits_config_epixhr2x2 = %.6f sec' % (time()-t0_sec))
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


def cbits_config_and_data_detector_alg(data, cbits, data_gain_bit, gain_bit_shift):
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

    #logger.info(info_ndarr(cbits, 'cbits', first=0, last=5))
    if cbits is None or (isinstance(cbits, list) and cbits[0] is None):
       logger.debug('cbits is None or [None]')
       return None

    if data is not None:
        #logger.debug(info_ndarr(data, 'data', first, last))
        # get array of data bit 15 and add it as a bit 5 to cbits
        datagainbit = np.bitwise_and(data, data_gain_bit)
        databit05 = np.right_shift(datagainbit, gain_bit_shift) # 0o100000 -> 0o40
        return np.bitwise_or(cbits, databit05) # create copy, DO NOT OVERRIDE cbits !!!

    return cbits


def cbits_config_and_data_detector(det_raw, evt=None):
    return cbits_config_and_data_detector_alg(\
             det_raw.raw(evt),\
             det_raw._cbits_config_detector(),\
             det_raw._data_gain_bit,\
             det_raw._gain_bit_shift)


def gain_maps_epix10ka_any_alg(cbits):
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

    if cbits is None:
        logger.debug('cbits is None')
        return None

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


def gain_maps_epix10ka_any(det_raw, evt=None):
    return gain_maps_epix10ka_any_alg(det_raw._cbits_config_and_data_detector(evt))


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
    """returns (str) with statistics of pixels in defferent gain modes in gain maps."""
    grp_stat = pixel_gain_mode_statistics(gmaps)
    return ', '.join(['%7d' % npix for npix in grp_stat])


def info_pixel_gain_mode_statistics_for_raw(det_raw, evt=None, msg='pixel gain mode statistics: '):
    """DOES ANYONE USE IT?
       returns (str) with statistics of pixels in defferent gain modes in raw data
    """
    gmaps = gain_maps_epix10ka_any(det_raw, evt)
    if gmaps is None:
        logger.debug('gmaps is None')
        return None
    return '%s%s' % (msg, info_pixel_gain_mode_statistics(gmaps))


def pixel_gain_mode_fractions(det_raw, evt=None):
    """returns fraction of pixels in defferent gain modes in gain maps."""
    gmaps = gain_maps_epix10ka_any(det_raw, evt)
    if gmaps is None:
        logger.debug('gmaps is None')
        return None
    pix_stat = pixel_gain_mode_statistics(gmaps)
    f = 1.0/gmaps[0].size
    return [npix*f for npix in pix_stat]


def info_pixel_gain_mode_for_fractions(grp_prob, msg='pixel gain mode fractions: '):
    return None if grp_prob is None else '%s%s' % (msg, ', '.join(['%.5f'%p for p in grp_prob]))


def info_pixel_gain_mode_fractions(det_raw, evt=None, msg='pixel gain mode fractions: '):
    """returns (str) with fraction of pixels in defferent gain modes in gain maps."""
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
    """Returns str gain mode name for int index in the list GAIN_MODES or None."""
    return GAIN_MODES[ind] if ind<len(GAIN_MODES) else None


def find_gain_mode(det_raw, evt=None):
    """Returns str gain mode from the list GAIN_MODES or None.
       if data=None: distinguish 5-modes w/o data
    """
    grp_prob = pixel_gain_mode_fractions(det_raw, evt)
    ind = gain_mode_index_from_fractions(grp_prob)
    if ind is None:
        logger.debug('ind is None')
        return None
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
    #assert cons is not None
    if gmaps is None:
        logger.debug('gmaps is None')
        return None
    if cons is None:
        logger.debug('cons is None')
        return None
    return np.select(gmaps, (cons[0,:], cons[1,:], cons[2,:], cons[3,:],\
                             cons[4,:], cons[5,:], cons[6,:]), default=default)


def event_constants(det_raw, evt, cons, default=0):
    gmaps = gain_maps_epix10ka_any(det_raw, evt) #tuple: 7 x shape:(4, 352, 384)
    if gmaps is None:
        logger.debug('gmaps is None')
        return None
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
    if gmaps is None:
        logger.debug('gmaps is None')
        return None
    factor = event_constants_for_gmaps(gmaps, gfac, default=1)
    pedest = event_constants_for_gmaps(gmaps, peds, default=0) # 6 msec total versus 5.5 using select directly
    print('XXX test_event_constants_for_gmaps consumed time = %.6f sec' % (time()-t0_sec)) # 6msec for epixquad ueddaq02 r557
    print(info_ndarr(gmaps,  'evt gmaps'))
    print(info_ndarr(pedest, 'evt pedest'))
    print(info_ndarr(factor, 'evt factor'))
    return factor, pedest


def print_gmaps_info(gmaps):
    logger.debug('%s\n%s' %\
      (info_gain_mode_arrays(gmaps), info_pixel_gain_mode_statistics(gmaps)))

def cond_msg(c, msg='is None', output_meth=logger.debug):
    if c: output_meth(msg)
    return c

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
            i.e: cmpars=(7,0,100) or (7,2,100) or (7,7,100)
    - **kwa - used here and passed to det_raw.mask_comb
      - nda_raw - substitute for det_raw.raw(evt)
      - mbits - parameter of the det_raw.mask_comb(...)
      - mask - user defined mask passed as optional parameter

    Returns
    -------
      - calibrated epix10ka data
    """

    #print('XXXX calib_epix10ka_any kwa:', kwa)

    nda_raw = kwa.get('nda_raw', None)
    raw = det_raw.raw(evt) if nda_raw is None else nda_raw # shape:(352, 384) or suppose to be later (<nsegs>, 352, 384) dtype:uint16
    if cond_msg(raw is None, msg='raw is None'): return None

    gmaps = gain_maps_epix10ka_any(det_raw, evt) #tuple: 7 x shape:(4, 352, 384)
    if cond_msg(gmaps is None, msg='gmaps is None'): return None

    store = det_raw._store_ = Storage(det_raw, cmpars=cmpars, **kwa) if det_raw._store_ is None else det_raw._store_  #perpix=True
    store.counter += 1
    if store.counter < 1: print_gmaps_info(gmaps)

    factor = event_constants_for_gmaps(gmaps, store.gfac, default=1)  # 3d gain factors
    pedest = event_constants_for_gmaps(gmaps, store.peds, default=0)  # 3d pedestals

    store.counter += 1
    if not store.counter%100: print_gmaps_info(gmaps)
    arrf = np.array(raw & det_raw._data_bit_mask, dtype=np.float32)
    if pedest is not None: arrf -= pedest

    if store.cmpars is not None:
        common_mode_epix_multigain_apply(arrf, gmaps, store)

    logger.debug(info_ndarr(arrf,  'arrf:'))

    if cond_msg(factor is None, msg='factor is None - substitute with 1', output_meth=logger.warning): factor = 1

    mask = store.mask
    return arrf * factor if mask is None else arrf * factor * mask # gain correction


def common_mode_epix_multigain_apply(arrf, gmaps, store):
    """Apply common mode correction to arrf."""
    cmpars, mask = store.cmpars, store.mask
    logger.debug('in common_mode_epix_multigain_apply for cmpars=%s' % str(cmpars))
    alg, mode, cormax = int(cmpars[0]), int(cmpars[1]), cmpars[2]
    npixmin = cmpars[3] if len(cmpars)>3 else 10
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

      #sh = (nsegs, 288, 384) # epixhr
      #sh = (nsegs, 352, 384) # epix10ka
      hrows = int(arrf.shape[1]/2) # 176 for epix10ka or 144 for epixhr # int(352/2)
      for s in range(arrf.shape[0]):

        if mode & 4: # in banks: (352/2,384/8)=(176,48) pixels
          common_mode_2d_hsplit_nbanks(arrf[s,:hrows,:], mask=gmask[s,:hrows,:], nbanks=8, cormax=cormax, npix_min=npixmin)
          common_mode_2d_hsplit_nbanks(arrf[s,hrows:,:], mask=gmask[s,hrows:,:], nbanks=8, cormax=cormax, npix_min=npixmin)

        if mode & 1: # in rows per bank: 384/8 = 48 pixels # 190ms
          common_mode_rows_hsplit_nbanks(arrf[s,], mask=gmask[s,], nbanks=8, cormax=cormax, npix_min=npixmin)

        if mode & 2: # in cols per bank: 352/2 = 176 pixels # 150ms
          common_mode_cols(arrf[s,:hrows,:], mask=gmask[s,:hrows,:], cormax=cormax, npix_min=npixmin)
          common_mode_cols(arrf[s,hrows:,:], mask=gmask[s,hrows:,:], cormax=cormax, npix_min=npixmin)

      #logger.debug('TIME common-mode correction = %.6f sec for cmpars=%s' % (time()-t0_sec_cm, str(cmpars)))


def map_gain_range_index_for_gmaps(gmaps, default=10):
    if gmaps is None:
        logger.debug('gmaps is None')
        return None
    #gr0, gr1, gr2, gr3, gr4, gr5, gr6 = gmaps
    return np.select(gmaps, (0, 1, 2, 3, 4, 5, 6), default=default)  # .astype(np.uint16) # int64 -> uint16


def map_gain_range_index(det_raw, evt, **kwa):
    """Returns array of epix10ka per pixel gain range indices [0:6] shaped as raw (<nsegs>, 352, 384) dtype:uint16
    """
    nda_raw = kwa.get('nda_raw', None)
    raw = det_raw.raw(evt) if nda_raw is None else nda_raw
    if raw is None:
        logger.debug('raw is None')
        return None

    gmaps = gain_maps_epix10ka_any(det_raw, evt)
    return map_gain_range_index_for_gmaps(gmaps)


calib_epix10ka = calib_epix10ka_any

# EOF

