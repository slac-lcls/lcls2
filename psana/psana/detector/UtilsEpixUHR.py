
"""
:py:class:`UtilsEpixUHR` contains utilities for epixuhr
=======================================================

Usage::
    import psana.detector.UtilsEpixUHR as ueu

EPIXUHR (my): https://confluence.slac.stanford.edu/x/JCWDHw
ePixUHR+35kHz+-+Pixel+configuration+and+gain+settings Lorenzo Rota"
  Lorenzo Rota https://confluence.slac.stanford.edu/x/JxOyGQ

@author Mikhail Diubrovin
Created on 2025-09-10
"""
#import os
import sys
import numpy as np
from time import time
import psana.detector.NDArrUtils as ndu
import psana.detector.Utils as ut # info_dict, is_true, is_none
import psana.detector.UtilsEpix10ka as ue10ka
info_gain_mode_arrays, pixel_gain_mode_statistics, info_pixel_gain_mode_statistics =\
    ue10ka.info_gain_mode_arrays, ue10ka.pixel_gain_mode_statistics, ue10ka.info_pixel_gain_mode_statistics

cond_msg = ue10ka.cond_msg
info_ndarr = ndu.info_ndarr

# info_gain_mode_arrays(gmaps, first=0, last=5), pixel_gain_mode_statistics(gmaps), info_pixel_gain_mode_statistics(gmaps)
import logging
logger = logging.getLogger(__name__)

SEGMENT_SHAPE = (336, 576)
#ASIC_SHAPE    = (168, 192)

GAIN_MODES = ('FHG', 'FMG', 'FLG1', 'FLG2', 'AHLG1', 'AHLG2', 'AMLG1', 'AMLG2')
GAINS = (1.51, 0.54, 0.0172, 0.0086) # mV/keV gains for FHG, FMG, FLG1, FLG2

# epix10ka data gainbit and mask
B14 = 0o40000 # 16384 or 1<<14 (15-th bit starting from 1)
M14 = 0x3fff  # 16383 or (1<<14)-1 - 14-bit mask

# epixuhr data gainbit and mask
B15 = 0o100000 # 32768 or 1<<15 (16-th bit starting from 1)
M15 = 0x7fff   # 32767 or (1<<15)-1 - 15-bit maskdef gain_bitword(dettype):

#def gain_bitword(dettype):
#    return {'epix10ka':B14, 'epixhr':B15, 'epixhr2x2':B15, 'epixhremu':B15}.get(dettype, None)

#def data_bitword(dettype):
#    return {'epix10ka':M14, 'epixhr':M14, 'epixhr2x2':M14, 'epixhremu':M14}.get(dettype, None)

def cbits_epixuhr(det):
    """det (run.Detector) or type(det) = <class 'psana.psexp.run.Run.Detector.<locals>.Container'>
       NOW:  returns raw shaped (<number-of-ASICs>, <2-d-ASIC-shape>) of control bits config.gainCSVAsic
       TO BE: returns raw shaped (<number-of-segments>, <2-d-panel-shape>) of control bits config.gainCSVAsic
    """
    cfgs = det.config._seg_configs()
    if ut.is_none(cfgs, 'det.config._seg_configs() is None', logger.debug): return None
    return np.vstack([cfgs[segnum].config.gainCSVAsic for segnum in det.raw._segment_numbers])

def gains_epixuhr(det):
    """det (run.Detector)
       NOW:  returns (<number-of-ASICs>, <6-gains-of-ASIC-per-segment>) per-ASIC gains of config.gainAsic
       TO BE: ????
    """
    cfgs = det.config._seg_configs()
    if ut.is_none(cfgs, 'det.config._seg_configs() is None', logger.debug): return None
    return np.vstack([cfgs[segnum].config.gainAsic for segnum in det.raw._segment_numbers])

def gain_maps_epixuhr(cbits):
    """The shape of output arrays is identiacl to input array cbits.
       return gr0, gr1, gr2, gr3, gr4, gr5, gr6, gr7 # per-pixel bool for 8 gain ranges
       returns maps of gain groups shape=(<number-of-segments>, <2-d-panel-shape>)
       works for both epixuhr 3x2: (336, 576)

       cbits - control bit array

       bit 6: data bit 15 is added here to distinguish gain modes in configuration
      / bit 5: LG_sel      low gain selector: 0-LG1, 1-LG2
     V / bit 4: g_auto     Controls auto-gain, see below
      V / bit 3: MG        0-High-gain, 1-Medium-gain
       V / bit 2: inj_en   1-enables injection in each pixel
        V / bit 1: mask    masks the pixel = CSA is always reset.
         V / bit 0: g-sel  controls auto-gain
          V /
           V
                              gain range index in calib files
                             /
                            V

      100000 = 32    FHG    0 injection OFF
      101x00 = 40    FMG    1
      000x01 =  1    FLG1   2
      100x01 = 33    FLG2   3
      010x00 = 16    AHLG1  4
      110x00 = 48    AHLG2  5
      011x00 = 24    AMLG1  6
      111x00 = 56    AMLG2  7
     1010x00 = 16+64 AHLG1  use 2
     1110x00 = 48+64 AHLG2  use 3
     1011x00 = 24+64 AMLG1  use 2
     1111x00 = 56+64 AMLG2  use 3
      xxx0xx       injection OFF
      xxx1xx       injection ON
      xxxx1x = 2   mask pixel
      x0xxxx       auto-gain OFF
      x1xxxx       auto-gain ON
      111111 =63   mask of all except data bit
     1000000 =64   data bit for gain switching to low
    """

    if ut.is_none(cbits, 'cbits is None', logger.debug): return None

    cbitsM = cbits & 0o73 # 59 = 63 - 4 cbits with masked injection bit 0o4
    cbitsF = cbitsM & 0o63 # for fixed gain modes gain bit does not matter
    return\
          (cbitsF == 32),\
          (cbitsF == 40),\
          (cbitsF ==  1),\
          (cbitsF == 33),\
          (cbitsM == 16),\
          (cbitsM == 48),\
          (cbitsM == 24),\
          (cbitsM == 56),\
          (cbitsM == 16+64),\
          (cbitsM == 48+64),\
          (cbitsM == 24+64),\
          (cbitsM == 56+64)


def event_constants_for_gmaps(gmaps, cons, default=0, cmt=''):
    """ 6 msec
    Parameters
    ----------
    - gmaps - tuple of 7 boolean maps ndarray(<nsegs>, 336, 576)
    - cons - 4d constants  (7, <nsegs>, 336, 576)
    - default value for constants

    Returns
    -------
    np.ndarray (<nsegs>, 336, 576) - per event constants
    """
    #assert cons is not None
    if cond_msg(gmaps is None, msg=cmt+'gmaps is None', output_meth=logger.debug):
        return None
    if cond_msg(cons is None, msg=cmt+'cons is None', output_meth=logger.warning):
        return None
    return np.select(gmaps, (cons[0,:], cons[1,:], cons[2,:], cons[3,:],\
                             cons[4,:], cons[5,:], cons[6,:], cons[7,:],\
                             cons[2,:], cons[3,:], cons[2,:], cons[3,:]), default=default)


def reshape_6x32256_to_6x168x192(a, shape_out=(6, 168, 192)):
    """returns array shaped as (2*3, 168, 192) from raw shape (6, 32256)"""
    a.shape = shape_out # (2,3,) + shape_asic
    return a


def stack_2x3_asics(a):
    """stacks asic from input array of shape (6, 168, 192) into 2d segment array,
       returns array shaped as (336, 576)=(2*168, 3*192)
       Dawood's numeration from
       https://confluence.slac.stanford.edu/spaces/ppareg/pages/655311578/ASIC+layout+and+Carrier+orientation

               *|       *|       * <- (0,0) pixel of (168, 192) == (rows, cols)
           A0   |   A1   |   A2
        --------+--------+--------
           A3   |   A4   |   A5
        *       |*       |*
    """
    return np.vstack((np.hstack((np.fliplr(a[0,:]), np.fliplr(a[1,:]), np.fliplr(a[2,:]))),\
                      np.hstack((np.flipud(a[3,:]), np.flipud(a[4,:]), np.flipud(a[5,:])))))
    #return np.vstack((np.hstack((rot180(a[1,:]), rot180(a[3,:]), rot180(a[5,:])))),\
    #                  np.hstack((       a[0,:],         a[2,:],         a[4,:]))))
#def rot180(arr2d):
#    return np.flipud(np.fliplr(arr2d))


def cbits_config_segment(cob):
    """
       returns segment gain control bits shape=(336, 576), stacked from 6 ASICs (6, 32256) = (6, 168, 192)
       cob=det.raw._seg_configs()[<seg-ind>].config - segment configuration object, where self=det.raw
    """
    logger.debug('XXX dir(cob): %s' % str(dir(cob))) #'gainAsic', 'gainCSVAsic']
    gasic = cob.gainAsic            # [56 56 56 56 56 56]
    cbits = cob.gainCSVAsic.copy()  # shape:(6, 32256)
    logger.debug('  XXX cob.gainAsic: %s' % str(gasic))
    logger.debug(info_ndarr(cbits, '  XXX cob.gainCSVAsic', last=10))
    cbits.shape = (6, 168, 192) # reshape_6x32256_to_6x168x192(cbits)
    for i, g in enumerate(gasic):
        if g > 0: cbits[i,:] = g # substitute code from cob.gainAsi if not 0
    cbits = stack_2x3_asics(cbits) # (336, 576)
    logger.info(info_ndarr(cbits, 'segment cbits', last=10))
    return cbits
   # return eb.cbits_config_epix10ka(cob, shape=(336, 576)) # in epix10ka.py


def bit_opers_2x3(a):
    """bit massaging of panel data
       The ePixUHR3x2, when not using "gain expansion" transmits the data as 12
       bits, in a 16 bit integer. The layout of this data is:

                           G D D D D D D D D D D D U U U U
                           | \___________________/ \_____/
                          /            |              |
                     Gain bit   11 bits of data  Unused bits

       Given the above representation, the data is not "packed" in the traditional
       sense. For space saving, the DAQ WILL pack the data, removing the unused bits.
    """
    gbit = a & B15                 # save array with gain bit in 16-th position
    a = np.right_shift(a & M15, 4) # mask 15 lower bits of data and move them 4 bits right
    a = np.bitwise_or(a, gbit)     # set the gain bit
    return a


def raw_v01(det_raw, evt, sh_seg=(336,576)):
    """returns raw for all segments shaped as (<number of segments>, 336, 576)
       assembled from per panel 6-ASIC arrays (6, 32256)=(6, 168, 192)
       TBD: this operation should be done in FPGA
    """
    if cond_msg(evt is None, msg='evt is None - return None', output_meth=logger.warning):
        return None
    segs = det_raw._segments(evt) # {0: <psana.container.Container object at 0x7f9cd51e0bd0>}
    if segs is None:
        return None
    segnums = det_raw._segment_numbers # for now [0,]
    maxsegnum = max(segnums)
    out = np.zeros((maxsegnum+1,)+sh_seg, dtype=np.uint16)
    for iseg, nseg in enumerate(segnums):
        raw_asics = segs[iseg].raw # shape:(6, 32256)
        arr2 = bit_opers_2x3(raw_asics)
        asics = reshape_6x32256_to_6x168x192(arr2) # (6, 168, 192)
        arrseg = stack_2x3_asics(asics) # (336, 576)
        out[nseg,:] = arrseg # save panel in the output array
    return out


def image_v01(det_raw,  evt, **kwargs):
    """returns raw[0,:] 2-d temporary image for a single panel raw data (1, 336, 576)"""
    if cond_msg(evt is None, msg='evt is None - return None', output_meth=logger.warning):
        return None
    det_raw._counter_image += 1
    if det_raw._counter_image < 3:
        logger.warning('TBD TEMPORARY det.raw.image returns 0-th panel: det.raw.image(evt) = det.raw.raw(evt)[0,:]')
    raw = det_raw.raw(evt)
    if raw is None:
        return None
    return raw[0,:]


def gain_default(nsegs=2, fgains=GAINS, shape_seg=SEGMENT_SHAPE):
    """returns array of default gain constants of shape (8, nsegs, <shape_seg>)"""
    import psana.pscalib.calib.CalibConstants as CC
    dtype_gain = CC.dic_calib_type_to_dtype[CC.PIXEL_GAIN] # np.float32
    a = np.empty((8, nsegs) + shape_seg, dtype=dtype_gain)
    for igm, ifg in zip((0,1,2,3,4,5,6,7), (0,1,2,3,0,0,1,1)):
        a[igm,:] = fgains[ifg]
    return a


class Storage_epixuhr_v01():
    def __init__(self, det_raw, **kwa):
        """Holds constants for 2-indices of the gain switching data bit 15.
        Parameters
        ----------
        - det_raw = det.raw
        - **kwa
          -------
          - cmpars (tuple) - common mode parameters, e.g. (7,2,100,10)
          - perpix (bool) - if True, preserves peds and gfac arrays shaped per pixel, as (<nsegs>, 336, 576, 7)
        """
        #Storage.__init__(self, det_raw, **kwa)

        self.peds = None
        self.gfac = None
        self.mask = None
        self.counter = -1

        cmpars = kwa.get('cmpars', None)
        perpix = kwa.get('perpix', False)

        gain = det_raw._gain()      # - 4d gains  (8, <nsegs>, 336, 576)
        peds = det_raw._pedestals() # - 4d pedestals
        if cond_msg(gain is None, msg='gain is None - use default', output_meth=logger.warning):
            gain = gain_default(nsegs=peds.shape[1])
        logger.info('Storage_epixuhr_v01'\
             +info_ndarr(peds, '\n  peds')\
             +info_ndarr(gain, '\n  gain'))

        cbits_hm = cbits = det_raw._cbits_config_detector() # full detector shape (<nsegs>, 336, 576)
        cbits_lo = ue10ka.cbits_config_add_bit(cbits, bit=0b1000000) # force adding the 6-th gain bit to the config control bits
        gmaps_hm = gain_maps_epixuhr(cbits_hm) # gr0, gr1, gr2, ..., gr11 boolean maps of shape (<nsegs>, 336, 576)
        gmaps_lo = gain_maps_epixuhr(cbits_lo)

        self.shape_det = tuple(peds.shape)[-3:]
        self.shape_as_daq = det_raw._shape_as_daq()

        # select per/pixel constants for H,M / L gains for gain bit 0/1, respectively
        peds_hm = event_constants_for_gmaps(gmaps_hm, peds, default=0, cmt='peds for HM ') # full detector shape (<nsegs>, 336, 576)
        peds_lo = event_constants_for_gmaps(gmaps_lo, peds, default=0, cmt='peds for LO ')
        gain_hm = event_constants_for_gmaps(gmaps_hm, gain, default=1, cmt='gain for HM ')
        gain_lo = event_constants_for_gmaps(gmaps_lo, gain, default=1, cmt='gain for LO ')

        mask = det_raw._mask(**kwa)
        if mask is None: mask = det_raw._mask_from_status(**kwa)
        if mask is None: mask = np.ones(self.shape_det, dtype=DTYPE_MASK)
        self.mask = mask

        # combine switching gain constants in 4-d arrays
        peds_sw = np.stack((peds_hm, peds_lo)) # 2x (H/M,L) detector shape (2, <nsegs>, 336, 576)
        gain_sw = np.stack((gain_hm, gain_lo))

        #self.arr1 = np.ones(self.shape_det, dtype=np.int8)
        # evaluate gfac = 1/gain and apply mask

        self.gfac = None

        if gain is not None:
          gfac_sw = ndu.divide_protected(np.ones_like(gain_sw), gain_sw)
          gfac_sw[0,:] *= mask
          gfac_sw[1,:] *= mask
          self.gfac = arrNgrToPerPixelCons(gfac_sw) if perpix else gfac_sw

        self.peds = arrNgrToPerPixelCons(peds_sw) if perpix else peds_sw

        self.cmpars = det_raw._common_mode() if cmpars is None else cmpars

        s = 'Storage_v02 constants:'\
          + f'\n  shape_det: {self.shape_det}\n  shape_as_daq: {self.shape_as_daq}'\
          + f'\n  cmpars: {self.cmpars}'\
          + info_ndarr(cbits_hm,  '\n  cbits_hm')\
          + info_ndarr(cbits_lo,  '\n  cbits_lo')\
          + info_ndarr(self.mask, '\n  mask')\
          + info_ndarr(self.peds, '\n  peds')\
          + info_ndarr(self.gfac, '\n  gfac')
        logger.info(s)


def calib_v01(det_raw, evt, **kwa):
    print('UtilsEpixUHR: calib_v01')
    return det_raw.raw(evt)


def calib_v02(det_raw, evt, **kwa):
    """ """
    logger.debug(f'UtilsEpixUHR: calib_v02 kwa: {str(kwa)}')

    nda_raw = kwa.get('nda_raw', None)
    cmpars  = kwa.get('cmpars', None)
    raw = det_raw.raw(evt) if nda_raw is None else nda_raw # shape: (<nsegs>, 336, 576) dtype:uint16
    if cond_msg(raw is None, msg='raw is None', output_meth=logger.info): return None

    store = det_raw._store_ = Storage_epixuhr_v01(det_raw, **kwa) if det_raw._store_ is None else det_raw._store_  #perpix=True
    store.counter += 1

    igr = ue10ka.grindex_array(raw, gbit=det_raw._data_gain_bitnum) # per-pixel array of gain indices 0 or 1

    t0_sec = time()
    pedest = np.select((igr==0, igr==1), (store.peds[0,:], store.peds[1,:]))
    factor = np.select((igr==0, igr==1), (store.gfac[0,:], store.gfac[1,:]))
    logger.debug('np.select for pedest & factor time: %.6f sec' % (time() - t0_sec)\
                 +info_ndarr(factor,  '\n    factor:')\
                 +info_ndarr(pedest,  '\n    pedest:'))

    raw11 = np.bitwise_and(raw, det_raw._data_bit_mask)
    arrf = np.array(raw11, dtype=np.float32)
    if pedest is not None: arrf -= pedest

    #if store.cmpars is not None:
    #    common_mode_epix_multigain_apply(arrf, gmaps, store)

    if cond_msg(factor is None, msg='factor is None - substitute with 1', output_meth=logger.warning): factor = 1

    mask = store.mask
    calib = arrf * factor if mask is None else arrf * factor * mask

    #logger.info(info_ndarr(calib,  'calib:'))
    return calib




# EOF
