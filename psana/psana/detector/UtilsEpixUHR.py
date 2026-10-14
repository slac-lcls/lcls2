
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
info_gain_mode_arrays, pixel_gain_mode_statistics, info_pixel_gain_mode_statistics,\
    cbits_config_add_bit, arrNgrToPerPixelCons =\
    ue10ka.info_gain_mode_arrays, ue10ka.pixel_gain_mode_statistics,\
    ue10ka.info_pixel_gain_mode_statistics, ue10ka.cbits_config_add_bit,\
    ue10ka.arrNgrToPerPixelCons
info_ndarr = ndu.info_ndarr

# info_gain_mode_arrays(gmaps, first=0, last=5), pixel_gain_mode_statistics(gmaps), info_pixel_gain_mode_statistics(gmaps)
import logging
logger = logging.getLogger(__name__)

SEGMENT_SHAPE = (336, 576)
#ASIC_SHAPE    = (168, 192)

# order of gain modes as in dark: datinfo -k exp=mfx101628626,run=163 -d epixuhr3x2
GAIN_MODES = ('FHG', 'FMG', 'FLG1', 'FLG2', 'AHLG1', 'AHLG2', 'AMLG1', 'AMLG2') #  8 total in dark
GAIN_STATES_LOW = ('AHLG1_L', 'AHLG2_L', 'AMLG1_L', 'AMLG2_L')                  #  4
GAIN_STATES = GAIN_MODES + GAIN_STATES_LOW                                      # 12 total
GAINS = (1.51, 0.54, 0.0172, 0.0086) # mV/keV gains for FHG, FMG, FLG1, FLG2
GAIN_FACTORS = list([GAINS[i] for i in (0,1,2,3,0,0,1,1,2,3,2,3)]) # for 12 states
dic_gain_state_to_factor = dict(zip(GAIN_STATES, GAIN_FACTORS))

# epix10ka data gainbit and mask
B14 = 0o40000 # 16384 or 1<<14 (15-th bit starting from 1)
M14 = 0x3fff  # 16383 or (1<<14)-1 - 14-bit mask

# epixuhr data gainbit and mask
B15 = 0o100000 # 32768 or 1<<15 (16-th bit starting from 1)
M15 = 0x7fff   # 32767 or (1<<15)-1 - 15-bit maskdef gain_bitword(dettype):

DTYPE_MASK = np.uint8

def cond_msg(c, msg='condition is True', output_meth=logger.debug):
    if c: output_meth(msg)
    return c

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
    """The shape of output arrays is identiacl to the shape of the input array cbits,
       shape=(<number-of-segments>, <2-d-panel-shape>)
       return gr0, gr1, gr2, gr3, gr4, gr5, gr6, gr7, gr8, gr9, gr10, gr11
       per-pixel bool for 12 gain ranges - maps of gain groups.
       works for both epixuhr 2x2 (336, 384) and 3x2 (336, 576)
       see: https://confluence.slac.stanford.edu/x/JCWDHw

       cbits - control bit array

       bit 6: data bit 0 is added here to distinguish gain modes in configuration
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
     1010x00 = 16+64 AHLG1  use 2 + offset AHLG1_L
     1110x00 = 48+64 AHLG2  use 3 + offset AHLG2_L
     1011x00 = 24+64 AMLG1  use 2 + offset AMLG1_L
     1111x00 = 56+64 AMLG2  use 3 + offset AMLG2_L
      xxx0xx       injection OFF
      xxx1xx       injection ON
      xxxx1x = 2   mask pixel
      x0xxxx       auto-gain OFF
      x1xxxx       auto-gain ON
      111111 = 63  mask of all except data bit
     1000000 = 64  data bit for gain switching to low
     1111111 =127  mask of all significant config bits
    """

    if ut.is_none(cbits, 'cbits is None', logger.debug): return None

    cbitsM = cbits & 59    # full bit-mask ignoring  gain bit 64, ignoring injection bit 0o4, 59 = 63 - 4
    cbitsF = cbits & 59+64 # full bit-mask including gain bit 64, ignoring injection bit 0o4, 59 = 63 - 4
    return\
          (cbitsM == 32),\
          (cbitsM == 40),\
          (cbitsM ==  1),\
          (cbitsM == 33),\
          (cbitsF == 16),\
          (cbitsF == 48),\
          (cbitsF == 24),\
          (cbitsF == 56),\
          (cbitsF == 16+64),\
          (cbitsF == 48+64),\
          (cbitsF == 24+64),\
          (cbitsF == 56+64)


def event_constants_for_gmaps(gmaps, cons, default=0, cmt=''):
    """ 6 msec
    Parameters
    ----------
    - gmaps - tuple of 8 boolean maps ndarray(<nsegs>, 336, 576)
    - cons - 4d constants  (8, <nsegs>, 336, 576)
    - default value for constants

    - assuming that pedestals are calibrated for 8 gain modes (4-FIXED, 2-H>H, 2-M>M),
    - but canstants can be also defined for all 12 states of gain modes (4-FIXED, 2-H>H, 2-M>M, 2-H>L, 2-M>L),

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
                             cons[8,:], cons[9,:], cons[10,:],cons[11,:]), default=default)
    #indices for LOW gain states AHLG1_L, AHLG2_L, AMLG1_L, AMLG2_L
    #if 8 or 12 gain states are defined in cons for pedestals or pixel_gain
    #i08, i09, i10, i11 = (2,3,2,3) if cons.shape[0] < 9 else (8,9,10,11)
    #                         cons[i08,:], cons[i09,:], cons[i10,:], cons[i11,:]), default=default)


def reshape_6x32256_to_6x168x192(a, shape_out=(6, 168, 192)):
    """returns array shaped as (2*3, 168, 192) from raw shape (6, 32256)"""
    a.shape = shape_out # (2,3,) + shape_asic
    return a


def stack_3x2_asics(a):
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


def cbits_config_segment_3x2(cob):
    """used in lcls2/psana/psana/detector/epixuhr3x2.py: _cbits_config_segment
       returns segment gain control bits shape=(336, 576), stacked from 6 ASICs (6, 32256) = (6, 168, 192)
       cob=det.raw._seg_configs()[<seg-ind>].config - segment configuration object, where self=det.raw
    """
    gasic = cob.gainAsic            # [56 56 56 56 56 56]
    cbits = cob.gainCSVAsic.copy()  # shape:(6, 32256)
    logger.debug(f'\n  XXX dir(cob): {str(dir(cob))}'\
                +f'\n  XXX cob.gainAsic: {str(gasic)}'\
                +info_ndarr(cbits, '\n  XXX cob.gainCSVAsic', last=10))
    cbits.shape = (6, 168, 192) # reshape_6x32256_to_6x168x192(cbits)
    for i, g in enumerate(gasic):
        if g > 0: cbits[i,:] = g # substitute code from cob.gainAsic if not 0
    cbits = stack_3x2_asics(cbits) # (336, 576)
    logger.debug(info_ndarr(cbits, 'segment cbits', last=10))
    return cbits


def bit_opers_3x2_raw(a):
    """bit operations for raw data - do not shift data, just apply 12-bit mask.
       The ePixUHR3x2, when not using "gain expansion" transmits the data as 12
       bits, in a 16 bit integer. The layout of this 16-bit word data is:

                           U U U U D D D D D D D D U U U G
                           \_____/ \___________________/ |
                              |              |            \
                         Unused bits    11 data bits    Gain bit
    """
    return a & 0x0FFF


def bit_opers_3x2_calib(a):
    """bits right_shift operations for calib - 1-bit shift raw data
    """
    return np.right_shift(a, 1) #, dtype=np.uint16)
    #return a >>= 1


def raw_v01(det_raw, evt, sh_seg=(336,576)):
    """TBD: this operation should be done in FPGA
       stack raw data for 6 ASICs of each panel/segment,
       returns raw for all segments shaped as (<number of segments>, 336, 576)
       assembled from per panel 6-ASIC arrays (6, 32256)=(6, 168, 192)
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
        arr2 = bit_opers_3x2_raw(raw_asics)
        asics = reshape_6x32256_to_6x168x192(arr2) # (6, 168, 192)
        arrseg = stack_3x2_asics(asics) # (336, 576)
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


def gain_default(nsegs=2, gfactors=GAIN_FACTORS, shape_seg=SEGMENT_SHAPE):
    """returns array of default gain constants of shape (12, nsegs, <shape_seg>)"""
    import psana.pscalib.calib.CalibConstants as CC
    dtype_gain = CC.dic_calib_type_to_dtype[CC.PIXEL_GAIN] # np.float32
    a = np.empty((12, nsegs) + shape_seg, dtype=dtype_gain)
    for igm, gf in enumerate(gfactors):
        a[igm,:] = gf
    return a


def combine_peds_offs(peds, offs):
    """returns 4d pedestals for entire detector (12, <nsegs>, 336, 576)
       combined from calibration constants:
         peds = det_raw._pedestals() # - 4d pedestals  (8, <nsegs>, 336, 576) from dark
         offs = det_raw._offset()    # - 4d offsets    (4, <nsegs>, 336, 576) from Philip

       Philip and Zongde on 2026-09-13:
       12 gains for FHG, FMG, FLG1, FLG2, AHLG1, AHLG2, AMLG1, AMLG2, AHLG1_L, AHLG2_L, AMLG1_L, AMLG2_L
       8 pedestals for FHG, FMG, FLG1, FLG2, AHLG1, AHLG2, AMLG1, AMLG2
       4 offsets for AHLG1, AHLG2, AMLG1, AMLG2

       ped_AHLG1_L = ped_FLG1 + offset_AHLG1
       ped_AHLG2_L = ped_FLG2 + offset_AHLG2
       ped_AMLG1_L = ped_FLG1 + offset_AMLG1
       ped_AMLG2_L = ped_FLG2 + offset_AMLG2
    """
    if cond_msg(peds is None, msg='in combine_peds_offs: peds is None', output_meth=logger.warning):
        return None
    logger.debug('XXX combine_peds_offs:'\
                +info_ndarr(peds, '\n  peds')\
                +info_ndarr(offs, '\n  offs'))

    sh_seg = peds.shape[1:]
    if offs is None:
       offs = np.zeros((4,)+sh_seg, dtype=peds.dtype)
       logger.warning(info_ndarr(offs, 'offs = None are substituted with zeros:'))

    peds_AHLG_L = peds[2:4,:] + offs[0:2,:] # where 2:4 and 0:2 stands for two indices LG1,2
    peds_AMLG_L = peds[2:4,:] + offs[2:4,:]
    peds = np.vstack((peds, peds_AHLG_L, peds_AMLG_L))
    return peds


class Storage_epixuhr_v01():
    def __init__(self, det_raw, **kwa):
        """Holds constants for 2-indices of the gain switching data bit (1st for epixuhr3x2).
        Parameters
        ----------
        - det_raw = det.raw
        - **kwa
          -------
          - cmpars (tuple) - common mode parameters, e.g. (7,2,100,10)
          - perpix (bool) - if True, preserves peds and gfac arrays shaped per pixel, as (<nsegs>, 336, 576, 7)
        """

        self.peds = None
        self.gfac = None
        self.mask = None
        self.counter = -1

        cmpars = kwa.get('cmpars', None)
        perpix = kwa.get('perpix', False)

        logger.info(f'Storage_epixuhr_v01 {det_raw._info_calibconst()}')

        gain = det_raw._gain()      # - 4d gains     (12, <nsegs>, 336, 576)
        peds = det_raw._pedestals() # - 4d pedestals  (8, <nsegs>, 336, 576)
        offs = det_raw._offset()    # - 4d offsets    (4, <nsegs>, 336, 576)

        logger.debug('Storage_epixuhr_v01 calib constants from DB:'\
             +info_ndarr(peds, '\n  peds')\
             +info_ndarr(gain, '\n  gain')\
             +info_ndarr(offs, '\n  offs'))

        peds = combine_peds_offs(peds, offs)

        if cond_msg(gain is None, msg='Storage_epixuhr_v01 gain is None - use default', output_meth=logger.warning):
            if peds is not None:
                gain = gain_default(nsegs=peds.shape[1])

        logger.debug(info_ndarr(peds, 'XXX combined_peds_offs:'))
        logger.debug(info_ndarr(gain, 'XXX gain:'))
        #sys.exit('TEST EXIT')

        cbits_hm = cbits = det_raw._cbits_config_detector() # full detector shape (<nsegs>, 336, 576)
        cbits_lo = cbits_config_add_bit(cbits, bit=0b1000000) # cbits | bit - force adding the 6-th gain bit to the config control bits
        gmaps_hm = gain_maps_epixuhr(cbits_hm) # gr0, gr1, gr2, ..., gr11 boolean maps of shape (<nsegs>, 336, 576)
        gmaps_lo = gain_maps_epixuhr(cbits_lo)

        self.shape_det = tuple(peds.shape)[-3:] if peds is not None else None
        self.shape_as_daq = det_raw._shape_as_daq()

        # select per/pixel constants for H,M / L1,2 gains for gain bit 0/1, respectively
        # full detector shape (<nsegs>, 336, 576)
        peds_hm = event_constants_for_gmaps(gmaps_hm, peds, default=0, cmt='peds for HM ')
        peds_lo = event_constants_for_gmaps(gmaps_lo, peds, default=0, cmt='peds for LO ')
        gain_hm = event_constants_for_gmaps(gmaps_hm, gain, default=1, cmt='gain for HM ')
        gain_lo = event_constants_for_gmaps(gmaps_lo, gain, default=1, cmt='gain for LO ')

        mask = det_raw._mask(**kwa)
        if mask is None: mask = det_raw._mask_from_status(**kwa)
        if mask is None: mask = np.ones(self.shape_det, dtype=DTYPE_MASK)
        self.mask = mask

        # combine switching gain constants in 4-d arrays
        peds_sw = np.stack((peds_hm, peds_lo)) # 2x (H/M,L1/2) detector shape (2, <nsegs>, 336, 576)
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

        s = 'Storage_epixuhr_v01 constants:'\
          + f'\n  shape_det: {self.shape_det}\n  shape_as_daq: {self.shape_as_daq}'\
          + f'\n  cmpars: {self.cmpars}'\
          + info_ndarr(cbits_hm,  '\n  cbits_hm')\
          + info_ndarr(cbits_lo,  '\n  cbits_lo')\
          + info_ndarr(self.mask, '\n  mask')\
          + info_ndarr(self.peds, '\n  peds')\
          + info_ndarr(self.gfac, '\n  gfac')
        logger.info(s)


def calib_v01(det_raw, evt, **kwa):
    print('UtilsEpixUHR: calib_v01 - returns raw')
    return det_raw.raw(evt)


def grindex_array(raw, gbit=0):
    """grab the gain bit in its position in raw data and move it to the right-most position,
       returns array of 0/1 for gain index
       gbit (starting from 0) - gain index position in raw data, = 0 for epixuhr3x2
    """
    arr1b = np.bitwise_and(raw, 1<<gbit)
    return arr1b if gbit==0 else np.right_shift(arr1b, gbit)
    #return (raw & 1<<gbit) >> gbit


def calib_v02(det_raw, evt, **kwa):
    """ """
    logger.debug(f'UtilsEpixUHR: calib_v02 kwa: {str(kwa)}')

    nda_raw = kwa.get('nda_raw', None)
    #cmpars  = kwa.get('cmpars', None)
    raw = det_raw.raw(evt) if nda_raw is None else nda_raw # shape: (<nsegs>, 336, 576) dtype:uint16
    if cond_msg(raw is None, msg='raw is None', output_meth=logger.warning): return None

    store = det_raw._store_ = Storage_epixuhr_v01(det_raw, **kwa) if det_raw._store_ is None else det_raw._store_  #perpix=True
    store.counter += 1

    igr = grindex_array(raw, gbit=det_raw._data_gain_bitnum) # raw & 1 - for epahuhr3x2

    t0_sec = time()
    if store.peds is not None and store.peds[0] is not None:
        pedest = np.select((igr==0, igr==1), (store.peds[0,:], store.peds[1,:]))
        factor = np.select((igr==0, igr==1), (store.gfac[0,:], store.gfac[1,:]))
    else:
        pedest = 0
        factor = 1
    logger.debug('np.select for pedest & factor time: %.6f sec' % (time() - t0_sec)\
                 +info_ndarr(factor,  '\n    factor:')\
                 +info_ndarr(pedest,  '\n    pedest:'))

    # Shift data down 1 bit, since the gain bit was the LSB
    arrf = np.array(bit_opers_3x2_calib(raw), dtype=np.float32)
    #arrf = np.array(raw >>= 1, dtype=np.float32)

    #if cond_msg(pedest is None, msg='pedest is None - return raw', output_meth=logger.debug): return arrf
    arrf -= pedest

    #if store.cmpars is not None:
    #    common_mode_epix_multigain_apply(arrf, gmaps, store)

    #if cond_msg(factor is None, msg='factor is None - return raw-peds', output_meth=logger.debug): return arrf
    arrf *= factor # factor * mask

    #logger.info(info_ndarr(arrf,  'calib:'))
    return arrf

# EOF
