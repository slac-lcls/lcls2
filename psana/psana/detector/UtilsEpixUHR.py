
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
#import sys
import numpy as np
#from time import time
import psana.detector.NDArrUtils as ndu
import psana.detector.Utils as ut # info_dict, is_true, is_none
import psana.detector.UtilsEpix10ka as ue10ka
info_gain_mode_arrays, pixel_gain_mode_statistics, info_pixel_gain_mode_statistics =\
    ue10ka.info_gain_mode_arrays, ue10ka.pixel_gain_mode_statistics, ue10ka.info_pixel_gain_mode_statistics

cond_msg = ue10ka.cond_msg


# info_gain_mode_arrays(gmaps, first=0, last=5), pixel_gain_mode_statistics(gmaps), info_pixel_gain_mode_statistics(gmaps)
import logging
logger = logging.getLogger(__name__)

#SEGMENT_SHAPE = (336, 576)
#ASIC_SHAPE    = (168, 192)

GAIN_MODES = ('FHG', 'FMG', 'FLG1', 'FLG2', 'AHLG1', 'AHLG2', 'AMLG1', 'AMLG2')

# epix10ka data gainbit and mask
B14 = 0o40000 # 16384 or 1<<14 (15-th bit starting from 1)
M14 = 0x3fff  # 16383 or (1<<14)-1 - 14-bit mask

# epixuhr data gainbit and mask
B15 = 0o100000 # 32768 or 1<<15 (16-th bit starting from 1)
M15 = 0x7fff   # 32767 or (1<<15)-1 - 15-bit maskdef gain_bitword(dettype):

def gain_bitword(dettype):
    return {'epix10ka':B14, 'epixhr':B15, 'epixhr2x2':B15, 'epixhremu':B15}.get(dettype, None)

def data_bitword(dettype):
    return {'epix10ka':M14, 'epixhr':M14, 'epixhr2x2':M14, 'epixhremu':M14}.get(dettype, None)

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
    """
       return gr0, gr1, gr2, gr3, gr4, gr5, gr6, gr7 # per-pixel bool for 8 gain ranges
       seturns maps of gain groups shape=(<number-of-segments>, <2-d-panel-shape>)
       works for both epixuhr ??????????? (352, 384) and epixhr2x2 (288, 384)

       cbits - control bit array

       bit 6: data bit 15 is moved here 1/0 for H,M/L
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

      100000 =32   FHG   0 injection OFF
      101x00 =40   FMG   1
      000x01 = 1   FLG1  2
      100x01 =33   FLG2  3
      010x00 =16   AHLG1 4
      110x00 =48   AHLG2 5
      011x00 =24   AMLG1 6
      111x00 =56   AMLG2 7
      xxx0xx       injection OFF
      xxx1xx       injection ON
      xxxx1x = 2   mask pixel
      x0xxxx       auto-gain OFF
      x1xxxx       auto-gain ON

      ==== epix10ka stuff
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

    if ut.is_none(cbits, 'cbits is None', logger.debug): return None

    cbitsM = cbits & 0o73 # 59 = 63 - 4 cbits with masked injection bit 0o4
    return\
          (cbitsM == 32),\
          (cbitsM == 40),\
          (cbitsM ==  1),\
          (cbitsM == 33),\
          (cbitsM == 16),\
          (cbitsM == 48),\
          (cbitsM == 24),\
          (cbitsM == 56)


def cbits_config_segment(cob):
    """cob=det.raw._seg_configs()[<seg-ind>].config - segment configuration object, where self=det.raw
       returns segment gain control bits # shape=(336, 576)
    """
    logger.debug('XXX dir(cob): %s' % str(dir(cob))) #'gainAsic', 'gainCSVAsic']
    gasic = cob.gainAsic            # [56 56 56 56 56 56]
    cbits = cob.gainCSVAsic.copy()  # shape:(6, 32256)
    logger.debug('  XXX cob.gainAsic: %s' % str(gasic))
    logger.debug(ndu.info_ndarr(cbits, '  XXX cob.gainCSVAsic', last=10))
    cbits.shape = (6, 168, 192) # reshape_6x32256_to_6x168x192(cbits)
    for i, g in enumerate(gasic):
        if g > 0: cbits[i,:] = g
    cbits = stack_2x3_asics(cbits) # (336, 576)
    logger.info(ndu.info_ndarr(cbits, 'segment cbits', last=10))
    return cbits
   # return eb.cbits_config_epix10ka(cob, shape=(352, 384)) # in epix10ka.py


def bit_opers_2x3(a):
    """bit massaging of panel data"""
    gbit = a & 0o100000                # save array with gain bit in 16-th position
    a = np.right_shift(a & 0o77777, 4) # mask 15 lower bits of data and move them 4 bits right
    a = np.bitwise_or(a, gbit)         # set the gain bit
    return a


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

def calib(det_raw, evt, **kwa):
    """"""
    #logger.debug('UtilsEpixUHR: calib')
    print('UtilsEpixUHR: calib')
    return det_raw.raw(evt)

# EOF
