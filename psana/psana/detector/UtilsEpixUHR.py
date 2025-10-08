
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
#import numpy as np
#from time import time
import psana.detector.Utils as ut # info_dict, is_true, is_none
import logging
logger = logging.getLogger(__name__)

GAIN_MODES = ('FHG', 'FMG', 'FLG1', 'FLG2', 'AHLG1', 'AHLG2', 'AMLG1', 'AMLG2')

#B04 =    0o20 #    16 or 1<<4   (5-th bit starting from 1)
#B05 =    0o40 #    32 or 1<<5   (6-th bit starting from 1)

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

def gain_maps_epixuhr(cbits):
    """Returns maps of gain groups shape=(<number-of-segments>, <2-d-panel-shape>)
       works for both epixuhr ??????????? (352, 384) and epixhr2x2 (288, 384)

       cbits - pixel control bit array

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

      100x00 =32   FHG   0
      101x00 =40   FMG   1
      000x01 = 1   FLG1  2
      100x01 =33   FLG2  3
      010x00 =16   AHLG1 4
      110x00 =48   AHLG2 5
      011x00 =24   AMLG1 6
      111x00 =56   AMLG2 7
      XXXX1X = 2   mask pixel



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



# EOF
