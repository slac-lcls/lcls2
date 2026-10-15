
"""
:py:class:`UtilsEpixm320` contains utilities for epixm320
=========================================================

Usage::
    import psana.detector.UtilsEpixm320 as uem

EPIXM (my): https://confluence.slac.stanford.edu/spaces/PSDM/pages/436570012/EPIXM

@author Mikhail Diubrovin
Created on 2026-10-07
"""
#import os
import sys
import numpy as np
from time import time
import psana.detector.NDArrUtils as ndu
import psana.detector.Utils as ut # info_dict, is_true, is_none
import logging
logger = logging.getLogger(__name__)

is_none = ut.is_none
info_ndarr = ndu.info_ndarr


class Storage_epixm320_v01():
    def __init__(self, det_raw, **kwa):
        """Holds constants for 2-indices of the gain switching data bit (16-th for epixm320).
        Parameters
        ----------
        - det_raw = det.raw
        - **kwa
          -------
          - cmpars (tuple) - common mode parameters, e.g. (7,2,100,10)
        """
        logger.info(f'{det_raw.__class__.__name__}.{sys._getframe().f_code.co_name}')

        #self.set_config(det_raw, **kwa)

        self.peds = None
        self.gfac = None
        self.mask = None
        self.counter = -1

        cmpars = kwa.get('cmpars', None)
        perpix = kwa.get('perpix', False)

        logger.info(f'Storage_epixm320_v01 {det_raw._info_calibconst()}\n')

        gain = det_raw._gain()      # - 4d gains     (12, <nsegs>, 336, 576)
        peds = det_raw._pedestals() # - 4d pedestals  (8, <nsegs>, 336, 576)
        offs = det_raw._offset()    # - 4d offsets    (4, <nsegs>, 336, 576)

        #logger.info('Storage_epixm320_v01 calib constants from DB:'\
        #     +info_ndarr(peds, '\n  peds')\
        #     +info_ndarr(gain, '\n  gain')\
        #     +info_ndarr(offs, '\n  offs'))

        self.shape_det = tuple(peds.shape)[-3:] if peds is not None else None
        self.shape_as_daq = det_raw._shape_as_daq()

        mask = det_raw._mask(**kwa)
        if mask is None: mask = det_raw._mask_from_status(**kwa)
        if mask is None: mask = np.ones(self.shape_det, dtype=DTYPE_MASK)
        self.mask = mask
        #logger.info(info_ndarr(self.mask, '\n  mask'))

        igm_SH_gain = 1
        igm_SH_peds = 1

        gain_SH = gain[igm_SH_gain, :]
        peds_SH = gain[igm_SH_peds, :]

        self.gfac = ndu.divide_protected(np.ones_like(gain_SH), gain_SH) * mask
        self.peds = peds_SH

        logger.info(f'Storage_epixm320_v01 calibration constants for A SINGLE gain mode SH'\
                    +info_ndarr(self.peds, '\n  peds')\
                    +info_ndarr(self.gfac, '\n  gfac')\
                    +info_ndarr(self.mask, '\n  mask'))

        
        if False:
             # =================
             # STAF FROM EPIXUHR

             peds = combine_peds_offs(peds, offs)

             if cond_msg(gain is None, msg='Storage_epixm320_v01 gain is None - use default', output_meth=logger.warning):
                 if peds is not None:
                     gain = gain_default(nsegs=peds.shape[1])

             logger.debug(info_ndarr(peds, 'XXX combined_peds_offs:'))
             logger.debug(info_ndarr(gain, 'XXX gain:'))
             #sys.exit('TEST EXIT')

             cbits_hm = cbits = det_raw._cbits_config_detector() # full detector shape (<nsegs>, 336, 576)
             cbits_lo = cbits_config_add_bit(cbits, bit=0b1000000) # cbits | bit - force adding the 6-th gain bit to the config control bits
             gmaps_hm = gain_maps_epixm320(cbits_hm) # gr0, gr1, gr2, ..., gr11 boolean maps of shape (<nsegs>, 336, 576)
             gmaps_lo = gain_maps_epixm320(cbits_lo)

             # select per/pixel constants for H,M / L1,2 gains for gain bit 0/1, respectively
             # full detector shape (<nsegs>, 336, 576)
             peds_hm = event_constants_for_gmaps(gmaps_hm, peds, default=0, cmt='peds for HM ')
             peds_lo = event_constants_for_gmaps(gmaps_lo, peds, default=0, cmt='peds for LO ')
             gain_hm = event_constants_for_gmaps(gmaps_hm, gain, default=1, cmt='gain for HM ')
             gain_lo = event_constants_for_gmaps(gmaps_lo, gain, default=1, cmt='gain for LO ')

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

             s = 'Storage_epixm320_v01 constants:'\
               + f'\n  shape_det: {self.shape_det}\n  shape_as_daq: {self.shape_as_daq}'\
               + f'\n  cmpars: {self.cmpars}'\
               + info_ndarr(cbits_hm,  '\n  cbits_hm')\
               + info_ndarr(cbits_lo,  '\n  cbits_lo')\
               + info_ndarr(self.mask, '\n  mask')\
               + info_ndarr(self.peds, '\n  peds')\
               + info_ndarr(self.gfac, '\n  gfac')
             logger.info(s)


    def set_config(self, det_raw, **kwa):
        sconfs = det_raw._seg_configs()
        print('\ndet_raw._seg_configs():', sconfs) # {0: <psana.container.Container object at 0x7f93260f61f0>}
        #print('\ndir(det.raw._seg_configs()):', dir(sconfs))
        print('\nsconfs.keys():', sconfs.keys())

        for k,v in det_raw._seg_configs().items():
            cob = v.config
            print('dir(cob) w/o underscores:', [v for v in tuple(dir(cob)) if v[0]!='_'])
            print('  cob.step', cob.step)
            print('  cob.startCol', cob.startCol)
            print('  cob.endCol', cob.endCol)
            print('  cob.currentAsic', cob.currentAsic)
            print('  cob.CompTH_ePixM', cob.CompTH_ePixM)
            print('  cob.Precharge_DAC_ePixM', cob.Precharge_DAC_ePixM)














def calib_v00(det_raw, evt, **kwa):
    print('UtilsEpixm320: calib_v00 - returns raw')
    return det_raw.raw(evt)


def calib_v01(det_raw, evt): # already defined in epix_base and AreaDetectorRaw
    """returns (raw-peds)*mask"""
    logger.debug(f'{det_raw.__class__.__name__}.{sys._getframe().f_code.co_name}')

    if is_none(evt, 'evt is None - return None', logger.debug): return None

    #t0_sec = time()
    raw = det_raw.raw(evt)
    if is_none(raw, 'det_raw.raw(evt) is None - return None'): return None

    # Subtract pedestals
    peds = det_raw._pedestals()
    if is_none(peds, 'det.raw._pedestals() is None - return det.raw.raw(evt)', logger.debug):
        return raw
    #print(info_ndarr(peds,'XXX peds', first=1000, last=1005))

    gr1 = (raw & det_raw._data_gain_bit) > 0

    #print(info_ndarr(gr1,'XXX gr1', first=1000, last=1005))
    pedgr = np.select((gr1,), (peds[1,:],), default=peds[0,:])
    arrf = np.array(raw & det_raw._data_bit_mask, dtype=np.float32)
    arrf -= pedgr

    #print('XXX time for calib: %.6f sec' % (time()-t0_sec)) # 4ms on drp-neh-cmp001
    mask = det_raw._mask()

    #print(info_ndarr(mask,'XXX mask', first=1000, last=1005)) # IT WORKS mask is available

    return arrf if is_none(mask, 'det.raw._mask() is None - return raw-peds', logger.info) else\
           arrf * mask


def calib_v02(det_raw, evt, **kwa):
    """WORKS FOR SH GAIN MODE ONLY, returns FOR SL: (raw-peds)*gfactor*mask"""
    if det_raw._count_calib == 0:
       det_raw._count_calib += 1
       logger.info(f'{det_raw.__class__.__name__}.{sys._getframe().f_code.co_name}')

    nda_raw = kwa.get('nda_raw', None)
    raw = det_raw.raw(evt) if nda_raw is None else nda_raw # shape:(4, 192, 384) size:294912 dtype:uint16
    if is_none(raw, 'det_raw.raw(evt) is None - return None', logger.warning): return None

    store = det_raw._store_ = Storage_epixm320_v01(det_raw, **kwa) if det_raw._store_ is None else det_raw._store_  #perpix=True
    store.counter += 1

    if is_none(store.peds, 'store.peds is None - return raw', logger.warning): return raw

    arrf = np.array(raw & det_raw._data_bit_mask, dtype=np.float32)

    return (arrf - store.peds) * store.gfac

#    # STAF FROM EPIXUHR calib for gain mode selection
#    #================================================
#    igr = grindex_array(raw, gbit=det_raw._data_gain_bitnum) # raw & 1 - for epahuhr3x2

#    t0_sec = time()
#    if store.peds is not None and store.peds[0] is not None:
#        pedest = np.select((igr==0, igr==1), (store.peds[0,:], store.peds[1,:]))
#        factor = np.select((igr==0, igr==1), (store.gfac[0,:], store.gfac[1,:]))
#    else:
#        pedest = 0
#        factor = 1


def calib_versions(self, evt, **kwa):
    """switch between versions"""
    version = kwa.get('version', 2)
    return calib_v02(self, evt, **kwa) if version == 2 else\
           calib_v01(self, evt, **kwa) if version == 1 else\
           calib_v00(self, evt)

# EOF

