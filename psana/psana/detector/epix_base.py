
"""
Data access BASIC METHODS for composite detectors made of epix10ka panels
=========================================================================

Usage::

  from psana.detector.epix_base import epix_base

  o = epix_base(*args, **kwargs) # inherits from AreaDetector
  a = o.calib(evt)
  m = o._mask_from_status(gain_range_inds=(0,1,2,3,4), **kwa)
  m = o._mask_edges(self, edge_rows=1, edge_cols=1, center_rows=0, center_cols=0, dtype=DTYPE_MASK, **kwa)

2020-11-06 created by Mikhail Dubrovin
"""

from amitypes import Array3d

import logging
logger = logging.getLogger(__name__)

from psana.detector.areadetector import sgs, AreaDetectorRaw, np, ut, DTYPE_MASK, DTYPE_STATUS, au
from psana.detector.UtilsEpix10ka import np, calib_epix10ka_any, map_gain_range_index,\
  cbits_config_epix10ka, cbits_config_epixhr2x2, cbits_config_epixhr1x4,\
  cbits_config_and_data_detector, M14, M15, B14, B15
import psana.detector.UtilsMask as um #import merge_status
info_ndarr, reshape_to_3d = au.info_ndarr, au.reshape_to_3d


class epix_base(AreaDetectorRaw):

    def __init__(self, *args, **kwa):
        logger.debug('epix_base.__init__') # self.__class__.__name__
        AreaDetectorRaw.__init__(self, *args, **kwa)
        self._seg_geo = None
        self._data_bit_mask = M14 # for epix10ka data
        self._data_gain_bit = B14 # for epix10ka data
        self._gain_bit_shift = 9  # for epix10ka data
        self._gains_def = (16.4, 5.466, 0.164) # ADU/keV epix10ka  H:M:L = 1 : 1/3 : 1/100
        #self._gains_def = (41.0, 13.7, 0.512) # ADU/keV epixhr2x2 H:M:L = 1 : 1/3 : 1/80


    def calib(self, evt, **kwa) -> Array3d:
        """Returns calibrated data array."""
        logger.debug('epix_base.calib - the same for epix10ka and epixhr2x2')
        return calib_epix10ka_any(self, evt, **kwa)
        #return self.raw(evt)


    def _gain_range_index(self, evt, **kwa):
        """Returns array (shaped as raw) per pixel gain range index or None."""
        logger.debug('epix_base._gain_range_index - the same for epix10ka and epixhr2x2')
        return map_gain_range_index(self, evt, **kwa)


    def _segment_indices(self):
        """Returns list det.raw._segment_numbers, e.g. [0, 1, 2, 3] if not re-implemented it is self._sorted_segment_inds"""
        return self._segment_numbers


    def _fullname(self):
        """Returns detector full name, e.g. for epix
           epix_3926196238-0175152897-1157627926-0000000000-0000000000-0000000000-0000000000\
               _3926196238-0174824449-0268435478-0000000000-0000000000-0000000000-0000000000\
               _3926196238-0175552257-3456106518-0000000000-0000000000-0000000000-0000000000\
               _3926196238-0176373505-4043309078-0000000000-0000000000-0000000000-0000000000
        """
        return self._uniqueid


    def _segment_ids(self):
        """Returns list of detector segment ids, e.g. for epix10ka
        [3926196238-0175152897-1157627926-0000000000-0000000000-0000000000-0000000000,
         3926196238-0174824449-0268435478-0000000000-0000000000-0000000000-0000000000,
         3926196238-0175552257-3456106518-0000000000-0000000000-0000000000-0000000000,
         3926196238-0176373505-4043309078-0000000000-0000000000-0000000000-0000000000]
        """
        return self._uniqueid.split('_')[1:]


#    def _config_object(self):
#        """Returns [dict]={<seg-index>:<cob>} of configuration objects for det.raw
#        """
#        #logger.debug('det_raw._seg_configs(): ' + str(self._seg_configs()))
#        return self._seg_configs()


    def _cbits_config_segment(self, cob):
        """for epix10ka and epixhr2x2, ...
           returns per-pixel array of gain control bits from segment configuration object
           cob=det.raw._seg_configs()[<seg-ind>].config
        """
        logger.warning('epix_base._cbits_config_segment - MUST BE REIMPLEMENTED - return None')
        return None


    def _cbits_segment_ind(self, i):
        """for epix10ka and epixhr2x2, ...
           returns per-pixel array of gain control bits for segment index
        """
        return self._cbits_config_segment(self._seg_configs()[i].config)


    def _config_object(self):
        """protected call to _seg_configs"""
        dcfg = self._seg_configs()
        if dcfg is None:
            logger.debug('epix_base._config_object - self._seg_configs is None - return None')
            return None
        return dcfg


    def _cbits_config_detector(self):
        """Returns array of control bits shape=(<number-of-segments>, 352, 384) from any config object."""
        dcfg = self._config_object() # or alias self._seg_configs()
        if dcfg is None: return None
        #print('XXX dcfg:', dcfg)
        lst_cbits = [self._cbits_config_segment(v.config) for k,v in dcfg.items()]
        return np.stack(tuple(lst_cbits))


    def _cbits_config_and_data_detector(self, evt=None):
        return cbits_config_and_data_detector(self, evt)


    def _mask_from_status(self, status_bits=0xffff, gain_range_inds=(0,1,2,3,4), dtype=DTYPE_MASK, **kwa):
        """re-implementation of AreaDetectorRaw._mask_from_status for multi-gain detector.
        """
        logger.debug('epix_base._mask_from_status - implementation for epix10ka - merged status masks for gain ranges')
        return AreaDetectorRaw._mask_from_status(self, status_bits=status_bits, gain_range_inds=gain_range_inds, dtype=dtype, **kwa)
        #return um.merge_mask_for_grinds(smask, gain_range_inds=gain_range_inds, dtype=dtype, **kwa)


if __name__ == "__main__":
    import sys
    sys.exit('See example in test_%s if available...' % sys.argv[0].split('/')[-1])

# EOF
