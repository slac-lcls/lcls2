
"""
Data access BASIC METHODS for composite detectors made of epix10ka panels
=========================================================================

Usage::

  from psana.detector.epix_base import epix_base

  o = epix_base(*args, **kwargs) # inherits from AreaDetector
  a = o.calib(evt)
  m = o._mask_from_status(grinds=(0,1,2,3,4), **kwa)
  m = o._mask_edges(self, edge_rows=1, edge_cols=1, center_rows=0, center_cols=0, dtype=DTYPE_MASK, **kwa)

2020-11-06 created by Mikhail Dubrovin
"""

from amitypes import Array3d

import logging
logger = logging.getLogger(__name__)

from psana.detector.areadetector import AreaDetector, np, DTYPE_MASK, DTYPE_STATUS
from psana.detector.UtilsEpix10ka import np, calib_epix10ka_any, map_gain_range_index,\
  cbits_config_epix10ka, cbits_config_epixhr2x2,\
  cbits_config_and_data_detector, M14, M15, B14, B15
from psana.detector.UtilsMask import merge_status
#from psana.pscalib.geometry.SegGeometryEpix10kaV1 import epix10ka_one as seg
from psana.pscalib.geometry.SegGeometryStore import sgs
from psana.pyalgos.generic.NDArrUtils import info_ndarr


class epix_base(AreaDetector):

    def __init__(self, *args, **kwa):
        logger.debug('epix_base.__init__') # self.__class__.__name__
        AreaDetector.__init__(self, *args, **kwa)
        self._seg_geo = None
        self._data_bit_mask = M14 # for epix10ka data
        self._data_gain_bit = B14 # for epix10ka data
        self._gain_bit_shift = 9  # for epix10ka data
        self._gains_def = (16.4, 5.466, 0.164) # epix10ka ADU/keV H:M:L = 1 : 1/3 : 1/100
        #self._gains_def = (41.0, 13.7, 0.512) # epixhr2x2 ADU/keV H:M:L = 1 : 1/3 : 1/80


    def calib(self, evt, **kwa) -> Array3d:
        """Returns calibrated data array."""
        logger.debug('epix_base.calib - TO BE REIMPLEMENTED - return raw')
        return self.raw(evt)
        #return calib_epix10ka_any(self, evt, **kwa)


    def _gain_range_index(self, evt, **kwa):
        """Returns array (shaped as raw) per pixel gain range index or None."""
        logger.debug('epix_base._gain_range_index - TO BE REIMPLEMENTED - returns zeroes shaped as raw')
        return np.zeros_like(self.raw(evt), dtype=np.uint16)
        #return map_gain_range_index(self, evt, **kwa)


    def _segment_indices(self):
        """Returns list det.raw._sorted_segment_ids, e.g. [0, 1, 2, 3]"""
        return self._sorted_segment_ids

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
        logger.debug('epix_base._cbits_config_segment - MUST BE REIMPLEMENTED - return None')
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
        lst_cbits = [self._cbits_config_segment(v.config) for k,v in dcfg.items()]
        return np.stack(tuple(lst_cbits))


    def _cbits_config_and_data_detector(self, evt=None):
        return cbits_config_and_data_detector(self, evt)


    def _mask_from_status(self, **kwa):
        """
        Parameters **kwa
        ----------------
        'grinds', (0,1,2,3,4)) # gain range indexes for 'FH','FM','FL','AHL-H','AML-M'
        Returns
        -------
        mask made of status: np.array, ndim=3, shaped as full detector data
        """
        logger.debug('epix_base._mask_from_status')
        _grinds = kwa.get('grinds',(0,1,2,3,4))
        status = self._status() # pixel_status from calibration constants
        statmrg = merge_status(status, grinds=_grinds, dtype=DTYPE_STATUS) # dtype=np.uint64
        return np.asarray(np.select((statmrg>0,), (0,), default=1), dtype=DTYPE_MASK)
        #logger.info(info_ndarr(status, 'status '))
        #return statmrg


#    def _seg_geo(**kwa):
#        #logger.debug('epix_base._seg_geo MUST BE RE-IMPLEMENTED')
#        return None


    def _mask_edges(self, edge_rows=1, edge_cols=1, center_rows=0, center_cols=0, dtype=DTYPE_MASK, **kwa):
        """
        Parameters
        ----------
        edge_rows: int number of edge rows to mask on both side of the panel
        edge_cols: int number of edge columns to mask on both side of the panel
        center_rows: int number of edge rows to mask for all ASICs
        center_cols: int number of edge columns to mask for all ASICs
        dtype: numpy dtype of the output array
        **kwa: is not used
        Returns
        -------
        mask: np.ndarray, ndim=3, shaped as full detector data, mask of the panel and asic edges
        """
        logger.debug('epix_base._mask_edges')
        mask1 = self._seg_geo.pixel_mask_array(edge_rows, edge_cols, center_rows, center_cols, dtype)
        nsegs = self._number_of_segments_total()
        if nsegs is None:
            logger.debug('_number_of_segments_total is None')
            return None
        logger.info('_mask_edges for %d-segment epix10ka'%nsegs)
        mask = np.stack([mask1 for i in range(nsegs)])
        logger.info(info_ndarr(mask, '_mask_edges '))
        return mask


if __name__ == "__main__":
    import sys
    sys.exit('See example in test_%s if available...' % sys.argv[0].split('/')[-1])

# EOF
