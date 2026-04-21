"""Data access VERSIONS
   for composite detectors made of epix10ka segments/panels.
"""
import numpy as np
from amitypes import Array2d, Array3d
import psana.detector.epix_base as eb
from psana.detector.detector_impl import DetectorImpl
import psana.detector.Utils as ut # info_dict, is_true, is_none
from psana.detector.NDArrUtils import info_ndarr

import logging
logger = logging.getLogger(__name__)

# make an empty detector interface for Matt's hardware
# configuration object so that config_dump works - cpo
class epix10kaquad_config_2_0_0(DetectorImpl):
    def __init__(self, *args, **kwargs):
        super(epix10kaquad_config_2_0_0, self).__init__(*args)


class epix10k_raw_0_0_1(eb.epix_base):
    def __init__(self, *args, **kwargs):
        logger.debug('%s.__init__' % self.__class__.__name__)
        eb.epix_base.__init__(self, *args, **kwargs)
        self._seg_geo = eb.sgs.Create(segname='EPIX10KA:V1')
        self._path_geo_default = 'pscalib/geometry/data/geometry-def-epix10ka-TBD.data'


class epix10ka_raw_2_0_1(eb.epix_base):
    def __init__(self, *args, **kwargs):
        logger.debug('epix10ka_raw_2_0_1.__init__')
        eb.epix_base.__init__(self, *args, **kwargs)
        self._seg_geo = eb.sgs.Create(segname='EPIX10KA:V1')
        self._data_bit_mask = eb.M14 # for epix10ka data
        self._data_gain_bit = eb.B14
        self._gain_bit_shift = 9
        self._gains_def = (16.4, 5.466, 0.164) # epix10ka ADU/keV H:M:L = 1 : 1/3 : 1/100
        self._path_geo_default = 'pscalib/geometry/data/geometry-def-epix10kaquad.data'


    def _cbits_config_segment(self, cob):
        """cob=det.raw._seg_configs()[<seg-ind>].config - segment configuration object"""
        return eb.cbits_config_epix10ka(cob, shape=(352, 384))

    
    def raw(self, evt, copy=True) -> Array3d:
        return eb.epix_base.raw(self, evt, copy=copy)
        #return self.raw_v1(evt, copy=copy)


    def raw_v1(self, evt, copy=True) -> Array3d:
        """reshafle ASICs"""

        print('XXX RE-IMPLEMENTED raw')

        if evt is None: return None
        if ut.is_true(evt is None, 'evt is None'): return None

        segs = self._segments(evt)    # dict = {seg_index: seg_obj}
        if ut.is_true(segs is None, 'self._segments(evt) is None'): return None

        if len(segs) == 1:
            ind = self._segment_numbers[0]
            return segs[ind].raw

        #print('XXX self._segment_numbers', self._segment_numbers)
        first_seg = segs[self._segment_numbers[0]].raw
        dtype = first_seg.dtype
        shape = (len(self._segment_numbers),) + first_seg.shape
        if self._raw_buf is None or self._raw_shape != shape or self._raw_dtype != dtype:
            self._raw_buf = np.empty(shape, dtype=dtype)
            self._raw_shape = shape
            self._raw_dtype = dtype
            self._shape_seg = first_seg.shape
            self._shape_asic = [int(i/2) for i in self._shape_seg]
            #print('XXX shape_asic:', self._shape_asic)

        for idx, seg_id in enumerate(self._segment_numbers):
            np.copyto(self._raw_buf[idx], segs[seg_id].raw, casting='no')
            #print(info_ndarr(segs[seg_id].raw, f'segs[{seg_id}].raw'))
            seg = segs[seg_id].raw
            r,c = self._shape_asic
            asics = seg[:r,:c], seg[r:,:c], seg[r:,c:], seg[:r,c:]
            seg1 = np.vstack((np.hstack((np.flipud(np.fliplr(asics[2])),
                                    np.flipud(np.fliplr(asics[1])))),
                                    np.hstack((asics[3],asics[0]))))
            print(info_ndarr(seg1, 'reshaffled seg1'))
            np.copyto(self._raw_buf[idx], seg1, casting='no')

        arr = eb.reshape_to_3d(self._raw_buf)
        return arr.copy() if copy else arr


# calib is the same as in epix_base
#    def calib(self, evt, **kwa) -> Array3d:
#        logger.debug('epix10ka_raw_2_0_1.calib')
#        return eb.calib_epix10ka_any(self, evt, **kwa)


# MOVED TO epix_base
#    def _gain_range_index(self, evt, **kwa):
#        return eb.map_gain_range_index(self, evt, **kwa)


# MOVED TO epix_base
#    def _cbits_config_and_data_detector(self, evt=None):
#        return eb.cbits_config_and_data_detector_epix10ka(self, evt)


    def _array(self, evt) -> Array2d:
        f = None
        segs = self._segments(evt)
        if segs is None:
            pass
        else:
            nsegs = len(segs)
            if nsegs==4:
                nx = segs[0].raw.shape[1]
                ny = segs[0].raw.shape[0]
                f = np.zeros((ny*2,nx*2), dtype=segs[0].raw.dtype)
                xa = [nx, 0, nx, 0]
                ya = [ny, ny, 0, 0]
                for i in range(4):
                    x = xa[i]
                    y = ya[i]
                    f[y:y+ny,x:x+nx] = segs[i].raw & 0x3fff
        return f

#  Old detType for epix10ka
epix_raw_2_0_1 = epix10ka_raw_2_0_1

# EOF
