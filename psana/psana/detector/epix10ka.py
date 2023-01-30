"""Data access VERSIONS
   for composite detectors made of epix10ka segments/panels.
"""
import numpy as np
from amitypes import Array2d, Array3d
import psana.detector.epix_base as eb
from psana.detector.detector_impl import DetectorImpl
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


class epix10ka_raw_2_0_1(eb.epix_base):
    def __init__(self, *args, **kwargs):
        logger.debug('epix10ka_raw_2_0_1.__init__')
        eb.epix_base.__init__(self, *args, **kwargs)
        self._seg_geo = eb.sgs.Create(segname='EPIX10KA:V1')
        self._data_bit_mask = eb.M14 # for epix10ka data
        self._data_gain_bit = eb.B14
        self._gain_bit_shift = 9
        self._gains_def = (16.4, 5.466, 0.164) # epix10ka ADU/keV H:M:L = 1 : 1/3 : 1/100


    def _cbits_config_segment(self, cob):
        """cob=det.raw._seg_configs()[<seg-ind>].config - segment configuration object"""
        return eb.cbits_config_epix10ka(cob, shape=(352, 384))


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
