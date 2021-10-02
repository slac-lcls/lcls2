"""Data access VERSIONS
   for composite detectors made of epix10ka segments/panels.
"""
import numpy as np
from amitypes import Array2d, Array3d
import psana.detector.epix_base as eb
import logging
logger = logging.getLogger(__name__)

class epix10k_raw_0_0_1(eb.epix_base):
    def __init__(self, *args, **kwargs):
        logger.debug('%s.__init__' % self.__class__.__name__)
        eb.epix_base.__init__(self, *args, **kwargs)
        self.seg_geo = eb.sgs.Create(segname='EPIX10KA:V1')


class epix10ka_raw_2_0_1(eb.epix_base):
    def __init__(self, *args, **kwargs):
        logger.debug('epix10ka_raw_2_0_1.__init__')
        eb.epix_base.__init__(self, *args, **kwargs)
        self.seg_geo = eb.sgs.Create(segname='EPIX10KA:V1')
        self._data_bit_mask = eb.M14 # for epix10ka data


    def calib(self, evt, **kwa) -> Array3d:
        logger.debug('epix10ka_raw_2_0_1.calib')
        return eb.calib_epix10ka_any(self, evt, **kwa)


    def _gain_range_index(self, evt, **kwa):
        """Returns array (shaped as raw) per pixel gain range index or None."""
        return eb.map_gain_range_index(self, evt, **kwa)


    def _cbits_config_segment(self, cob):
        """cob=det.raw._seg_configs()[<seg-ind>].config - segment configuration object"""
        return eb.cbits_config_epix10ka(cob, shape=(352, 384))


    def _cbits_config_and_data_detector(self, evt=None):
        return eb.cbits_config_and_data_detector_epix10ka(self, evt)


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
