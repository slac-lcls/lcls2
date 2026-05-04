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

class epix10ka_config_3_0_0(DetectorImpl):
    def __init__(self, *args, **kwargs):
        logger.debug('XXX epix10ka_config_3_0_0')
        super().__init__(*args)
        #super(epix10kaquad_config_2_0_0, self).__init__(*args)


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
        """cob=det.raw._seg_configs()[<seg-ind>].config - segment configuration object, where self=det.raw"""
        return eb.cbits_config_epix10ka(cob, shape=(352, 384))


    def raw(self, evt, copy=True) -> Array3d:
        """TEST overrides lcls2/psana/psana/detector/areadetector.py AreaDetectorRaw.raw"""
        logger.debug('XXX epix10ka_raw_2_0_1.raw')
        return eb.epix_base.raw(self, evt, copy=copy)


    def calib(self, evt, **kwa) -> Array3d:
        """TEST overrides lcls2/psana/psana/detector/epix_base.py epix_base.calib"""
        logger.debug('XXX epix10ka_raw_2_0_1.calib')
        return eb.epix_base.calib(self, evt, **kwa)
#        return eb.ue.calib_epix10ka_v02(self, evt, **kwa)
#        #return eb.ue.calib_epix10ka_v02(self, evt, nda_raw=self.raw(evt), **kwa)


# MOVED TO epix_base
#    def _gain_range_index(self, evt, **kwa):
#        return eb.map_gain_range_index(self, evt, **kwa)

#    def _cbits_config_and_data_detector(self, evt=None):
#        return eb.cbits_config_and_data_detector_epix10ka(self, evt)


    def _array(self, evt) -> Array2d:
        f = None
        segs = self._segments(evt)
        if segs is not None:
            nsegs = len(segs)
            if nsegs==4:
                nr,nc = segs[0].raw.shape[-2:]
                f = np.zeros((nr*2,nc*2), dtype=segs[0].raw.dtype)
                ca = [nc, 0, nc, 0]
                ra = [nr, nr, 0, 0]
                for i in range(4):
                    c = ca[i]
                    r = ra[i]
                    f[r:r+nr,c:c+nc] = segs[i].raw & 0x3fff
        return f


    def _array_v2(self, evt) -> Array2d:
        """easy stacking 4 panels to imaging array"""
        segs = self._segments(evt)
        if segs is None or len(segs) !=4: return None
        return np.vstack((np.hstack((segs[3].raw, segs[2].raw)),\
                          np.hstack((segs[1].raw, segs[0].raw)))) & 0x3fff


class epix10ka_raw_3_0_1(epix10ka_raw_2_0_1):
    def __init__(self, *args, **kwargs):
        logger.debug('epix10ka_raw_3_0_1.__init__')
        epix10ka_raw_2_0_1.__init__(self, *args, **kwargs)

#    def _cbits_config_segment(self, cob):
#        """cob=det.raw._seg_configs()[<seg-ind>].config - segment configuration object, where self=det.raw"""
#        return eb.cbits_config_epix10ka(cob, shape=(352, 384))

    def _cbits_config_segment(self, cob):
        """cob=det.raw._seg_configs()[<seg-ind>].config - segment configuration object, where self=det.raw"""
        print('XXX _cbits_config_segment dir(cob):', dir(cob))
        print(info_ndarr(cob.asicPixelConfig, 'XXX epix10ka_raw_3_0_1._cbits_config_segment'))
        return cob.asicPixelConfig # expected panel shape:(352, 384) dtype:uint8
        #return eb.cbits_config_epix10ka_v02(cob) # shape=(352, 384)

    def raw(self, evt, copy=True) -> Array3d:
        """TEST overrides lcls2/psana/psana/detector/areadetector.py AreaDetectorRaw.raw"""
        logger.debug('XXX epix10ka_raw_3_0_1.raw')
        return eb.epix_base.raw(self, evt, copy=copy)

    def calib(self, evt, **kwa) -> Array3d:
        """TEST overrides lcls2/psana/psana/detector/epix_base.py epix_base.calib"""
        print('XXX epix10ka_raw_3_0_1.calib')
        return eb.ue.calib_epix10ka_v02(self, evt, **kwa)
        #return eb.ue.calib_epix10ka_v02(self, evt, nda_raw=self.raw(evt), **kwa)


#  Old detType for epix10ka
epix_raw_2_0_1 = epix10ka_raw_2_0_1
epix10ka_raw_3_0_1 = epix10ka_raw_2_0_1

# EOF
