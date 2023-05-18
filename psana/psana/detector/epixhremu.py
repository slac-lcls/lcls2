"""epixhremu - emulated data epixhr single- and 20- panel detector prototype
   Created on 2023-04-26
"""

import os
import numpy as np
from amitypes import Array2d
import psana.detector.epix_base as eb
import logging
from psana.detector.detector_impl import DetectorImpl
logger = logging.getLogger(__name__)


class epixhremu_raw_0_0_1(eb.epix_base):
    def __init__(self, *args, **kwargs):
        logger.debug('epixhremu_raw_0_0_1.__init__')
        eb.epix_base.__init__(self, *args, **kwargs)
        self._seg_geo = eb.sgs.Create(segname='EPIXHR1X4:V1')
        self._data_bit_mask = eb.M15 # for epixhremu data
        self._data_gain_bit = eb.B15
        self._gain_bit_shift = 10
        self._gains_def = (41.0, 13.7, 0.512) # epixhremu ADU/keV H:M:L = 1 : 1/3 : 1/80
        #self._path_geo_default = 'pscalib/geometry/data/geometry-def-epixhremu.data'
        self._path_geo_default = 'pscalib/geometry/data/geometry-def-epixhr1x4-20.data'


    def _array(self, evt) -> Array2d:
        f = None
        segs = self._segments(evt)
        if segs is None:
            pass
        else:
            nx = segs[0].raw.shape[1]
            ny = segs[0].raw.shape[0]
            f = segs[0].raw & self._data_bit_mask # 0x7fff
        return f


    def _cbits_config_segment(self, cob):
        """cob=det.raw._seg_configs()[<seg-ind>].config"""
        logger.debug('epixhremu._cbits_config_segment')
        return eb.cbits_config_epixhr1x4(cob, shape=(144, 768))


#    def _det_geotxt_default(self):
#        """returns (str) default geometry constants from lcls2/psana/psana/pscalib/geometry/data/geometry-def-*.data"""
#        dir_detector = os.path.abspath(os.path.dirname(__file__))
#        fname = '%s/../pscalib/geometry/data/geometry-def-epixhremu.data' % dir_detector
#        logger.warning('_det_geotxt_default from file: %s' % fname)
#        return eb.ut.load_textfile(fname)


#    def _segment_numbers(self, evt):
#        """OVERRIDE THIS METHOD TO FIX ISSUE in exp=tstx00417,run=277, which returns [3,]"""
#        segnums = eb.epix_base._segment_numbers(self, evt)
#        if segnums is not None and len(segnums)==1 and segnums[0]==3:
#           logger.warning('OWERRIDED epixhremu._segment_numbers fixes %s to [0]' % str(segnums))
#           segnums=[0]
#        return segnums


    def _config_object(self):
        """overrides epix_base._config_object() and returns fake configuration for epixhremu"""
        logger.debug('FAKE CONFIG OBJECTepixhremu._config_object._segment_indices():', self._segment_indices())
        # segment_indices(): [3] for 277 or [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] run 276
        return {i:fake_seg_config for i in self._segment_indices()}


class _fake_seg_config_content:
    trbit = [0,0,0,0]
    asicPixelConfig = np.zeros((144, 768), dtype=np.uint8)


class _fake_seg_config:
    config = _fake_seg_config_content()


fake_seg_config = _fake_seg_config()

# EOF
