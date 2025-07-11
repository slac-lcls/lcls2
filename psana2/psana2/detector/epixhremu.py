"""epixhremu - emulated data epixhr single- and 20- panel detector prototype
   Created on 2023-04-26
"""

import os
import numpy as np
from amitypes import Array2d, Array3d
import psana2.detector.epix_base as eb
import psana2.detector.areadetector as ad
import logging
from psana2.detector.detector_impl import DetectorImpl
logger = logging.getLogger(__name__)


class epixhremu_raw_0_0_1(eb.epix_base):
    def __init__(self, *args, **kwargs):
        logger.debug('epixhremu_raw_0_0_1.__init__')
        eb.epix_base.__init__(self, *args, **kwargs)
        self._seg_geo = ad.sgs.Create(segname='EPIXHR1X4:V1')
        self._data_bit_mask = eb.M14 # for epixhr2x2 data on 2023-10-30 Dionisio - HR has 14 data bits.
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
            first_key = next(iter(segs))
            nx = segs[first_key].raw.shape[1]
            ny = segs[first_key].raw.shape[0]
            f = segs[first_key].raw & self._data_bit_mask # 0x7fff
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


import libpressio as lp
import json

#class epixhremu_fex_0_0_1(DetectorImpl):
class epixhremu_fex_0_0_1(ad.AreaDetector):
    def __init__(self, *args, **kwargs):
        super(epixhremu_fex_0_0_1, self).__init__(*args)
        self._seg_geo = ad.sgs.Create(segname='EPIXHR1X4:V1')

        # Expect config to be the same for all segments, so pick first one
        for config in self._seg_configs().values(): break
        comp_config = json.loads(config.config.compressor_json)

        # Instantiate the compressor/decompressor
        self._compressor = lp.PressioCompressor.from_config(comp_config)

        # Hard code shape and data type because they're properies of the detector
        self._decompressed = np.empty_like(np.ndarray(shape=(144,192*4), dtype=np.float32))
        self._data_bit_mask = eb.M14

    def _calib(self, evt, **kwargs) -> Array2d:
        segs = self._segments(evt)
        dec = None
        if segs is not None:
            for data in segs.values(): break # Is there only 1?  What if there are more?
            dec = self._compressor.decode(data.fex, self._decompressed)
            #print(f'*** dec is a {type(dec)} of len {len(dec)}, dtype {dec.dtype}, shape {dec.shape}, ndim {dec.ndim}, size {dec.size}')

        return dec

    def calib(self, evt, **kwargs) -> Array3d:
        """decode per panel dict of _segments and return array of
           shape=(<number-of-def-panels>, <panel-number-of-rows>, <panel-number-of-cols>)
        """
        segs = self._segments(evt)
        return None if segs is None else\
               np.stack([self._compressor.decode(segs[k].fex, self._decompressed) for k in sorted(segs.keys())])

    def image(self, evt, **kwargs) -> Array2d: # value_for_missing_segments=1
        #ad.AreaDetector.image(evt, nda=self.calib(evt), **kwargs)
        return ad.AreaDetector.image(self, evt, **kwargs)

        #cc = self._calibconst # defined in DetectorImpl
        #print('cc[geometry] meta:\n%s' % str(cc['geometry'][1]))
        #print('cc.keys:\n%s' % str(cc.keys()))
        #segnums = self._segment_numbers = self._sorted_segment_inds # [2, 5, 11] - for run 286 or all 20 for 287
        #print('  segnums: %s' % str(segnums))

# EOF
