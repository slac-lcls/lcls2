from amitypes import Array3d

import logging
logger = logging.getLogger(__name__)
from amitypes import Array1d, Array2d, Array3d
import psana2.detector.UtilsJungfrau as uj
import psana2.detector.UtilsCalib as uc
import psana2.detector.areadetector as ad
AreaDetectorRaw = ad.AreaDetectorRaw

class jungfrau_raw_0_1_0(AreaDetectorRaw):
    def __init__(self, *args, **kwa):
        logger.debug('jungfrau_raw_0_1_0.__init__')
        AreaDetectorRaw.__init__(self, *args, **kwa)

        #self._segment_numbers = (0,3,4,5,6,8,9)
        #print('XXX self._segment_numbers', self._segment_numbers)
        segnum_max = max(self._segment_numbers)
        nsegs = uj.jungfrau_segments_tot(segnum_max) # 1,2,8, or 32
        sMpix = {1:'05M', 2:'1M', 8:'4M', 32:'16M'}.get(nsegs, 32)

        self._path_geo_default = 'pscalib/geometry/data/geometry-def-jungfrau%s.data' % sMpix
        self._seg_geo = ad.sgs.Create(segname='JUNGFRAU:V2')

        self._gain_modes = ('g0', 'g1', 'g2')
        self._data_bit_mask = 0x3fff

#        self._gains_def = (-100.7, -21.3, -100.7) # ADU/Pulser
#        self._gain_modes = ('DYNAMIC', 'FORCE_SWITCH_G1', 'FORCE_SWITCH_G2')

#    def _raw_random(self, evt, mu=0, sigma=10):
#        """ FOR DEBUGGING ONLY !!!
#            add random values to apread the same per event array
#        """
#        a = AreaDetectorRaw.raw(self, evt)
#        arrand = mu + sigma*ad.np.random.standard_normal(size=a.shape) #.astype(dtype=np.float64)
#        return (a+arrand).astype(dtype=a.dtype)

#    def _config_object(self):
#       """overrides epix_base._config_object()"""
#       return AreaDetectorRaw._config_object()

    def _seg_configs_user(self):
        """ list of det.raw._seg_configs()[ind].config.user for _segment_indices"""
        inds = self._segment_numbers # self.sorted_segment_inds()
        scfgs = [self._seg_configs()[i].config.user for i in inds]
        logger.debug('_seg_configs_user test:'\
                    +'\n    det.raw._segment_indices(): %s' % str(inds)\
                    +'\n    per-segment gainMode: %s' % str([str(cfg.gainMode.value) for cfg in scfgs])\
                    +'\n    per-segment gain0: %s' % str([str(cfg.gain0.value) for cfg in scfgs]))
        return scfgs

    def _detector_name_long_short(self):
        longname = self._uniqueid
        return longname, uc.detector_name_short(longname, maxsize=uj.MAX_DETNAME_SIZE)

    def calib(self, evt, **kwa) -> Array2d:
        #if ad.is_none(self.raw(evt), 'raw is None', logger_method=logger.debug): return None
        if self.raw(evt) is None: return None
        return uj.calib_jungfrau(self, evt, **kwa)

class jungfrau_raw_0_2_0(jungfrau_raw_0_1_0):
    def num_hot_pixels(self, evt):
        n_hot_pixels = 0
        segs = self._segments(evt)
        if segs is None:
            return None
        for _,seg in segs.items():
            n_hot_pixels += seg.numHotPixels
        return n_hot_pixels

    def hot_pixel_thresh(self, evt):
        hp_tresh = 0
        segs = self._segments(evt)
        if segs is None:
            return None
        for _,seg in segs.items():
            hp_tresh = seg.hotPixelThresh
            break
        return hp_tresh
# EOF
