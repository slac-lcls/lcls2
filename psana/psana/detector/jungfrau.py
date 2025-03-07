from amitypes import Array3d

import logging
logger = logging.getLogger(__name__)

import psana.detector.areadetector as ad
AreaDetectorRaw = ad.AreaDetectorRaw

class jungfrau_raw_0_1_0(AreaDetectorRaw):
    def __init__(self, *args, **kwa):
        logger.debug('jungfrau_raw_0_1_0.__init__')
        AreaDetectorRaw.__init__(self, *args, **kwa)
        #self._sorteed_segment_inds = (0,1,2,3,4,5,6,7)
        self._path_geo_default = 'pscalib/geometry/data/geometry-def-jungfrau4M.data'
        self._seg_geo = ad.sgs.Create(segname='JUNGFRAU:V2')
        flds = self._uniqueid.split('_')

        nsegs = None if flds is None else len(flds)-1
        sMpix = {1:'05M', 2:'1M', 3:'4M'}.get(nsegs, None)

        #self._path_geo_default = 'pscalib/geometry/data/geometry-def-jungfrau%s.data' % sMpix
        #self._path_geo_default = 'pscalib/geometry/data/geometry-def-jungfrau1M.data'
        #self._segment_numbers = (0,1,2,3,4,5,6,7)

#        self._gains_def = (-100.7, -21.3, -100.7) # ADU/Pulser
#        self._gain_modes = ('FH', 'FM', 'FL')

#    def raw(self, evt, mu=0, sigma=10):
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

    def _segment_ids(self):
        """returns list of segment ids"""
        #print('TBD _segment_ids for longname: %s' % self._uniqueid)
        return self._uniqueid.split('_')[1:]

# EOF
