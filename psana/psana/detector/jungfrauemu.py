from amitypes import Array3d

import logging
logger = logging.getLogger(__name__)

import psana.detector.areadetector as ad
#from psana.detector.areadetector import sgs, AreaDetectorRaw
#import psana.detector.UtilsMask as um #import merge_status

class jungfrauemu_raw_0_1_0(ad.AreaDetectorRaw):
    def __init__(self, *args, **kwa):
        logger.debug('jungfrauemu_raw_0_1_0.__init__')
        ad.AreaDetectorRaw.__init__(self, *args, **kwa)
        self._sorteed_segment_inds = (0,1,2,3,4,5,6,7)
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

    def raw(self, evt, mu=0, sigma=10):
        """ FOR DEBUGGING ONLY !!!
            add random values to apread the same per event array
        """
        a = ad.AreaDetectorRaw.raw(self, evt)
        arrand = mu + sigma*ad.np.random.standard_normal(size=a.shape) #.astype(dtype=np.float64)
        return (a+arrand).astype(dtype=a.dtype)

    def _config_object(self):
        """overrides epix_base._config_object() and returns fake configuration for epixhremu"""
        print('TBD _config_object')
        #logger.debug('FAKE CONFIG OBJECTepixhremu._config_object._segment_indices(): %s', self._segment_indices())
        #print('dir(self):', dir(self))
        #print('dir(self._seg_configs):', dir(self._seg_configs))
        #print('dir(self._config_object):', dir(self._config_object))
        return None

    def _segment_ids(self):
        """returns list of segment ids"""
        longname = self._uniqueid
        print('TBD _segment_ids for longname: %s' % longname)
        return longname.split('_')[1:]

# EOF
