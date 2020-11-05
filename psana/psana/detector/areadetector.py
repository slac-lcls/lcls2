"""Data access methods common for all AREA DETECTORS.
"""

from psana.detector.detector_impl import DetectorImpl

import logging
logger = logging.getLogger(__name__)

from psana.pscalib.geometry.GeometryAccess import GeometryAccess, img_from_pixel_arrays
from psana.pyalgos.generic.NDArrUtils import info_ndarr # print_ndarr

#----

class AreaDetector(DetectorImpl):

    def __init__(self, *args, **kwargs):
        logger.debug('AreaDetector.__init__') #  self.__class__.__name__
        DetectorImpl.__init__(self, *args, **kwargs)
        self.geo = None

    # example of some possible common behavior
    def _common_mode(self, *args, **kwargs):
        logger.debug('in %s._common_mode' % self.__class__.__name__)
        pass


    def raw(self,evt):
        data = {}
        segs = self._segments(evt)
        if segs is None: return None
        for segment,val in segs.items():
            data[segment]=val.raw
        return data


    def det_calibconst(self):
        logger.debug('AreaDetector.det_calibconst')
        #print('XXX dir(self):', dir(self))
        cc = self._calibconst
        if cc is None:
            logger.warning('self._calibconst is None')
            return None
        return cc


    def det_geotxt_and_meta(self):
        logger.debug('AreaDetector.det_geometry_txt')
        cc = self.det_calibconst()
        if cc is None: return None
        geotxt_and_meta = cc.get('geometry', None)
        if geotxt_and_meta is None:
            logger.warning('calibconst[geometry] is None')
            return None, None
        return geotxt_and_meta


    def det_geo(self):
        if self.geo is None:
            geotxt, meta = self.det_geotxt_and_meta()
            if geotxt is None:
                logger.warning('det_geo geotxt is None')
                return None            
            self.geo = GeometryAccess()
            self.geo.load_pars_from_str(geotxt)
        return self.geo
        

    def calib(self,evt):
        logger.debug('AreaDetector.calib TBD currently returns raw')
        print('XXX dir(self):', dir(self))
        #cc = self.det_calibconst()
        #if cc is None: return None
        #peds, peds_meta = cc.get('pedestals', None)
        #...
        return self.raw(evt)


    def image(self, evt, **kwa):
        logger.debug('AreaDetector.image TBD')
        #print('XXX dir(self):', dir(self))
        geo = self.det_geo()
        if geo is None:
            logger.warning('image geo is None')
            return None
            
        #geo.print_list_of_geos()
        X, Y, Z = geo.get_pixel_coords(cframe=kwa.get('cframe',0))

        logger.info(info_ndarr(X, 'X'))
        logger.info(info_ndarr(Y, 'Y'))
        logger.info(info_ndarr(Z, 'Z'))

        #img = img_from_pixel_arrays

        calib = self.calib(evt)
        #logger.info(info_ndarr(calib, 'calib'))
        logger.info('XXX calib:' + str(calib))
            
        return None

#----

if __name__ == "__main__":
    import sys
    sys.exit('See example in test_%s if available...' % sys.argv[0].split('/')[-1])

#----
