"""Data access methods common for all AREA DETECTORS.
"""

from psana.detector.detector_impl import DetectorImpl

import logging
logger = logging.getLogger(__name__)

import numpy as np

from psana.pscalib.geometry.GeometryAccess import GeometryAccess #, img_from_pixel_arrays
from psana.pyalgos.generic.NDArrUtils import info_ndarr, reshape_to_3d # print_ndarr,shape_as_2d, shape_as_3d, reshape_to_2d
from psana.detector.UtilsAreaDetector import dict_from_arr3d # arr3d_from_dict

#----

class AreaDetector(DetectorImpl):

    def __init__(self, *args, **kwargs):
        logger.debug('AreaDetector.__init__') #  self.__class__.__name__
        DetectorImpl.__init__(self, *args, **kwargs)
        # caching
        self.geo = None
        self.inds_rc = None, None

        #logger.info('XXX dir(self):\n' + str(dir(self)))
        #logger.info('XXX self._segments:\n' + str(self._segments))


    # example of some possible common behavior
    def _common_mode(self, *args, **kwargs):
        logger.debug('in %s._common_mode' % self.__class__.__name__)
        pass


    def raw(self,evt):
        data = {}
        segs = self._segments(evt)
        if segs is None: return None
        for k,v in segs.items():
            data[k]=v.raw
        return data


    def det_calibconst(self):
        logger.debug('AreaDetector.det_calibconst')
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
        

    def evaluate_pixel_coord_indexes(self, **kwa):
        """ uses from kwa
        """
        logger.debug('AreaDetector.det_pixel_coords')
        #print('XXX dir(self):', dir(self))
        geo = self.det_geo()
        if geo is None:
            logger.warning('geo is None')
            return
            
        #geo.print_list_of_geos()
        #resp = geo.get_pixel_coords(cframe=kwa.get('cframe',0))

        resp = rows, cols = geo.get_pixel_coord_indexes(\
            pix_scale_size_um  = kwa.get('pix_scale_size_um',None),\
            xy0_off_pix        = kwa.get('xy0_off_pix',None),\
            do_tilt            = kwa.get('do_tilt',True),\
            cframe             = kwa.get('cframe',0))

        if any(v is None for v in resp): return

        s = 'responce of geo.get_pixel_coords:'
        for i,v in enumerate(resp): s += info_ndarr(v, '\n  pix coordinate inds rc[%d]: '%i, last=3)
        logger.info(s)

        self.inds_rc = [dict_from_arr3d(reshape_to_3d(v)) for v in resp]

        s = 'evaluate_pixel_coords content of X:'
        for k,v in self.inds_rc[0].items(): s += info_ndarr(v, '\n  panel:%02d '%k, last=3)
        logger.info(s)


    def calib(self,evt):
        logger.warning('AreaDetector.calib TBD currently returns raw')
        #print('XXX dir(self):', dir(self))
        #cc = self.det_calibconst()
        #if cc is None: return None
        #peds, peds_meta = cc.get('pedestals', None)
        #...
        return self.raw(evt)


    def image(self, evt, **kwa):
        logger.debug('in AreaDretector.image')
        if any(v is None for v in self.inds_rc):
            self.evaluate_pixel_coord_indexes(**kwa)
            if any(v is None for v in self.inds_rc): return None

        dicdata = self.calib(evt)
        #logger.info(info_ndarr(calib, 'calib'))
        #logger.info('XXX calib:' + str(calib))
        if dicdata is None: return None
            
        s = 'AreaDretector.image content of dicdata:'
        for k,v in dicdata.items(): s += info_ndarr(v, '\n  panel:%02d '%k, last=3)
        logger.info(s)

        return img_from_pixel_dicts(self.inds_rc[0], self.inds_rc[1], weight=dicdata, vbase=1)
        #return img_from_pixel_dicts(self.inds_rc[0], self.inds_rc[1], weight=2.0, vbase=1)

#----

def img_from_pixel_dicts(rows, cols, weight=2.0, dtype=np.float32, vbase=0):
    """Returns image from rows, cols index arrays and associated weights W.
       Methods like matplotlib imshow(img) plot 2-d image array oriented as matrix(rows,cols).
    """
    assert isinstance(rows, dict)
    assert isinstance(cols, dict)
    assert (isinstance(weight, (dict,float)))

    rsize = 0
    csize = 0
    for k,v in rows.items():
        rsize = int(max(rsize, v.max()))
        csize = int(max(rsize, cols[k].max()))

    logger.debug('evaluated image shape = (%d, %d)' % (rsize, csize))

    img = np.ones((rsize+1,csize+1), dtype=dtype)*vbase if vbase else\
         np.zeros((rsize+1,csize+1), dtype=dtype)

    if isinstance(weight, float):
        for k,v in rows.items():
            rowsfl = v.flatten()
            colsfl = cols[k].flatten()
            img[rowsfl,colsfl] = weight
    else:
        for k,v in weight.items():
            rowsfl = rows[k].flatten()
            colsfl = cols[k].flatten()
            img[rowsfl,colsfl] = v.flatten()

    return img

#----

if __name__ == "__main__":
    import sys
    sys.exit('See example in test_%s if available...' % sys.argv[0].split('/')[-1])

#----
