"""Data access methods common for all AREA DETECTORS.
"""

from psana.detector.detector_impl import DetectorImpl

import logging
logger = logging.getLogger(__name__)

import numpy as np

from psana.pscalib.geometry.GeometryAccess import GeometryAccess #, img_from_pixel_arrays
from psana.pyalgos.generic.NDArrUtils import info_ndarr, reshape_to_3d # print_ndarr,shape_as_2d, shape_as_3d, reshape_to_2d
from psana.detector.UtilsAreaDetector import dict_from_arr3d, arr3d_from_dict,\
        img_from_pixel_arrays, statistics_of_pixel_arrays, img_multipixel_max, img_multipixel_mean,\
        img_interpolated, init_interpolation_parameters, statistics_of_holes, fill_holes
import psana.detector.UtilsEpix10ka as ue

#----

class AreaDetector(DetectorImpl):

    def __init__(self, *args, **kwargs):
        logger.debug('AreaDetector.__init__') #  self.__class__.__name__
        DetectorImpl.__init__(self, *args, **kwargs)
        # caching
        self.geo = None
        self.pix_rc = None, None
        self.pix_xyz = None, None, None
        self.interpol_pars = None
        self._pedestals = None
        self._gain = None
        self._rms = None
        self._status = None
        self._common_mode = None
        #logger.info('XXX dir(self):\n' + str(dir(self)))
        #logger.info('XXX self._segments:\n' + str(self._segments))


    # example of some possible common behavior
    def _common_mode(self, *args, **kwargs):
        """
        """
        logger.debug('in %s._common_mode' % self.__class__.__name__)
        pass


    def raw(self,evt):
        """
        Returns dense 3-d numpy array of segment data
        from dict self._segments(evt)

        Parameters
        ----------
        evt: event
            psana event object, ex. run.events().next().

        Returns
        -------
        raw data: np.array, ndim=3, shape: as data
        """
        segs = self._segments(evt)
        if segs is None:
            logger.warning('self._segments(evt) is None')
            return None
        return arr3d_from_dict({k:v.raw for k,v in segs.items()})


    def segments(self,evt) :
        """ Returns dense 1-d numpy array of segment indexes.
        from dict self._segments(evt)    
        """
        segs = self._segments(evt)
        if segs is None: 
            logger.warning('self._segments(evt) is None')
            return None
        return np.array(sorted(segs.keys()), dtype=np.uint16)


    def det_calibconst(self):
        logger.debug('AreaDetector.det_calibconst')
        cc = self._calibconst
        if cc is None:
            logger.warning('self._calibconst is None')
            return None
        return cc


    def calibcons_and_meta_for_ctype(self, ctype='pedestals'):
        logger.debug('AreaDetector.calibcons_and_meta_for_ctype(ctype="%s")'%ctype)
        cc = self.det_calibconst()
        if cc is None: return None
        cons_and_meta = cc.get(ctype, None)
        if cons_and_meta is None:
            logger.warning('calibconst["%s"] is None'%ctype)
            return None, None
        return cons_and_meta


    def cached_array(self, p, ctype='pedestals'):
        """cached array
        """
        if p is None: p = self.calibcons_and_meta_for_ctype(ctype)[0] # 0-data/1-metadata
        return p

    def pedestals(self): return self.cached_array(self._pedestals, 'pedestals')
    def gain(self):      return self.cached_array(self._gain, 'pixel_gain')
    def rms(self):       return self.cached_array(self._rms, 'pixel_rms')
    def status(self):    return self.cached_array(self._status, 'pixel_status')

    def common_mode(self): return None # self.cached_array(self._common_mode, 'common_mode')


    def det_geotxt_and_meta(self):
        logger.debug('AreaDetector.det_geotxt_and_meta')
        cc = self.det_calibconst()
        if cc is None: return None
        geotxt_and_meta = cc.get('geometry', None)
        if geotxt_and_meta is None:
            logger.warning('calibconst[geometry] is None')
            return None, None
        return geotxt_and_meta


    def det_geo(self):
        """
        """
        if self.geo is None:
            geotxt, meta = self.det_geotxt_and_meta()
            if geotxt is None:
                logger.warning('det_geo geotxt is None')
                return None            
            self.geo = GeometryAccess()
            self.geo.load_pars_from_str(geotxt)
        return self.geo
        

    def pixel_coord_indexes(self, **kwa):
        """
        """
        logger.debug('AreaDetector.pixel_coord_indexes')
        geo = self.det_geo()
        if geo is None:
            logger.warning('geo is None')
            return None
            
        return geo.get_pixel_coord_indexes(\
            pix_scale_size_um  = kwa.get('pix_scale_size_um',None),\
            xy0_off_pix        = kwa.get('xy0_off_pix',None),\
            do_tilt            = kwa.get('do_tilt',True),\
            cframe             = kwa.get('cframe',0))


    def pixel_coords(self, **kwa):
        """
        """
        logger.debug('AreaDetector.pixel_coords')
        geo = self.det_geo()
        if geo is None:
            logger.warning('geo is None')
            return None
        #return geo.get_pixel_xy_at_z(self, zplane=None, oname=None, oindex=0, do_tilt=True, cframe=0)
        return geo.get_pixel_coords(\
            do_tilt            = kwa.get('do_tilt',True),\
            cframe             = kwa.get('cframe',0))


    def cached_pixel_coord_indexes(self, evt, **kwa):
        """
        """
        logger.debug('AreaDetector.cached_pixel_coord_indexes')

        resp = self.pixel_coord_indexes(**kwa)
        if resp is None: return None

        # PRESERVE PIXEL INDEXES FOR USED SEGMENTS ONLY
        segs = self.segments(evt)
        if segs is None: return None
        logger.debug(info_ndarr(segs, 'preserve pixel indices for segments '))

        rows, cols = self.pix_rc = [reshape_to_3d(a)[segs,:,:] for a in resp]
        #self.pix_rc = [dict_from_arr3d(reshape_to_3d(v)) for v in resp]

        s = 'evaluate_pixel_coord_indexes:'
        for i,a in enumerate(self.pix_rc): s += info_ndarr(a, '\n  %s '%('rows','cols')[i], last=3)
        logger.info(s)

        mapmode = kwa.get('mapmode',2)
        if mapmode <4:
          self.img_entries, self.dmulti_pix_to_img_idx, self.dmulti_imgidx_numentries=\
            statistics_of_pixel_arrays(rows, cols)

        if mapmode==4:
            rsp = self.pixel_coords(**kwa)
            if rsp is None: return None
            x,y,z = self.pix_xyz = [reshape_to_3d(a)[segs,:,:] for a in rsp]
            self.interpol_pars = init_interpolation_parameters(rows, cols, x, y)

        if mapmode <4 and kwa.get('fillholes',True):
            self.img_pix_ascend_ind, self.img_holes, self.hole_rows, self.hole_cols, self.hole_inds1d =\
               statistics_of_holes(rows, cols, **kwa)

        # TBD parameters for image interpolation
        if False:
            t0_sec = time()
            self.imgind_to_seg_row_col = image_of_pixel_seg_row_col(img_pix_ascend_ind, arr_shape)
            logger.debug('statistics_of_holes.imgind_to_seg_row_col time (sec) = %.6f' % (time()-t0_sec)) # 47ms
            logger.debug(info_ndarr(self.imgind_to_seg_row_col, ' imgind_to_seg_row_col '))
    
            if False:
                s = ' imgind_to_seg_row_col '
                # (n,352,384)
                first = (352+5)*384 + 380
                for i in range(first,first+10): s += '\n    s:%02d r:%03d c:%03d' % tuple(imgind_to_seg_row_col[i])
                logger.debug(s)


    def calib(self,evt):
        """
        Create calibrated data array.
        """
        logger.info('AreaDetector.calib')
        #return self.raw(evt)
        return ue.calib_epix10ka_any(self, evt)


    def image(self, evt, nda=None, **kwa):
        """
        Create 2-d image.

        Parameters
        ----------
        evt: event
            psana event object, ex. run.events().next().

        mapmode: int, optional, default: 2
            control on overlapping pixels on image map.
            0/1/2/3/4: statistics of entries / last / max / mean pixel intensity / interpolated (TBD) - ascending data index.

        fillholes: bool, optional, default: True
            control on map bins inside the panel with 0 entries from data.
            True/False: fill empty bin with minimal intensity of four neares neighbors/ do not fill.

        vbase: float, optional, default: 0
            value substituted for all image map bins without entry from data.  

        Returns
        -------
        image: np.array, ndim=2
        
        """
        logger.debug('in AreaDretector.image')
        if any(v is None for v in self.pix_rc):
            self.cached_pixel_coord_indexes(evt, **kwa)
            if any(v is None for v in self.pix_rc): return None

        vbase     = kwa.get('vbase',0)
        mapmode   = kwa.get('mapmode',2)
        fillholes = kwa.get('fillholes',True)

        if mapmode==0: return self.img_entries

        data = self.calib(evt) if nda is None else nda
        if data is None:
            logger.warning('AreaDetector.image calib returns None')
            return None
            
        #logger.debug(info_ndarr(data, 'data ', last=3))

        rows, cols = self.pix_rc
        img = img_from_pixel_arrays(rows, cols, weight=data, vbase=vbase) # mapmode==1
        if   mapmode==2: img_multipixel_max(img, data, self.dmulti_pix_to_img_idx)
        elif mapmode==3: img_multipixel_mean(img, data, self.dmulti_pix_to_img_idx, self.dmulti_imgidx_numentries)

        if mapmode<4 and fillholes: fill_holes(img, self.hole_rows, self.hole_cols)

        return img if mapmode<4 else\
               img_interpolated(data, self.interpol_pars) if mapmode==4 else\
               self.img_entries

#----

if __name__ == "__main__":
    import sys
    sys.exit('See example in test_%s if available...' % sys.argv[0].split('/')[-1])

#----
