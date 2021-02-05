"""
Data access methods common for all AREA DETECTORS
=================================================

Usage::

  from psana.detector.areadetector import AreaDetector

  o = AreaDetector(*args, **kwa) # enherits from DetectorImpl(*args, **kwa)

  a = o.raw(evt)
  a = o.segments(evt)
  a = o._det_calibconst()
  a = o._calibcons_and_meta_for_ctype(ctype='pedestals')
  a = o._cached_array(p, ctype='pedestals')
  a = o._pedestals()
  a = o._gain()
  a = o._rms()
  a = o._status()
  a = o._mask_calib()
  a = o._common_mode()
  a = o._det_geotxt_and_meta()
  a = o._det_geo()
  a = o._pixel_coord_indexes(pix_scale_size_um=None, xy0_off_pix=None, do_tilt=True, cframe=0, **kwa)
  a = o._pixel_coords(do_tilt=True, cframe=0, **kwa)
  a = o._cached_pixel_coord_indexes(evt, **kwa) # **kwa - the same as above

  a = o._shape_as_daq()
  a = o._number_of_segments_total()
  m = o._mask_default(dtype=DTYPE_MASK)
  m = o._mask_calib_or_default(dtype=DTYPE_MASK)
  m = o._mask_from_status()
  m = o._mask_edges(edge_rows=1, edge_cols=1, dtype=DTYPE_MASK, **kwa)
  m = o._mask(calib=False, status=False, edges=False, dtype=DTYPE_MASK, **kwa) # TBD: neighbors=False
  m = o._mask_comb(mbits=0o377, **kwa)

  a = o.calib(evt, cmpars=(7,2,100,10),\
                            mbits=0o7, mask=None, edge_rows=10, edge_cols=10, center_rows=5, center_cols=5)
  a = o.calib(evt, **kwa)
  a = o.image(self, evt, nda=None, **kwa)

2020-11-06 created by Mikhail Dubrovin
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
from psana.detector.UtilsMask import CC, DTYPE_MASK, DTYPE_STATUS, mask_edges, merge_masks

from amitypes import Array2d, Array3d

#----

class AreaDetector(DetectorImpl):

    def __init__(self, *args, **kwargs):
        logger.debug('AreaDetector.__init__') #  self.__class__.__name__
        DetectorImpl.__init__(self, *args, **kwargs)
        # caching
        self._geo_ = None
        self._pix_rc_ = None, None
        self._pix_xyz_ = None, None, None
        self._interpol_pars_ = None
        self._pedestals_ = None
        self._gain_ = None
        self._rms_ = None
        self._status_ = None
        self._common_mode_ = None
        self._mask_calib_ = None
        self._mask_ = None
        #logger.info('XXX dir(self):\n' + str(dir(self)))
        #logger.info('XXX self._segments:\n' + str(self._segments))


    def raw(self,evt) -> Array3d:
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


    def segments(self,evt):
        """ Returns dense 1-d numpy array of segment indexes.
        from dict self._segments(evt)    
        """
        segs = self._segments(evt)
        if segs is None: 
            logger.warning('self._segments(evt) is None')
            return None
        return np.array(sorted(segs.keys()), dtype=np.uint16)


    def _det_calibconst(self):
        logger.debug('AreaDetector._det_calibconst')
        cc = self._calibconst
        if cc is None:
            logger.warning('self._calibconst is None')
            return None
        return cc


    def _calibcons_and_meta_for_ctype(self, ctype='pedestals'):
        logger.debug('AreaDetector._calibcons_and_meta_for_ctype(ctype="%s")'%ctype)
        cc = self._det_calibconst()
        if cc is None: return None
        cons_and_meta = cc.get(ctype, None)
        if cons_and_meta is None:
            logger.warning('calibconst["%s"] is None'%ctype)
            return None, None
        return cons_and_meta


    def _cached_array(self, p, ctype='pedestals'):
        """cached array
        """
        if p is None: p = self._calibcons_and_meta_for_ctype(ctype)[0] # 0-data/1-metadata
        return p


    def _pedestals(self): return self._cached_array(self._pedestals_, 'pedestals')
    def _gain(self):      return self._cached_array(self._gain_, 'pixel_gain')
    def _rms(self):       return self._cached_array(self._rms_, 'pixel_rms')
    def _status(self):    return self._cached_array(self._status_, 'pixel_status')
    def _mask_calib(self):return self._cached_array(self._mask_calib_, 'pixel_mask')
    def _common_mode(self):return self._cached_array(self._common_mode_, 'common_mode')


    def _det_geotxt_and_meta(self):
        logger.debug('AreaDetector._det_geotxt_and_meta')
        cc = self._det_calibconst()
        if cc is None: return None
        geotxt_and_meta = cc.get('geometry', None)
        if geotxt_and_meta is None:
            logger.warning('calibconst[geometry] is None')
            return None, None
        return geotxt_and_meta


    def _det_geo(self):
        """
        """
        if self._geo_ is None:
            geotxt, meta = self._det_geotxt_and_meta()
            if geotxt is None:
                logger.warning('_det_geo geotxt is None')
                return None            
            self._geo_ = GeometryAccess()
            self._geo_.load_pars_from_str(geotxt)
        return self._geo_
        

    def _pixel_coord_indexes(self, **kwa):
        """
        """
        logger.debug('AreaDetector._pixel_coord_indexes')
        geo = self._det_geo()
        if geo is None:
            logger.warning('geo is None')
            return None
            
        return geo.get_pixel_coord_indexes(\
            pix_scale_size_um  = kwa.get('pix_scale_size_um',None),\
            xy0_off_pix        = kwa.get('xy0_off_pix',None),\
            do_tilt            = kwa.get('do_tilt',True),\
            cframe             = kwa.get('cframe',0))


    def _pixel_coords(self, **kwa):
        """
        """
        logger.debug('AreaDetector._pixel_coords')
        geo = self._det_geo()
        if geo is None:
            logger.warning('geo is None')
            return None
        #return geo.get_pixel_xy_at_z(self, zplane=None, oname=None, oindex=0, do_tilt=True, cframe=0)
        return geo.get_pixel_coords(\
            do_tilt            = kwa.get('do_tilt',True),\
            cframe             = kwa.get('cframe',0))


    def _cached_pixel_coord_indexes(self, evt, **kwa):
        """
        """
        logger.debug('AreaDetector._cached_pixel_coord_indexes')

        resp = self._pixel_coord_indexes(**kwa)
        if resp is None: return None

        # PRESERVE PIXEL INDEXES FOR USED SEGMENTS ONLY
        segs = self.segments(evt)
        if segs is None: return None
        logger.debug(info_ndarr(segs, 'preserve pixel indices for segments '))

        rows, cols = self._pix_rc_ = [reshape_to_3d(a)[segs,:,:] for a in resp]
        #self._pix_rc_ = [dict_from_arr3d(reshape_to_3d(v)) for v in resp]

        s = 'evaluate_pixel_coord_indexes:'
        for i,a in enumerate(self._pix_rc_): s += info_ndarr(a, '\n  %s '%('rows','cols')[i], last=3)
        logger.info(s)

        mapmode = kwa.get('mapmode',2)
        if mapmode <4:
          self.img_entries, self.dmulti_pix_to_img_idx, self.dmulti_imgidx_numentries=\
            statistics_of_pixel_arrays(rows, cols)

        if mapmode==4:
            rsp = self._pixel_coords(**kwa)
            if rsp is None: return None
            x,y,z = self._pix_xyz_ = [reshape_to_3d(a)[segs,:,:] for a in rsp]
            self._interpol_pars_ = init_interpolation_parameters(rows, cols, x, y)

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


    def calib(self, evt, **kwa) -> Array3d:
        """
        """
        logger.debug('%s.calib(evt) is implemented for generic case of area detector as raw - pedestals' % self.__class__.__name__\
                      +'\n  If needed more, it needs to be re-implemented for this detector type.')
        raw = self.raw(evt)
        if raw is None:
            logger.debug('det.raw.raw(evt) is None')
            return None

        peds = self._pedestals()
        if peds is None:
            logger.debug('det.raw._pedestals() is None - return det.raw.raw(evt)')
            return raw
        return raw - peds


    def image(self, evt, nda=None, **kwa) -> Array2d:
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
        if any(v is None for v in self._pix_rc_):
            self._cached_pixel_coord_indexes(evt, **kwa)
            if any(v is None for v in self._pix_rc_): return None

        vbase     = kwa.get('vbase',0)
        mapmode   = kwa.get('mapmode',2)
        fillholes = kwa.get('fillholes',True)

        if mapmode==0: return self.img_entries

        data = self.calib(evt) if nda is None else nda
        if data is None:
            logger.warning('AreaDetector.image calib returns None')
            return None
            
        #logger.debug(info_ndarr(data, 'data ', last=3))

        rows, cols = self._pix_rc_
        img = img_from_pixel_arrays(rows, cols, weight=data, vbase=vbase) # mapmode==1
        if   mapmode==2: img_multipixel_max(img, data, self.dmulti_pix_to_img_idx)
        elif mapmode==3: img_multipixel_mean(img, data, self.dmulti_pix_to_img_idx, self.dmulti_imgidx_numentries)

        if mapmode<4 and fillholes: fill_holes(img, self.hole_rows, self.hole_cols)

        return img if mapmode<4 else\
               img_interpolated(data, self._interpol_pars_) if mapmode==4 else\
               self.img_entries


    def _shape_as_daq(self):
        peds = self._pedestals()
        if peds is None: 
            logger.warning('In _shape_as_daq pedestals are None, can not define daq data shape')
            return None
        return peds.shape if peds.ndim<4 else peds.shape[-3:]


    def _number_of_segments_total(self):
        shape = self._shape_as_daq()
        return None if shape is None else shape[-3] # (7,n,352,384) - define through calibration constants


    def _mask_default(self, dtype=DTYPE_MASK):
        shape = self._shape_as_daq()
        return None if shape is None else np.ones(shape, dtype=dtype)


    def _mask_calib_or_default(self, dtype=DTYPE_MASK):
        mask = self._mask_calib()
        return self._mask_default(dtype) if mask is None else mask.astype(dtype)


    def _mask_from_status(self, **kwa):
        """
        For simple detectors w/o multi-gain ranges

        Parameters **kwa
        ----------------
        ##mode - int 0/1/2 masks zero/four/eight neighbors around each bad pixel

        Returns
        -------
        mask made of status: np.array, ndim=3, shape: as full detector data
        """
        status = self._status()
        if status is None:
            logger.warning('pixel_status is None')
            return None
        return np.asarray(np.select((status>0,), (0,), default=1), dtype=DTYPE_MASK)


    def _mask_edges(self, **kwa): # -> Array3d:
        mask = self._mask_default(self, dtype=DTYPE_MASK)
        return None if mask is None else\
          mask_edges(mask,\
            edge_rows=kwa.get('edge_rows', 1),\
            edge_cols=kwa.get('edge_cols', 1),\
            dtype=DTYPE_MASK) # kwa.get('dtype', DTYPE_MASK)):


#     def _mask_neighbors(self, **kwa) -> Array3d:
       #mode = kwargs.get('mode', 0)
        #if mode: smask = gu.mask_neighbors(smask, allnbrs=(True if mode>=2 else False))

        #segs = self._segments(evt)
        #if segs is None:
        #    logger.warning('self._segments(evt) is None')
        #    return None
        #return arr3d_from_dict({k:v.raw for k,v in segs.items()})


    def _mask(self, calib=False, status=False, edges=False, neighbors=False, **kwa):
        """Returns per-pixel array with mask values (per-pixel product of all requested masks).
           Parameters
           - calib    : bool - True/False = on/off mask from calib directory.
           - status   : bool - True/False = on/off mask generated from calib pixel_status. 
           - edges    : bool - True/False = on/off mask of edges. 
           - neighbors: bool - True/False = on/off mask of neighbors. 
           - kwa      : dict - additional parameters passed to low level methods (width,...) 
                        for edges: edge_rows=1, edge_cols=1, center_rows=0, center_cols=0, dtype=DTYPE_MASK
                        for status of epix10ka: grinds=(0,1,2,3,4)
           Returns
           - np.array - per-pixel mask values 1/0 for good/bad pixels.
        """
        dtype = kwa.get('dtype', DTYPE_MASK) 
        mask = self._mask_calib_or_default(dtype) if calib else self._mask_default(dtype)
        if status: mask = merge_masks(mask, self._mask_from_status(**kwa)) 
        if edges: mask = merge_masks(mask, self._mask_edges(**kwa))
        #if neighbors: mask = merge_masks(mask, self._mask_neighbors(self, **kwa))
        return mask


    def _mask_comb(self, **kwa):
        mbits=kwa.get('mbits', 1)      
        return self._mask(\
          calib     = mbits & 1,\
          status    = mbits & 2,\
          edges     = mbits & 4,\
          neighbors = mbits & 8,\
          **kwa)

#----

if __name__ == "__main__":
    import sys
    sys.exit('See example in test_%s if available...' % sys.argv[0].split('/')[-1])

#----
