
"""
Class CalibConstants with convenience methods to access info from dict returned from DB
=======================================================================================

Usage::

  from psana.detector.calibconstants import CalibConstants

  o = CalibConstants(calibconst, **kwa)
  v = o.calibconst()
  v = o.cons_and_meta_for_ctype(ctype='pedestals')
  v = o.cached_array(p, ctype='pedestals')
  v = o.pedestals()
  v = o.rms()
  v = o.status(ctype='pixel_status') # for ctype pixel_status or specified
  v = o.status_extra()               # for ctype status_extra
  v = o.mask_calib()
  v = o.common_mode()
  v = o.gain()
  v = o.gain_factor()
  v = o.shape_as_daq()
  v = o.number_of_segments_total()
  v = o.geotxt_and_meta()
  v = o.geotxt_default()
  v = o.geo()
  v = o.pixel_coords(**kwa)
  v = o.pixel_coord_indexes(**kwa)
  v = o.cached_pixel_coord_indexes(segnums, **kwa)
  v = o.image(self, nda, segnums=None, **kwa)
  v = o.pix_rc()
  v = o.pix_xyz()
  v = o.interpol_pars()
  s = o.info_calibconst()

2022-07-07 created by Mikhail Dubrovin
"""
import logging
logger = logging.getLogger(__name__)
import sys
import numpy as np

from psana.detector.NDArrUtils import info_ndarr, divide_protected, reshape_to_3d  # print_ndarr,shape_as_2d, shape_as_3d, reshape_to_2d
from psana.pscalib.geometry.GeometryAccess import GeometryAccess  # img_from_pixel_arrays

from psana.detector.UtilsAreaDetector import dict_from_arr3d, arr3d_from_dict,\
        img_from_pixel_arrays, statistics_of_pixel_arrays, img_multipixel_max, img_multipixel_mean,\
        img_interpolated, init_interpolation_parameters, statistics_of_holes, fill_holes

from psana.detector.UtilsMask import DTYPE_MASK, DTYPE_STATUS

#from psana.detector.Utils import is_none
def is_none(par, msg, logger_method=logger.debug):
    resp = par is None
    if resp: logger_method(msg)
    return resp

def is_dict_like(d):
    import weakref
    return isinstance(d, dict) or isinstance(d, weakref.WeakValueDictionary)

class CalibConstants:
    _registry = {}
    def __new__(cls, calibconst, detname, **kwargs):
        if detname not in cls._registry:
            cls._registry[detname] = super().__new__(cls)
        else:
            cls._registry[detname]._reset(calibconst)
        return cls._registry[detname]

    def __init__(self, calibconst, detname, **kwa):
        """
        Parameters
        ----------
        calibconst: dict - retrieved from DB by method calib_constants_all_types from psana.pscalib.calib.MDBWebUtils.
        calib_constants_all_types is USED BY psana/psexp/ds_base.py TO RETRIEVE ALL CONSTANTS FROM DB

        **kwa: not used
        """
        logger.debug('__init__') #  self.__class__.__name__

        assert is_dict_like(calibconst), 'Input parameter should be dict-like: %s' % str(calibconst)
        self._reset(calibconst)
        self._kwa = kwa
        self._logmet_init = kwa.get('logmet_init', logger.debug)

    def _reset(self, calibconst):
        self._calibconst = calibconst
        self._geo = None
        self._pedestals = None
        self._gain = None # ADU/eV
        self._gain_factor = None # keV/ADU
        self._rms = None
        self._status = None
        self._status_extra = None
        self._common_mode = None
        self._mask_calib = None

        self._pix_rc = None, None
        self._pix_xyz = None, None, None
        self._interpol_pars = None
        self._rc_tot_max = None

    def calibconst(self):
        logger.debug('calibconst')
        cc = self._calibconst
        if is_none(cc, 'self._calibconst is None'): return None
        return cc

    def cons_and_meta_for_ctype(self, ctype='pedestals'):
        logger.debug('cons_and_meta_for_ctype(ctype="%s")'%ctype)
        cc = self.calibconst()
        if cc is None: return None, None
        cons_and_meta = cc.get(ctype, None)
        if is_none(cons_and_meta, 'calibconst["%s"] is None'%ctype, logger_method=logger.debug): return None, None
        return cons_and_meta

    def cached_array(self, p, ctype='pedestals'):
        """Returns cached array of constants for ctype."""
        if p is None: p = self.cons_and_meta_for_ctype(ctype)[0] # 0-data/1-metadata
        return p

    def pedestals(self):   return self.cached_array(self._pedestals, 'pedestals')

    def rms(self):         return self.cached_array(self._rms, 'pixel_rms')

    def common_mode(self): return self.cached_array(self._common_mode, 'common_mode')

    def gain(self):        return self.cached_array(self._gain, 'pixel_gain')

    def mask_calib(self):
        a = self.cached_array(self._mask_calib, 'pixel_mask')
        return a if a is None else a.astype(DTYPE_MASK)

    def status(self, ctype='pixel_status'):
        """for ctype pixel_status"""

        d = {'pixel_status': self._status,
             'status_extra': self._status_extra}
        cach = d.get(ctype, None)
        if cach is None:
            logger.warning('cached array is not reserved for ctype: %s' % ctype\
                           +'  known ctypes: %s' % str(d.keys()))
        a = self.cached_array(cach, ctype)
        return a if a is None else a.astype(DTYPE_STATUS)

    def status_extra(self):
        """for ctype status_extra"""
        a = self.cached_array(self._status_extra, 'status_extra')
        return a if a is None else a.astype(DTYPE_STATUS)

    def gain_factor(self):
        """Evaluates and returns gain factor as 1/gain if gain is available in calib constants else 1."""
        if self._gain_factor is None:
            g = self.gain()
            self._gain_factor = divide_protected(np.ones_like(g), g) if isinstance(g, np.ndarray) else 1
        return self._gain_factor

    def shape_as_daq(self):
        peds = self.pedestals()
        #print(info_ndarr(peds, 'XXX shape_as_daq pedesstals'))
        if is_none(peds, 'shape_as_daq - pedestals is None, can not define daq data shape - returns None'): return None
        return peds.shape if peds.ndim<4 else peds.shape[-3:]

    def number_of_segments_total(self):
        shape = self.shape_as_daq()
        return None if shape is None else\
               1 if len(shape) < 3 else\
               shape[-3] # (7,n,352,384) - define through calibration constants

    def segment_numbers_total(self):
        """Returns total list of segment numbers."""
        nsegs = self.number_of_segments_total()
        segnums = None if is_none(nsegs, 'number_of_segments_total is None') else\
                  list(range(nsegs))
                  #tuple(np.arange(nsegs, dtype=np.uint16))
        logger.debug('segnums: %s' % str(segnums))
        return segnums

    def geotxt_and_meta(self):
        logger.debug('geotxt_and_meta')
        cc = self.calibconst()
        if cc is None: return None
        geotxt_and_meta = cc.get('geometry', None)
        if is_none(geotxt_and_meta, 'calibconst["geometry"] is None', logger_method=logger.debug): return None, None
        return geotxt_and_meta

    def geo(self):
        """Return GeometryAccess() object."""
        if self._geo is None:
            geotxt, meta = self.geotxt_and_meta()
            if geotxt is None:
                if is_none(geotxt, 'geometry constants for this detector are missing in DB > return None'): return None
            self._geo = GeometryAccess()
            self._geo.load_pars_from_str(geotxt)
        return self._geo

    def seg_geo(self):
        """returns SegGeometry object from full GeometryAccess object or None if GeometryAccess is None"""
        logger.debug('seg_geo')

        odet = self._kwa.get('odet', None)
        if odet is not None:
            return odet._seg_geo
        geo = self.geo()
        if is_none(geo, 'in seg_geo() both odet and self.geo() are None'):
            return geo.get_seg_geo().algo

    def pixel_coords(self, **kwa):
        """DEPRECATED - can't load detector-dependent default geometry here...
           returns x, y, z - three np.ndarray
        """
        logger.debug('pixel_coords')
        geo = self.geo()
        if is_none(geo, 'in pixel_coords geo is None'): return None

        #return geo.get_pixel_xy_at_z(self, zplane=None, oname=None, oindex=0, do_tilt=True, cframe=0)
        return geo.get_pixel_coords(\
            do_tilt            = kwa.get('do_tilt',True),\
            cframe             = kwa.get('cframe',0))

    def pixel_coord_indexes(self, **kwa):
        """DEPRECATED - can't load detector-dependent default geometry here...
           returns ix, iy - two np.ndarray
        """
        logger.debug('pixel_coord_indexes')
        geo = self.geo()
        if is_none(geo, 'in pixel_coord_indexes geo is None'): return None

        return geo.get_pixel_coord_indexes(\
            pix_scale_size_um  = kwa.get('pix_scale_size_um',None),\
            xy0_off_pix        = kwa.get('xy0_off_pix',None),\
            do_tilt            = kwa.get('do_tilt',True),\
            cframe             = kwa.get('cframe',0))

    def cached_pixel_coord_indexes(self, segnums=None, **kwa):
        logger.debug('CalibConstants.cached_pixel_coord_indexes')

        resp = self.pixel_coord_indexes(**kwa)
        if resp is None: return None

        logger.debug(info_ndarr(resp, 'detector total rows, cols: '))
        self._rc_tot_max = [np.max(np.ravel(a)) for a in resp]
        logger.debug('_rc_tot_max: %s' % str(self._rc_tot_max))

        # PRESERVE PIXEL INDEXES FOR USED SEGMENTS ONLY
        if segnums is None:
            segnums = self.segment_numbers_total()

        logmet_init = self._logmet_init
        logmet_init(info_ndarr(segnums, 'preserve pixel indices for segments '))

        logmet_init(info_ndarr(resp, 'self.pixel_coord_indexes '))

        rows, cols = self._pix_rc = [reshape_to_3d(a)[segnums,:,:] for a in resp]

        s = 'evaluate_pixel_coord_indexes:'
        for i,a in enumerate(self._pix_rc): s += info_ndarr(a, '\n  %s '%('rows','cols')[i], last=3)
        logmet_init(s)

        mapmode = kwa.get('mapmode',2)
        fillholes = kwa.get('fillholes',True)

        if mapmode <4:
          self.img_entries, self.dmulti_pix_to_img_idx, self.dmulti_imgidx_numentries=\
            statistics_of_pixel_arrays(rows, cols)

        if mapmode==4:
            rsp = self._pixel_coords(**kwa)
            if rsp is None: return None
            x,y,z = self._pix_xyz = [reshape_to_3d(a)[segnums,:,:] for a in rsp]
            self._interpol_pars = init_interpolation_parameters(rows, cols, x, y)

        if mapmode <4 and fillholes:
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

    def image(self, nda, segnums=None, **kwa):
        """
        Create 2-d image.

        Parameters
        ----------
        nda: np.array, ndim=3
            array shaped as daq raw data.

        segnums: list/tuple of segment (uint) indexes

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
        logger.debug('in CalibConstants.image')

        if any(v is None for v in self._pix_rc):
            self.cached_pixel_coord_indexes(segnums, **kwa)
            if any(v is None for v in self._pix_rc): return None

        vbase     = kwa.get('vbase',0)
        mapmode   = kwa.get('mapmode',2)
        fillholes = kwa.get('fillholes',True)

        #logger.debug('in CalibConstants.image segnums', segnums, 'mapmode:', mapmode)

        if mapmode==0: return self.img_entries

        if is_none(nda, 'CalibConstants.image calib returns None', logger_method=logger.warning): return None

        logger.debug(info_ndarr(nda, 'nda ', last=3))
        rows, cols = self._pix_rc
        logger.debug(info_ndarr(rows, 'rows ', last=3))
        logger.debug(info_ndarr(cols, 'cols ', last=3))

        img = img_from_pixel_arrays(rows, cols, weight=nda, vbase=vbase, rc_tot_max=self._rc_tot_max) # mapmode==1

        if   mapmode==2: img_multipixel_max(img, nda, self.dmulti_pix_to_img_idx)
        elif mapmode==3: img_multipixel_mean(img, nda, self.dmulti_pix_to_img_idx, self.dmulti_imgidx_numentries)

        if mapmode<4 and fillholes: fill_holes(img, self.hole_rows, self.hole_cols)

        return img if mapmode<4 else\
               img_interpolated(nda, self._cached_interpol_pars()) if mapmode==4 else\
               self.img_entries

    def pix_rc(self): return self._pix_rc

    def pix_xyz(self): return self._pix_xyz

    def interpol_pars(self): return self._interpol_pars

    def info_calibconst(self):
        """grabs det.raw._calibconst from self and returns info about available constants"""
        cc =self.calibconst()
        keys = cc.keys()
        s = '    det.raw._calibconst.keys(): %s' % (', '.join(keys))
        for k,v in cc.items():
            nda, meta = v
            if k == 'geometry': continue
            s += info_ndarr(nda, '\n    %s from exp:%s run:%04d' % (k.ljust(12), meta['experiment'], meta['run']), last=5)

        geo = cc.get('geometry', None)
        s += '\n    geometry IS MISSING' if geo is None else\
             '\n    geometry from exp:%s run:%04d\n%s'%\
                    (meta['experiment'], meta['run'], str(geo)[:500])
        return s

# EOF
