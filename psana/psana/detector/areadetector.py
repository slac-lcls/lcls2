"""
Data access methods common for all AREA DETECTORS
=================================================

Usage::

  from psana.detector.areadetector import AreaDetector, AreaDetectorRaw

  o = AreaDetector(*args, **kwa)    # inherits from DetectorImpl(*args, **kwa)

  a = o._segment_numbers  # alias of o._sorted_segment_inds, where inds stands for indices
  a = o._det_calibconst()

  a = o._pedestals(all_segs=False)
  a = o._gain(all_segs=False)
  a = o._rms(all_segs=False)
  a = o._status(all_segs=False)
  a = o._mask_calib(all_segs=False)
  a = o._common_mode()
  a = o._det_geotxt_and_meta()
  a = o._det_geotxt_default()
  a = o._det_geo()
  a = o._pixel_coord_indexes(pix_scale_size_um=None, xy0_off_pix=None, do_tilt=True, cframe=0, all_segs=False, **kwa)
  a = o._pixel_coords(do_tilt=True, cframe=0, all_segs=False, **kwa)

  a = o._shape_as_daq()
  a = o._number_of_segments_total()

  a = o._mask_default(dtype=DTYPE_MASK, all_segs=False)
  a = o._mask_calib()
  a = o._mask_calib_or_default(dtype=DTYPE_MASK)
  a = o._mask_from_status(status_bits=(1<<64)-1,\
                          stextra_bits=(1<<64)-1,\
                          dtype=DTYPE_MASK, **kwa) # gain_range_inds=(0,1,2,3,4)
  a = o._mask_neighbors(mask, rad=9, ptrn='r')
  a = o._mask_edges(width=0, edge_rows=1, edge_cols=1, dtype=DTYPE_MASK, **kwa)
  a = o._mask_center(wcenter=0, center_rows=1, center_cols=1, dtype=DTYPE_MASK, **kwa)
  a = o._mask_comb(**kwa) # the same as _mask but w/o caching
  a = o._mask(status=True, status_bits=0xffff, stextra_bits=(1<<64)-1, gain_range_inds=(0,1,2,3,4),\
              neighbors=False, rad=3, ptrn='r',\
              edges=True, width=0, edge_rows=10, edge_cols=5,\
              center=True, wcenter=0, center_rows=5, center_cols=3,\
              calib=False,\
              umask=None,\
              force_update=False,\
              all_segs=False)
  a = o.image(self, evt, nda=None, value_for_missing_segments=None, **kwa)


  o = AreaDetectorRaw(*args, **kwa) # inherits from AreaDetector(*args, **kwa), adds raw and calib methods
  a = o.raw(evt)
  a = o.calib(evt, cmpars=(7,2,100,10), **kwa)
  a = o.calib(evt, **kwa)

2020-11-06 created by Mikhail Dubrovin
"""

from psana.detector.detector_impl import DetectorImpl

import logging
logger = logging.getLogger(__name__)

import os
import sys
import time
import numpy as np
from psana.detector.calibconstants import CalibConstants
from psana.pscalib.geometry.SegGeometryStore import sgs  # used in epix_base.py and derived
from psana.pscalib.geometry.GeometryAccess import GeometryAccess
import psana.detector.NDArrUtils as au # reshape_to_3d, shape_as_3d, shape_as_3d, reshape_to_2d
info_ndarr, reshape_to_3d =  au.info_ndarr, au.reshape_to_3d
from psana.detector.mask_algos import MaskAlgos, DTYPE_MASK, DTYPE_STATUS
from amitypes import Array2d, Array3d
import psana.detector.Utils as ut

is_none, is_true = ut.is_none, ut.is_true

class AreaDetector(DetectorImpl):
    """Collection of methods common for self = det.raw, det.fex, etc."""

    def __init__(self, *args, **kwargs):
        logger.debug('AreaDetector.__init__')
        DetectorImpl.__init__(self, *args, **kwargs)
        self._args = args
        self._kwargs = kwargs
        self._calibc_ = None
        self._geo = None
        self._seg_geo = None
        self._path_geo_default = None
        self._segment_numbers = self._sorted_segment_inds  # [0, 1, 2,... 17, 18, 19]
        self._maskalgos_ = None
        self._store_ = None  # detector dependent storage of cached parameters for method calib
        self._logmet_init = kwargs.get('logmet_init', logger.debug)


    def _calibconstants(self, **kwa):
        """Returns cached object of CalibConstants derived from DetectorImpl._calibconst - dict from DB."""
        if self._calibc_ is None:
            cc = {} if self._calibconst is None else self._calibconst # defined in DetectorImpl # dict  of {ctype:(data, metadata)}
            #logger.debug('AreaDetector._calibconst.keys() / ctypes:', self._calibconst.keys())
            kwa.setdefault('logmet_init', self._logmet_init)
            self._calibc_ = CalibConstants(cc, self._det_name, **kwa)
            shared_cache = getattr(self, "_shared_calibc_cache", None)
            if shared_cache is not None:
                self._calibc_._shared_calibc_cache = shared_cache
                self._calibc_._drp_class_name = getattr(self, "_drp_class_name", "raw")
            self._apply_calibc_preload_cache()
            self._logmet_init('AreaDetector._calibconstants - makes CalibConstants\n%s'%\
                              self._calibc_.info_calibconst())
        return self._calibc_


    def _info_calibconst(self):
        return 'det.raw._info_calibconst for %s\n' % self._det_name\
             + self._calibconstants().info_calibconst()


    def _arr_for_daq_segments(self, arr, **kwa):
        """Returns unchanged arr if all_segs=True, othervise returns sub-array for presented in daq segments."""
        all_segs = kwa.get('all_segs', False)
        logger.debug('AreaDetector._arr_for_segments - all_segs: %s\n    %s' % (str(all_segs), info_ndarr(arr, 'arr:')))
        return arr if (all_segs or not isinstance(arr, np.ndarray) or arr.ndim==2) else\
               np.take(arr, self._segment_numbers, axis=-3)


    def _det_calibconst(self, metname, **kwa):
        """Returns constants of ctype metname for daq segments of for entire detector is all_segs=True."""
        logger.debug('AreaDetector._det_calibconst')
        o = self._calibconstants(**kwa)
        if is_none(o, 'self._calibconstants is None', logger_method=logger.debug):
            return None
        cc_for_ctype = getattr(o, metname)(**kwa)
        return cc_for_ctype if not isinstance(cc_for_ctype, np.ndarray) else\
               self._arr_for_daq_segments(cc_for_ctype, **kwa)


    def _pedestals(self, **kwa):    return self._det_calibconst('pedestals', **kwa)
    def _rms(self, **kwa):          return self._det_calibconst('rms', **kwa)
    def _status(self, **kwa):       return self._det_calibconst('status', **kwa)
    def _mask_calib(self, **kwa):   return self._det_calibconst('mask_calib', **kwa)
    def _common_mode(self, **kwa):  return self._det_calibconst('common_mode', **kwa)
    def _gain(self, **kwa):         return self._det_calibconst('gain', **kwa)
    def _gain_factor(self, **kwa):  return self._det_calibconst('gain_factor', **kwa)
    def _det_geotxt_and_meta(self): return self._det_calibconst('geotxt_and_meta')


    def _fname_geotxt_default(self):
        """returns (str) file name for default geometry constants lcls2/psana/psana/pscalib/geometry/data/geometry-def-*.data"""
        dir_psana = os.path.abspath(os.path.dirname(__file__)).rstrip('detector')
        path = os.path.join(dir_psana, self._path_geo_default)
        #print('default geometry:', path)
        return path # os.path.join(dir_psana, self._path_geo_default)


    def _det_geotxt_default(self):
        """returns (str) default geometry constants from lcls2/psana/psana/pscalib/geometry/data/geometry-def-*.data"""
        fname = self._fname_geotxt_default()
        logger.debug('_det_geotxt_default - load default geometry from file: %s' % fname)
        return ut.load_textfile(fname)


    def _det_geo(self):
        """Returns cached object self._geo of GeometryAccess() from CalibConstants,
           loads it from default file if missing in CalibConstants.
        """
        if self._path_geo_default is None: return None
        self._geo = self._det_calibconst('geo')
        if self._geo is None:
            geotxt = self._det_geotxt_default()
            self._geo = GeometryAccess(detector=self)
            self._geo.load_pars_from_str(geotxt)
            logger.debug('AreaDetector._det_geo DEFAULT GEOMETRY IS LOADED from file %s' % self._fname_geotxt_default())
        return self._geo


    def _pixel_coord_indexes(self, **kwa):
        geo = self._det_geo()
        if geo is None: return None
        cache = getattr(self, "_shared_geo_cache", None)
        if cache is not None and getattr(cache, "enabled", False):
            geotxt = None
            calibconst = getattr(self, "_calibconst", None)
            if isinstance(calibconst, dict):
                geotxt_entry = calibconst.get("geometry")
                if geotxt_entry:
                    geotxt = geotxt_entry[0]
            if geotxt is None and self._path_geo_default is not None:
                try:
                    geotxt = self._det_geotxt_default()
                except Exception:
                    geotxt = None
            segnums = None if kwa.get("all_segs", False) else self._segment_numbers
            geom_id = cache.build_geom_id(
                geotxt,
                segnums,
                {
                    "pix_scale_size_um": kwa.get("pix_scale_size_um", None),
                    "xy0_off_pix": kwa.get("xy0_off_pix", None),
                    "do_tilt": kwa.get("do_tilt", True),
                    "cframe": kwa.get("cframe", 0),
                    "all_segs": kwa.get("all_segs", False),
                },
            )
            det_name = getattr(self, "_det_name", "unknown")
            drp_class = getattr(self, "_drp_class_name", "raw")
            key = cache.make_key(det_name, drp_class, geom_id)
            cached_ix = cache.get_if_present(key, "pix_rows")
            cached_iy = cache.get_if_present(key, "pix_cols")
            if cached_ix is not None and cached_iy is not None:
                return cached_ix, cached_iy

            shm_comm = getattr(cache.shared_mem, "shm_comm", None)
            is_leader = getattr(cache.shared_mem, "is_leader", False)
            local_ix = local_iy = None
            shape_dtype = None
            if is_leader:
                local_ix, local_iy = geo.get_pixel_coord_indexes(\
                    pix_scale_size_um = kwa.get('pix_scale_size_um',None),\
                    xy0_off_pix       = kwa.get('xy0_off_pix',None),\
                    do_tilt           = kwa.get('do_tilt',True),\
                    cframe            = kwa.get('cframe',0))
                local_ix = self._arr_for_daq_segments(local_ix, **kwa)
                local_iy = self._arr_for_daq_segments(local_iy, **kwa)
                shape_dtype = (
                    local_ix.shape,
                    str(local_ix.dtype),
                    local_iy.shape,
                    str(local_iy.dtype),
                )
            if shm_comm is not None:
                shape_dtype = shm_comm.bcast(shape_dtype, root=0)
            if shape_dtype is None:
                ix,iy = geo.get_pixel_coord_indexes(\
                    pix_scale_size_um = kwa.get('pix_scale_size_um',None),\
                    xy0_off_pix       = kwa.get('xy0_off_pix',None),\
                    do_tilt           = kwa.get('do_tilt',True),\
                    cframe            = kwa.get('cframe',0))
                return self._arr_for_daq_segments(ix, **kwa),\
                       self._arr_for_daq_segments(iy, **kwa)

            (sx, dx, sy, dy) = shape_dtype
            arr_ix, _ = cache.get_or_allocate(key, "pix_rows", sx, np.dtype(dx), zero_init=False)
            arr_iy, _ = cache.get_or_allocate(key, "pix_cols", sy, np.dtype(dy), zero_init=False)
            if is_leader and local_ix is not None:
                np.copyto(arr_ix, local_ix)
                np.copyto(arr_iy, local_iy)
            if shm_comm is not None:
                shm_comm.Barrier()
            return arr_ix, arr_iy

        ix,iy = geo.get_pixel_coord_indexes(\
            pix_scale_size_um = kwa.get('pix_scale_size_um',None),\
            xy0_off_pix       = kwa.get('xy0_off_pix',None),\
            do_tilt           = kwa.get('do_tilt',True),\
            cframe            = kwa.get('cframe',0))
        return self._arr_for_daq_segments(ix, **kwa),\
               self._arr_for_daq_segments(iy, **kwa)


    def _pixel_coords(self, **kwa):
        geo = self._det_geo()
        if geo is None: return None
        cache = getattr(self, "_shared_geo_cache", None)
        if cache is not None and getattr(cache, "enabled", False):
            geotxt = None
            calibconst = getattr(self, "_calibconst", None)
            if isinstance(calibconst, dict):
                geotxt_entry = calibconst.get("geometry")
                if geotxt_entry:
                    geotxt = geotxt_entry[0]
            if geotxt is None and self._path_geo_default is not None:
                try:
                    geotxt = self._det_geotxt_default()
                except Exception:
                    geotxt = None
            segnums = None if kwa.get("all_segs", False) else self._segment_numbers
            geom_id = cache.build_geom_id(
                geotxt,
                segnums,
                {
                    "do_tilt": kwa.get("do_tilt", True),
                    "cframe": kwa.get("cframe", 0),
                    "all_segs": kwa.get("all_segs", False),
                },
            )
            det_name = getattr(self, "_det_name", "unknown")
            drp_class = getattr(self, "_drp_class_name", "raw")
            key = cache.make_key(det_name, drp_class, geom_id)
            cached_x = cache.get_if_present(key, "pix_x")
            cached_y = cache.get_if_present(key, "pix_y")
            cached_z = cache.get_if_present(key, "pix_z")
            if cached_x is not None and cached_y is not None and cached_z is not None:
                return cached_x, cached_y, cached_z

            shm_comm = getattr(cache.shared_mem, "shm_comm", None)
            is_leader = getattr(cache.shared_mem, "is_leader", False)
            local_x = local_y = local_z = None
            shape_dtype = None
            if is_leader:
                local_x, local_y, local_z = geo.get_pixel_coords(\
                    do_tilt = kwa.get('do_tilt',True),\
                    cframe = kwa.get('cframe',0))
                local_x = self._arr_for_daq_segments(local_x, **kwa)
                local_y = self._arr_for_daq_segments(local_y, **kwa)
                local_z = self._arr_for_daq_segments(local_z, **kwa)
                shape_dtype = (
                    local_x.shape,
                    str(local_x.dtype),
                    local_y.shape,
                    str(local_y.dtype),
                    local_z.shape,
                    str(local_z.dtype),
                )
            if shm_comm is not None:
                shape_dtype = shm_comm.bcast(shape_dtype, root=0)
            if shape_dtype is None:
                x,y,z = geo.get_pixel_coords(\
                    do_tilt = kwa.get('do_tilt',True),\
                    cframe = kwa.get('cframe',0))
                return self._arr_for_daq_segments(x, **kwa),\
                       self._arr_for_daq_segments(y, **kwa),\
                       self._arr_for_daq_segments(z, **kwa)

            (sx, dx, sy, dy, sz, dz) = shape_dtype
            arr_x, _ = cache.get_or_allocate(key, "pix_x", sx, np.dtype(dx), zero_init=False)
            arr_y, _ = cache.get_or_allocate(key, "pix_y", sy, np.dtype(dy), zero_init=False)
            arr_z, _ = cache.get_or_allocate(key, "pix_z", sz, np.dtype(dz), zero_init=False)
            if is_leader and local_x is not None:
                np.copyto(arr_x, local_x)
                np.copyto(arr_y, local_y)
                np.copyto(arr_z, local_z)
            if shm_comm is not None:
                shm_comm.Barrier()
            return arr_x, arr_y, arr_z

        x,y,z = geo.get_pixel_coords(\
            do_tilt = kwa.get('do_tilt',True),\
            cframe = kwa.get('cframe',0))
        return self._arr_for_daq_segments(x, **kwa),\
               self._arr_for_daq_segments(y, **kwa),\
               self._arr_for_daq_segments(z, **kwa)


    def _pixel_xy_at_z(self, **kwa):
        geo = self._det_geo()
        if geo is None: return None
        x,y = geo.get_pixel_xy_at_z(\
             zplane = kwa.get('zplane', None),\
             oname  = kwa.get('oname', None),
             oindex = kwa.get('oindex', 0),\
             do_tilt= kwa.get('do_tilt', True),\
             cframe = kwa.get('cframe', 0))
        return self._arr_for_daq_segments(x, **kwa),\
               self._arr_for_daq_segments(y, **kwa)


    def _number_of_segments_daq(self): return len(self._segment_numbers)


    def _number_of_segments_total(self): return self._det_calibconst('number_of_segments_total')


    def _shape_as_daq(self):
        return (self._number_of_segments_daq(),) + tuple(self._seg_geo.shape())


    def _shape_total(self):
        return (self._number_of_segments_total(),) + tuple(self._seg_geo.shape())
        #return self._det_calibconst('shape_as_daq')


#    def _segment_ids(self):
#        """Returns list of detector segment ids"""
#        return self._uniqueid.split('_')[1:]


    def _substitute_value_for_missing_segments(self, nda_daq, value) -> Array3d:
        nsegs_tot = self._number_of_segments_total()
        nsegs_daq = self._number_of_segments_daq()
        if nsegs_daq == nsegs_tot:
            return nda_daq
        arr = value * np.ones(self._shape_total(), dtype=nda_daq.dtype)
        for itot,idaq in zip(self._segment_numbers, tuple(range(nsegs_daq))):
            arr[itot,:] = nda_daq[idaq,:]
        return arr


    def image(self, evt, nda=None, value_for_missing_segments=None, **kwa) -> Array2d:
        """Returns 2-d image.
           If value_for_missing_segments is specified,
             all missing segments will be substituted with this value and shape = shape for all segments,
             otherwice image for available self._segment_numbers will be generated.
        """
        value = value_for_missing_segments

        _nda = self.calib(evt) if nda is None else nda

        segnums = self._segment_numbers

        if value is not None:
            _nda = self._substitute_value_for_missing_segments(_nda, value)
            segnums = None

        o = self._calibconstants(**kwa)
        #if is_none(o, 'in AreaDetector det.raw._calibconstants(...) is None', logger_method=logger.info): return None
        if o.geo() is None: o._geo = self._det_geo()
        return o.image(_nda, segnums=segnums, **kwa)


    def _maskalgos(self, logger_method=logger.debug):
        """ returns cached (in self._maskalgos_) MaskAlgos.
            Uses self._kwargs from AreaDetector to initialize MaskAlgos
        """
        if is_none(self._maskalgos_, 'AreaDetector._maskalgos - make MaskAlgos, **AreaDetector._kwargs: %s' % str(self._kwargs),\
                   logger_method=logger_method):
            cc = self._calibconst   # defined in DetectorImpl from detector_impl.py
            if is_none(cc, 'self._calibconst is None', logger_method=logger_method): return None
            mkwa = self._kwargs
            mkwa.setdefault('logmet_init', self._logmet_init)
            mkwa.setdefault('odet', self) # need it to get self._seg_geo
            logger_method('in AreaDetector creating MaskAlgos for _det_name:%s with **mkwa:%s'%(self._det_name, str(mkwa)))
            self._maskalgos_ = MaskAlgos(cc, self._det_name, **mkwa)
        return self._maskalgos_


    def _mask_method_wrapper(self, metname, **kwa):
        """wrapper for all _mask_* methods to utilise them from class MaskAlgos.
           Use it in stead of similar decorator
        """
        o = self._maskalgos()
        logger.debug('in _mask_method_wrapper(%s, **kwa) **kwa: %s' % (metname, str(kwa)))
        if is_none(o, 'self._maskalgos is None', logger_method=logger.debug): return None
        met = getattr(o, metname, None)
        if is_none(met, 'metname %s IS NOT FOUND IN MaskAlgos', logger_method=logger.warning): return None
        mask = met(**kwa)
        return self._arr_for_daq_segments(mask, **kwa)


    def _mask_default(self, dtype=DTYPE_MASK, **kwa):
        return self._mask_method_wrapper('mask_default', dtype=DTYPE_MASK, **kwa)

    def _mask_calib_or_default(self, dtype=DTYPE_MASK, **kwa):
        return self._mask_method_wrapper('mask_calib_or_default', dtype=DTYPE_MASK, **kwa)

    def _mask_from_status(self, status_bits=0xffff, stextra_bits=(1<<64)-1, stci_bits=(1<<64)-1,\
                          gain_range_inds=None, dtype=DTYPE_MASK, **kwa):
        return self._mask_method_wrapper('mask_from_status', status_bits=status_bits, stextra_bits=stextra_bits,\
                                         stci_bits=stci_bits, gain_range_inds=gain_range_inds, dtype=dtype, **kwa)

    def _mask_neighbors(self, mask, rad=9, ptrn='r', **kwa):
        return self._mask_method_wrapper('mask_neighbors', mask=mask, rad=rad, ptrn=ptrn, **kwa)

    def _mask_edges(self, width=0, edge_rows=1, edge_cols=1, dtype=DTYPE_MASK, **kwa):
        return self._mask_method_wrapper('mask_edges', width=width, edge_rows=edge_rows, edge_cols=edge_cols, dtype=dtype, **kwa)

    def _mask_center(self, wcenter=0, center_rows=1, center_cols=1, dtype=DTYPE_MASK, **kwa):
        return self._mask_method_wrapper('mask_center', wcenter=wcenter, center_rows=center_rows, center_cols=center_cols,\
                                         dtype=dtype, **kwa)

    def _mask_comb(self, status=True, neighbors=False, edges=False, center=False, calib=False, umask=None, dtype=DTYPE_MASK, **kwa):
        """Returns combined mask controlled by the keyword arguments.
           Parameters
           ----------
           - status   : bool : True  - mask from pixel_status constants,
                                       kwa: status_bits=0xffff - status bits for calib-type pixel_status to use in mask,
                                       kwa: stextra_bits=(1<<64)-1 - status bits for calib-type status_extra to use in mask.
                                       Status bits show why pixel is considered as bad.
                                       Content of the bitword depends on detector and code version.
                                       It is wise to exclude pixels with any bad status by setting status_bits=(1<<64)-1.
                                       kwa: gain_range_inds=(0,1,2,3,4) - list of gain range indexes to merge for epix10ka or jungfrau
           - neighbor : bool : False - mask of neighbors of all bad pixels,
                                       kwa: rad=5 - radial parameter of masked region
                                       kwa: ptrn='r'-rhombus, 'c'-circle, othervise square region around each bad pixel
           - edges    : bool : False - mask edge rows and columns of each panel,
                                       kwa: width=0 or edge_rows=1, edge_cols=1 - number of masked edge rows, columns
           - center   : bool : False - mask center rows and columns of each panel consisting of ASICS (cspad, epix, jungfrau),
                                       kwa: wcenter=0 or center_rows=1, center_cols=1 -
                                       number of masked center rows and columns in the segment,
                                       works for cspad2x1, epix100, epix10ka, jungfrau panels
           - calib    : bool : False - apply user's defined mask from pixel_mask constants
           - umask    : np.array: None - apply user's defined mask from input parameters (shaped as data)
           - dtype    : np.dtype: np.uint8 - mask array data type

           Returns
           -------
           np.array: dtype=np.uint8, shape as det.raw - mask array of 1 or 0 or None if all switches are False.
        """
        return self._mask_method_wrapper('mask_comb', status=status, neighbors=neighbors, edges=edges,\
                                         center=center, calib=calib, umask=umask, dtype=dtype, **kwa)

    def _mask(self, **kwa):
        """Returns cached mask. **kwargs passed from Detector(..., **kwargs)"""
        logger.debug('in AreaDetector._mask(**kwa - not used, set them in Detector(..., **kwa))')
        return self._mask_method_wrapper('mask')

#    def _mask(self, status=True, neighbors=False, edges=False, center=False,\
#              calib=False, umask=None, force_update=False, dtype=DTYPE_MASK, **kwa):
#        """Returns cached mask. Dict of kwargs is the same as in _mask_comb."""
#        return self._mask_method_wrapper('mask', status=status, neighbors=neighbors, edges=edges, center=center,\
#                                         calib=calib, umask=umask, force_update=force_update, dtype=dtype, **kwa)


class AreaDetectorRaw(AreaDetector):
    """Collection of methods for self = det.raw, e.g. det.raw.raw(...)/calib/image etc."""

    def __init__(self, *args, **kwargs):
        logger.debug('AreaDetectorRaw.__init__') #  self.__class__.__name__
        AreaDetector.__init__(self, *args, **kwargs)
        self._raw_buf = None
        self._raw_shape = None
        self._raw_dtype = None


    def raw(self, evt) -> Array3d:
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

        #print('XXX AreaDetectorRaw.raw')
        if evt is None: return None
        stack_timing = getattr(evt, '_det_raw_timing', None)
        t0 = time.perf_counter() if stack_timing is not None else None
        segs = self._segments(evt)    # dict = {seg_index: seg_obj}
        if stack_timing is not None and t0 is not None:
            stack_timing['segments'] += time.perf_counter() - t0
        if is_none(segs, 'self._segments(evt) is None'): return None
        if len(segs) == 1:
            ind = self._segment_numbers[0]
            return segs[ind].raw

        first_seg = segs[self._segment_numbers[0]].raw
        dtype = first_seg.dtype
        shape = (len(self._segment_numbers),) + first_seg.shape
        if self._raw_buf is None or self._raw_shape != shape or self._raw_dtype != dtype:
            self._raw_buf = np.empty(shape, dtype=dtype)
            self._raw_shape = shape
            self._raw_dtype = dtype

        t1 = time.perf_counter() if stack_timing is not None else None
        for idx, seg_id in enumerate(self._segment_numbers):
            np.copyto(self._raw_buf[idx], segs[seg_id].raw, casting='no')
        if stack_timing is not None and t1 is not None:
            stack_timing['stack'] += time.perf_counter() - t1
        return reshape_to_3d(self._raw_buf)


    def calib(self, evt, **kwa) -> Array3d:
        """Returns calibrated array of data shaped as daq: calib = (raw - peds) * gfac * mask.
           Should be overridden for more complicated cases.
        """
        logger_method = kwa.get('logger_method', logger.debug)
        #logger_method('AreaDetectorRaw.calib')
        logger_method('%s.calib(evt) is implemented for generic case of area detector as (raw - peds) * gfac * mask' % self.__class__.__name__\
                     +'\n  If needed more, it should to be re-implemented for specific detector type.')
        raw = self.raw(evt)
        if is_none(raw, 'det.raw.raw(evt) is None', logger_method): return None

        peds = self._pedestals()
        if is_none(peds, 'det.raw._pedestals() is None, return raw', logger_method): return raw
        arr = raw - peds

        gfac = self._gain_factor()
        if is_none(gfac, 'det.raw._gain_factor() is None, return raw - peds', logger_method): return arr
        if gfac != 1: arr *= gfac

        #logger_method('XXX call det._mask(**self.kwa) from AreaDetectorRaw.calib')
        mask = self._mask(**kwa)
        if is_none(mask, 'det.raw._mask() is None - return (raw - peds)*gfac', logger_method): return arr
        return arr*mask


if __name__ == "__main__":
    import sys
    sys.exit('See example in test_%s if available...' % sys.argv[0].split('/')[-1])

# EOF
