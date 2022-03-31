"""
Data access methods common for all AREA DETECTORS
=================================================

Usage::

  from psana.detector.areadetector import AreaDetector

  o = AreaDetector(*args, **kwa) # inherits from DetectorImpl(*args, **kwa)

  a = o.raw(evt)
  a = o._segment_numbers(evt)
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
  a = o._det_geotxt_default()
  a = o._det_geo()
  a = o._pixel_coord_indexes(pix_scale_size_um=None, xy0_off_pix=None, do_tilt=True, cframe=0, **kwa)
  a = o._pixel_coords(do_tilt=True, cframe=0, **kwa)
  a = o._cached_pixel_coord_indexes(evt, **kwa) # **kwa - the same as above

  a = o._shape_as_daq()
  a = o._number_of_segments_total()

  a = o._mask_default(dtype=DTYPE_MASK)
  a = o._mask_calib()
  a = o._mask_calib_or_default(dtype=DTYPE_MASK)
  a = o._mask_from_status(status_bits=0xffff, dtype=DTYPE_MASK, **kwa) # gain_range_inds=(0,1,2,3,4) - gain ranges to merge for apropriate detectors
  a = o._mask_neighbors(mask, rad=9, ptrn='r')
  a = o._mask_edges(width=0, edge_rows=1, edge_cols=1, dtype=DTYPE_MASK, **kwa)
  a = o._mask_center(wcenter=0, center_rows=1, center_cols=1, dtype=DTYPE_MASK, **kwa)
  a = o._mask_comb(**kwa) # the same as _mask but w/o caching
  a = o._mask(status=True, status_bits=0xffff, gain_range_inds=(0,1,2,3,4),\
              neighbors=False, rad=3, ptrn='r',\
              edges=True, width=0, edge_rows=10, edge_cols=5,\
              center=True, wcenter=0, center_rows=5, center_cols=3,\
              calib=False,\
              umask=None,\
              force_update=False)

  a = o.calib(evt, cmpars=(7,2,100,10), *kwargs)
  a = o.calib(evt, **kwa)
  a = o.image(self, evt, nda=None, **kwa)

2020-11-06 created by Mikhail Dubrovin
"""

from psana.detector.detector_impl import DetectorImpl

import logging
logger = logging.getLogger(__name__)

import numpy as np

from psana.pscalib.geometry.SegGeometryStore import sgs
from psana.pscalib.geometry.GeometryAccess import GeometryAccess #, img_from_pixel_arrays
from psana.detector.NDArrUtils import info_ndarr, reshape_to_3d # print_ndarr,shape_as_2d, shape_as_3d, reshape_to_2d
from psana.detector.UtilsAreaDetector import dict_from_arr3d, arr3d_from_dict,\
        img_from_pixel_arrays, statistics_of_pixel_arrays, img_multipixel_max, img_multipixel_mean,\
        img_interpolated, init_interpolation_parameters, statistics_of_holes, fill_holes
import psana.detector.UtilsMask as um
import psana.detector.Utils as ut

DTYPE_MASK, DTYPE_STATUS = um.DTYPE_MASK, um.DTYPE_STATUS

from amitypes import Array2d, Array3d


def is_none(par, msg):
    resp = par is None
    if resp: logger.debug(msg)
    return resp


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
        self._gain_ = None # ADU/eV
        self._gain_factor_ = None # keV/ADU
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
        if evt is None: return None
        segs = self._segments(evt)
        if is_none(segs, 'self._segments(evt) is None'): return None
        return arr3d_from_dict({k:v.raw for k,v in segs.items()})


    def _segment_numbers(self,evt):
        """ Returns dense 1-d numpy array of segment indexes.
        from dict self._segments(evt)
        """
        segs = self._segments(evt)
        if is_none(segs, 'self._segments(evt) is None'): return None
        return np.array(sorted(segs.keys()), dtype=np.uint16)


    def _det_calibconst(self):
        logger.debug('AreaDetector._det_calibconst')
        cc = self._calibconst
        if is_none(cc, 'self._calibconst is None'): return None
        return cc


    def _calibcons_and_meta_for_ctype(self, ctype='pedestals'):
        logger.debug('AreaDetector._calibcons_and_meta_for_ctype(ctype="%s")'%ctype)
        cc = self._det_calibconst()
        if cc is None: return None
        cons_and_meta = cc.get(ctype, None)
        if is_none(cons_and_meta, 'calibconst["%s"] is None'%ctype): return None, None
        return cons_and_meta


    def _cached_array(self, p, ctype='pedestals'):
        """cached array
        """
        if p is None: p = self._calibcons_and_meta_for_ctype(ctype)[0] # 0-data/1-metadata
        return p


    def _pedestals(self): return self._cached_array(self._pedestals_, 'pedestals')
    def _rms(self):       return self._cached_array(self._rms_, 'pixel_rms')
    def _status(self):    return self._cached_array(self._status_, 'pixel_status')
    def _mask_calib(self):return self._cached_array(self._mask_calib_, 'pixel_mask')
    def _common_mode(self):return self._cached_array(self._common_mode_, 'common_mode')
    def _gain(self)       :return self._cached_array(self._gain_, 'pixel_gain')


    def _gain_factor(self):
        """Evaluate and return gain factor as 1/gain if gain is available in calib constants else 1."""
        if self._gain_factor_ is None:
            g = self._gain()
            self._gain_factor_ = divide_protected(np.ones_like(g), g) if isinstance(g, np.ndarray) else 1
        return self._gain_factor_


    def _det_geotxt_and_meta(self):
        logger.debug('AreaDetector._det_geotxt_and_meta')
        cc = self._det_calibconst()
        if cc is None: return None
        geotxt_and_meta = cc.get('geometry', None)
        if is_none(geotxt_and_meta, 'calibconst[geometry] is None'): return None, None
        return geotxt_and_meta


    def _det_geotxt_default(self):
        logger.debug('_det_geotxt_default should be re-implemented in specific detector subclass, othervise returns None')
        return None


    def _det_geo(self):
        """
        """
        if self._geo_ is None:
            geotxt, meta = self._det_geotxt_and_meta()
            if geotxt is None:
                geotxt = self._det_geotxt_default()
                if is_none(geotxt, '_det_geo geotxt is None'): return None
            self._geo_ = GeometryAccess()
            self._geo_.load_pars_from_str(geotxt)
        return self._geo_


    def _pixel_coord_indexes(self, **kwa):
        """
        """
        logger.debug('AreaDetector._pixel_coord_indexes')
        geo = self._det_geo()
        if is_none(geo, 'geo is None'): return None

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
        if is_none(geo, 'geo is None'): return None

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
        segs = self._segment_numbers(evt)
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
        if is_none(raw, 'det.raw.raw(evt) is None'): return None

        peds = self._pedestals()
        if is_none(peds, 'det.raw._pedestals() is None - return det.raw.raw(evt)'): return raw

        arr = raw - peds
        gfac = self._gain_factor()

        return arr*gfac if gfac != 1 else arr


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

        if is_none(data, 'AreaDetector.image calib returns None'): return None

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
        if is_none(peds, 'In _shape_as_daq pedestals is None, can not define daq data shape - returns None'): return None
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


    def _mask_from_status(self, status_bits=0xffff, dtype=DTYPE_MASK, **kwa):
        """
        Mask made of pixel_status calibration constants.

        Parameters **kwa
        ----------------

        - status_bits : uint - bitword for mask status codes.
        - dtype : numpy.dtype - mask np.array dtype

        Returns
        -------
        np.array - mask made of pixel_status calibration constants, shapeed as full detector data
        """
        status = self._status()
        if is_none(status, 'pixel_status is None'): return None
        return um.status_as_mask(status, status_bits=status_bits, dtype=DTYPE_MASK, **kwa)


    def _mask_neighbors(self, mask, rad=9, ptrn='r'):
        """Returns 2-d or n-d mask array with increased by radial paramerer rad region around all 0-pixels in the input mask.

           Parameter

           - mask : np.array - input mask of good/bad (1/0) pixels
           - rad  : int - radisl parameter for masking region aroung each 0-pixel in mask
           - ptrn : char - pattern of the masked region, ptrn='r'-rhombus, 'c'-circle, othervise square [-rad,+rad] in rows and columns.

           Returns

           - np.array - mask with masked neighbors, shape = mask.shape
        """
        return um.mask_neighbors(mask, rad, ptrn)


    def _mask_edges(self, width=0, edge_rows=1, edge_cols=1, dtype=DTYPE_MASK, **kwa):
        return um.mask_edges(self._mask_default(),\
            width=width, edge_rows=edge_rows, edge_cols=edge_cols, dtype=dtype, **kwa)


    def _mask_center(self, wcenter=0, center_rows=1, center_cols=1, dtype=DTYPE_MASK, **kwa):
        """
        Parameters
        ----------
        center_rows: int number of edge rows to mask for all ASICs
        center_cols: int number of edge columns to mask for all ASICs
        dtype: numpy dtype of the output array
        **kwa: is not used
        Returns
        -------
        mask: np.ndarray, ndim=3, shaped as full detector data, mask of the panel and asic edges
        """
        logger.debug('epix_base._mask_edges')
        mask1 = self._seg_geo.pixel_mask_array(width=0, edge_rows=0, edge_cols=0,\
                  center_rows=center_rows, center_cols=center_cols, dtype=dtype, **kwa)
        nsegs = self._number_of_segments_total()
        if is_none(nsegs, '_number_of_segments_total is None'): return None
        return np.stack([mask1 for i in range(nsegs)])


    def _mask_comb(self, status=True, neighbors=False, edges=False, center=False, calib=False, umask=None, dtype=DTYPE_MASK, **kwa):
        """Returns combined mask controlled by the keyword arguments.
           Parameters
           ----------
           - status   : bool : True  - mask from pixel_status constants,
                                       kwa: status_bits=0xffff - status bits to use in mask.
                                       Status bits show why pixel is considered as bad.
                                       Content of the bitword depends on detector and code version.
                                       It is wise to exclude pixels with any bad status by setting status_bits=0xffff.
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
           - umask  : np.array: None - apply user's defined mask from input parameters (shaped as data)

           Returns
           -------
           np.array: dtype=np.uint8, shape as det.raw - mask array of 1 or 0 or None if all switches are False.
        """

        mask = None
        if status:
            status_bits = kwa.get('status_bits', 0xffff)
            gain_range_inds  = kwa.get('gain_range_inds', (0,1,2,3,4)) # works for epix10ka
            mask = self._mask_from_status(status_bits=0xffff, gain_range_inds=gain_range_inds, dtype=dtype)

#        if unbond and (self.is_cspad2x2() or self.is_cspad()):
#            mask_unbond = self.mask_geo(par, width=0, wcenter=0, mbits=4) # mbits=4 - unbonded pixels for cspad2x1 segments
#            mask = mask_unbond if mask is None else um.merge_masks(mask, mask_unbond)

        if neighbors and mask is not None:
            rad  = kwa.get('rad', 5)
            ptrn = kwa.get('ptrn', 'r')
            mask = um.mask_neighbors(mask, rad=rad, ptrn=ptrn)

        if edges:
            width = kwa.get('width', 0)
            erows = kwa.get('edge_rows', 1)
            ecols = kwa.get('edge_cols', 1)
            mask_edges = self._mask_edges(width=width, edge_rows=erows, edge_cols=ecols, dtype=dtype) # masks each segment edges only
            mask = mask_edges if mask is None else um.merge_masks(mask, mask_edges, dtype=dtype)

        if center:
            wcent = kwa.get('wcenter', 0)
            crows = kwa.get('center_rows', 1)
            ccols = kwa.get('center_cols', 1)
            mask_center = self._mask_center(wcenter=wcent, center_rows=crows, center_cols=ccols, dtype=dtype)
            mask = mask_center if mask is None else um.merge_masks(mask, mask_center, dtype=dtype)

        if calib:
            mask_calib = self._mask_calib_or_default(dtype=dtype)
            mask = mask_calib if mask is None else um.merge_masks(mask, mask_calib, dtype=dtype)

        if umask is not None:
            mask = umask if mask is None else um.merge_masks(mask, umask, dtype=dtype)

        return mask


    def _mask(self, status=True, neighbors=False, edges=False, center=False, calib=False, umask=None, force_update=False, dtype=DTYPE_MASK, **kwa):
        """returns cached mask.
        """
        if self._mask_ is None or force_update:
           self._mask_ = self._mask_comb(status=status, neighbors=neighbors, edges=edges, center=center, calib=calib, umask=umask, dtype=dtype, **kwa)
        return self._mask_


if __name__ == "__main__":
    import sys
    sys.exit('See example in test_%s if available...' % sys.argv[0].split('/')[-1])

# EOF
