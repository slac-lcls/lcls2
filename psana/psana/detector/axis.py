import numpy as np
import logging
logger = logging.getLogger(__name__)
from psana.detector.detector_impl import DetectorImpl
from amitypes import Array2d
import psana.detector.areadetector as ad
AreaDetectorRaw, sgs, sys, is_none, is_true = ad.AreaDetectorRaw, ad.sgs, ad.sys, ad.ut.is_none, ad.ut.is_true


class axis_raw_1_0_0(AreaDetectorRaw):
    def __init__(self, *args, **kwargs): # **kwargs intercepted by AreaDetectorRaw
        super().__init__(*args, **kwargs)
        self._seg_geo =  None
        self._geo = None

    def _init_geometry(self, shape, pix_size_rcsd_um=(100,100,100,400)):
        """delayed geometry initialization when raw.shape is available in det.raw.raw"""
        logger.info('_init_geometry for raw.shape %s' % str(shape))
        #self._seg_geo = sgs.Create(segname=f'MTRX:{shape[0]}:{shape[1]}:{pixel_size_microns}:{pixel_size_microns}', detector=self)
        self._seg_geo = sgs.Create(segname='MTRXANY:V1', detector=self)
        self._seg_geo.init_matrix_parameters(shape=shape, pix_size_rcsd_um=pix_size_rcsd_um)
        self._path_geo_default = 'pscalib/geometry/data/geometry-def-axis.data' # None

    def raw(self, evt) -> Array2d:
        segs = self._segments(evt)
        if segs is None: return None
        return segs[0].value

    def calib(self, evt, nda=None, **kwa) -> Array2d:
        raw = self.raw(evt) if nda is None else nda
        if is_none(raw, 'raw is None', logger.debug):
            return None
        #peds = self._calibconst['pedestals'][0]
        peds = self._pedestals() # - n-d pedestals
        if is_none(peds, 'no axis pedestals', logger.warning):
            return raw
        if is_true(peds.shape != raw.shape,\
                   f'incorrect axis pedestal shape: {peds.shape}, raw data shape: {raw.shape}',\
                   logger_method=logger.warning):
            return raw
        arr = raw-peds
        mask = self._mask() # **kwa
        return arr if is_none(mask, 'det.raw._mask() is None - return raw - peds', logger.info) else\
               arr*mask

    def image(self, evt, nda=None, **kwa) -> Array2d:
        """ **kwa keys: use_calib_as_image, pix_size_rcsd_um"""
        r = self.raw(evt)
        if r is None: return None
        use_calib = kwa.get('use_calib_as_image', True)
        if (not use_calib) and self._seg_geo is None and r is not None:
            self._init_geometry(r.shape, pix_size_rcsd_um=kwa.get('pix_size_rcsd_um', (10,10,10,400)))
        return self.calib(evt, nda=nda) if use_calib else\
               AreaDetectorRaw.image(self, evt, nda=nda, **kwa)

#EOF
