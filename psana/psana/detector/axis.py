import numpy as np
import logging
logger = logging.getLogger(__name__)
from psana.detector.detector_impl import DetectorImpl
from amitypes import Array2d
from psana.detector.areadetector import AreaDetectorRaw, sgs

def is_true(cond, msg, method=logger.debug):
    if cond: method(msg)
    return cond

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
        #self._geo = self._det_geo() # None

    def raw(self, evt) -> Array2d:
        segs = self._segments(evt)
        if segs is None: return None
        return segs[0].value

    def calib(self, evt, nda=None) -> Array2d:
        raw = self.raw(evt) if nda is None else nda
        if raw is None: return None
        peds = self._calibconst['pedestals'][0]
        if is_true(peds is None, 'no axis pedestals', method=logger.warning):
            return raw
        if is_true(peds.shape != raw.shape,\
                   f'incorrect axis pedestal shape: {peds.shape}, raw data shape: {raw.shape}',\
                   method=logger.warning):
            return raw
        cal = raw-peds
        return cal

    def image(self, evt, nda=None, **kwa) -> Array2d:
        """ **kwa keys: use_calib_as_image, pix_size_rcsd_um"""
        r = self.raw(evt)
        if r is None: return None
        use_calib = kwa.get('use_calib_as_image', True)
        if (not use_calib) and self._seg_geo is None and r is not None:
            self._init_geometry(r.shape, pix_size_rcsd_um=kwa.get('pix_size_rcsd_um', (10,10,10,400)))
        return self.calib(evt, nda=nda) if use_calib else\
               AreaDetectorRaw.image(self, evt, nda=nda, **kwa)
# EOF
