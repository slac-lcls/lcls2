import numpy as np
import logging
logger = logging.getLogger(__name__)
from psana.detector.detector_impl import DetectorImpl
from amitypes import Array2d
from psana.detector.areadetector import AreaDetectorRaw, sgs

class axis_raw_1_0_0(AreaDetectorRaw):
    def __init__(self, *args, **kwargs): # **kwargs intercepted by AreaDetectorRaw
        super().__init__(*args, **kwargs)
        self._seg_geo =  None
        self._geo = None

    def _init_geometry(self, shape):
        """delayed geometry initialization when raw.shape is available in det.raw.raw"""
        logger.info('_init_geometry for raw.shape %s' % str(shape))
        pixel_size_microns = 1
        self._seg_geo = sgs.Create(segname=f'MTRX:{shape[0]}:{shape[1]}:{pixel_size_microns}:{pixel_size_microns}', detector=self)
        self._path_geo_default = None # 'pscalib/geometry/data/geometry-def-axis.data'
        self._geo = self._det_geo() # None

    def raw(self, evt) -> Array2d:
        segs = self._segments(evt)
        if segs is None: return None
        return segs[0].value

    def calib(self, evt) -> Array2d:
        raw = self.raw(evt)
        if raw is None: return None
        peds = self._calibconst['pedestals'][0]
        if peds is None:
            logging.warning('no axis pedestals')
            return raw
        if peds.shape != raw.shape:
            logging.warning(f'incorrect axis pedestal shape: {peds.shape}, raw data shape: {raw.shape}')
            return raw
        cal = raw-peds
        return cal

    def image(self, evt, nda=None, **kwa) -> Array2d:
        r = self.raw(evt)
        if r is None: return None
        if self._geo is None and r is not None:
            self._init_geometry(r.shape)
        return self.calib(evt)

# EOF
