
"""
:py:class:`Mask` - alternative access to AreaDetector det.raw._mask_* methods
=============================================================================

Usage::

    from psana2.detector.mask import Mask

    o = Mask(det, status=True, neighbors=False, edges=False, center=False, calib=False, umask=None, dtype=DTYPE_MASK, **kwa)
    # caches requested mask in det._mask_

    mask = o.mask_comb(status=True, neighbors=False, edges=False, center=False, calib=False, umask=None, dtype=DTYPE_MASK, **kwa)
    # returns combined mask w/o caching

    # All other methods are shotcuts to det._mask* methods w/o "_"

    mask = o.set_mask(**kwa) # forces update
    mask = o.mask(**kwa)
    mask = o.mask_default()
    mask = o.mask_calib_or_default()
    mask = o.mask_from_status(status_bits=0xffff, dtype=DTYPE_MASK, **kwa)
    mask = o.mask_edges(width=0, edge_rows=1, edge_cols=1, dtype=DTYPE_MASK, **kwa)
    mask = o.mask_center(wcenter=0, center_rows=1, center_cols=1, dtype=DTYPE_MASK, **kwa)
    mask = o.mask_neighbors(mask, rad=9, ptrn='r')

2022-03-29 created by Mikhail Dubrovin
"""

import psana2.detector.UtilsMask as um
DTYPE_MASK, DTYPE_STATUS = um.DTYPE_MASK, um.DTYPE_STATUS


class Mask:
    def __init__(self, det, **kwa):
        """sets det._mask_, see parameters description in areadetector.py AreaDetector.mask_comb"""
        self.det = det
        self.det_raw = det.raw
        self.set_mask(**kwa)


    def set_mask(self, **kwa):
        """sets det._mask_, see parameters description in areadetector.py AreaDetector.mask_comb"""
        kwa['force_update'] = True
        _ = self.det_raw._mask(**kwa)


    def mask(self, **kwa):
        return self.det_raw._mask(**kwa)


    def mask_comb(self, **kwa):
        """shortcut to AreaDetector.mask_comb"""
        kwa['force_update'] = True
        return self.det_raw._mask_comb(**kwa)


    def mask_default(self, dtype=DTYPE_MASK):
        return self.det_raw._mask_default(dtype)


    def mask_calib_or_default(self, dtype=DTYPE_MASK):
        return self.det_raw._mask_calib_or_default(dtype)


    def mask_from_status(self, status_bits=0xffff, dtype=DTYPE_MASK, **kwa):
        kwa['status_bits'] = status_bits
        kwa['dtype'] = dtype
        return self.det_raw._mask_from_status(**kwa)


    def mask_edges(self, width=0, edge_rows=1, edge_cols=1, dtype=DTYPE_MASK, **kwa):
        return self.det_raw._mask_edges(width, edge_rows, edge_cols, dtype, **kwa)


    def mask_center(self, wcenter=0, center_rows=1, center_cols=1, dtype=DTYPE_MASK, **kwa):
        return self.det_raw._mask_center(wcenter, center_rows, center_cols, dtype, **kwa)


    def mask_neighbors(self, mask, rad=9, ptrn='r'):
        return self.det_raw._mask_neighbors(mask, rad, ptrn)

# EOF
