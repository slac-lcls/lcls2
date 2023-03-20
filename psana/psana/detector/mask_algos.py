
"""
Class MaskAlgos - methods to generae masks from calib constants and algorithms
==============================================================================

Usage::

  from psana.detector.mask_algos import MaskAlgos

  o = MaskAlgos(calibconst, **kwa)
  v = o.cco  # CalibConstants object


2022-07-08 created by Mikhail Dubrovin
"""

import logging
logger = logging.getLogger(__name__)

import numpy as np
from psana.detector.calibconstants import CalibConstants
import psana.detector.UtilsMask as um
DTYPE_MASK, DTYPE_STATUS = um.DTYPE_MASK, um.DTYPE_STATUS

from psana.detector.Utils import is_none
from psana.detector.NDArrUtils import info_ndarr


class MaskAlgos:

    def __init__(self, calibconst, **kwa):
        """calibconst: dict from DB
        """
        logger.debug('__init__')
        self.cco = CalibConstants(calibconst, **kwa)
        self._mask = None


    def mask_default(self, dtype=DTYPE_MASK):
        shape = self.cco.shape_as_daq()
        return None if shape is None else np.ones(shape, dtype=dtype)


    def mask_calib_or_default(self, dtype=DTYPE_MASK):
        mask = self.cco.mask_calib()
        return self.mask_default(dtype) if mask is None else mask.astype(dtype)


    def gain_range_indexes(self, grinds, smask):
        """returns list of gain range indexes, e.g. (0,1,2,3,4) to merge mask_from"""
        if grinds is not None: return grinds
        if smask.ndim<4: return None
        ngranges = smask.shape[0]
        if ngranges>5: ngranges=5 # protection for epix10ka which have 7 gain ranges, but two of them - evaluated
        return tuple(range(ngranges))


    def mask_from_status(self, status_bits=(1<<64)-1, stextra_bits=(1<<64)-1, gain_range_inds=None, dtype=DTYPE_MASK, **kwa):
        """
        Mask made of pixel_status calibration constants.

        Parameters **kwa
        ----------------

        - status_bits  : uint - bitword for calib-type pixel_status codes to mask
        - stextra_bits : uint - bitword for calib-type sataus_extra codes to mask
        - dtype : numpy.dtype - mask np.array dtype

        Returns
        -------
        np.array - mask made of pixel_status calibration constants, shapeed as full detector data
        """
        smask = None
        for ctype, sbits in
            (('pixel_status', status_bits),
             ('status_extra', stextra_bits),
            )
            #status = self.cco.status()
            status, meta = self.cco.cons_and_meta_for_ctype(ctype=ctype)
            status = status.astype(DTYPE_STATUS)

            logger.debug(info_ndarr(status, ctype))
            if is_none(status, 'array for ctype: %s is None' % ctype): continue # return None

            _smask = um.status_as_mask(status, status_bits=sbits, dtype=DTYPE_MASK, **kwa)
            # smask shape can be e.g.: (!!!7, 4, 352, 384)
            smask = _smask if smask is None else
                    um.merge_masks(smask, _smask, dtype=dtype)

        if is_none(smask, 'status_as_mask is None'): return None

        grinds = self.gain_range_indexes(gain_range_inds, smask)
        logger.debug('in MaskAlgos.mask_from_status'\
                + '\n   grinds: %s' % str(grinds)\
                + '\n   status_bits: %s' % hex(status_bits)\
                + '\n   stextra_bits: %s' % hex(stextra_bits)\
                + '\n   dtype: %s' % str(dtype)\
                + '\n   **kwa: %s' % str(kwa)\
                + '\n   the last in loop meta: %s' % str(meta))

        mask = smask if grinds is None else\
               um.merge_mask_for_grinds(smask, gain_range_inds=grinds, dtype=dtype, **kwa)

        logger.debug(info_ndarr(smask, 'in MaskAlgos.mask_from_status smask:'))
        logger.debug(info_ndarr(mask,  'in MaskAlgos.mask_from_status  mask:'))
        return mask


    def mask_neighbors(self, mask, rad=9, ptrn='r'):
        """Returns 2-d or n-d mask array with increased by radial paramerer rad region around all 0-pixels in the input mask.

           Parameter

           - mask : np.array - input mask of good/bad (1/0) pixels
           - rad  : int - radisl parameter for masking region aroung each 0-pixel in mask
           - ptrn : char - pattern of the masked region, ptrn='r'-rhombus, 'c'-circle, othervise square [-rad,+rad] in rows and columns.

           Returns

           - np.array - mask with masked neighbors, shape = mask.shape
        """
        return um.mask_neighbors(mask, rad, ptrn)


    def mask_edges(self, width=0, edge_rows=1, edge_cols=1, dtype=DTYPE_MASK, **kwa):
        return um.mask_edges(self.mask_default(),\
            width=width, edge_rows=edge_rows, edge_cols=edge_cols, dtype=dtype, **kwa)


    def mask_center(self, wcenter=0, center_rows=1, center_cols=1, dtype=DTYPE_MASK, **kwa):
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
        logger.debug('epix_base.mask_edges')
        seg_geo = self.cco.seg_geo()

        mask1 = seg_geo.pixel_mask_array(width=0, edge_rows=0, edge_cols=0,\
                  center_rows=center_rows, center_cols=center_cols, dtype=dtype, **kwa)
        nsegs = self.cco.number_of_segments_total()
        if is_none(nsegs, '_number_of_segments_total is None'): return None
        return np.stack([mask1 for i in range(nsegs)])


    def mask_comb(self, status=True, neighbors=False, edges=False, center=False, calib=False, umask=None, dtype=DTYPE_MASK, **kwa):
        """Returns combined mask controlled by the keyword arguments.
           Parameters
           ----------
           - status   : bool : True  - mask from pixel_status constants,
                                       kwa: status_bits=0xffff - status bits for calib-type pixel_status to use in mask.
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
           - umask  : np.array: None - apply user's defined mask from input parameters (shaped as data)

           Returns
           -------
           np.array: dtype=np.uint8, shape as det.raw - mask array of 1 or 0 or None if all switches are False.
        """

        logger.debug('MaskAlgos.mask_comb ---- mask evolution')

        mask = None
        if status:
            status_bits  = kwa.get('status_bits', (1<<64)-1)
            stextra_bits = kwa.get('stextra_bits', (1<<64)-1)
            gain_range_inds = kwa.get('gain_range_inds', None) # (0,1,2,3,4) works for epix10ka
            mask = self.mask_from_status(status_bits=status_bits, stextra_bits=stextra_bits, gain_range_inds=gain_range_inds, dtype=dtype)
            logger.debug(info_ndarr(mask, 'in mask_comb after mask_from_status'))

#        if unbond and (self.is_cspad2x2() or self.is_cspad()):
#            mask_unbond = self.mask_geo(par, width=0, wcenter=0, mbits=4) # mbits=4 - unbonded pixels for cspad2x1 segments
#            mask = mask_unbond if mask is None else um.merge_masks(mask, mask_unbond)

        if neighbors and mask is not None:
            rad  = kwa.get('rad', 5)
            ptrn = kwa.get('ptrn', 'r')
            mask = um.mask_neighbors(mask, rad=rad, ptrn=ptrn)
            logger.debug(info_ndarr(mask, 'in mask_comb after mask_neighbors:'))

        if edges:
            width = kwa.get('width', 0)
            erows = kwa.get('edge_rows', 1)
            ecols = kwa.get('edge_cols', 1)
            mask_edges = self.mask_edges(width=width, edge_rows=erows, edge_cols=ecols, dtype=dtype) # masks each segment edges only
            mask = mask_edges if mask is None else um.merge_masks(mask, mask_edges, dtype=dtype)
            logger.debug(info_ndarr(mask, 'in mask_comb after mask_edges:'))

        if center:
            wcent = kwa.get('wcenter', 0)
            crows = kwa.get('center_rows', 1)
            ccols = kwa.get('center_cols', 1)
            mask_center = self.mask_center(wcenter=wcent, center_rows=crows, center_cols=ccols, dtype=dtype)
            mask = mask_center if mask is None else um.merge_masks(mask, mask_center, dtype=dtype)
            logger.debug(info_ndarr(mask, 'in mask_comb after mask_center:'))

        if calib:
            mask_calib = self.mask_calib_or_default(dtype=dtype)
            mask = mask_calib if mask is None else um.merge_masks(mask, mask_calib, dtype=dtype)
            logger.debug(info_ndarr(mask, 'in mask_comb after mask_calib:'))

        if umask is not None:
            mask = umask if mask is None else um.merge_masks(mask, umask, dtype=dtype)

        logger.debug(info_ndarr(mask, 'in mask_comb at exit:'))

        return mask


    def mask(self, status=True, neighbors=False, edges=False, center=False, calib=False, umask=None, force_update=False, dtype=DTYPE_MASK, **kwa):
        """returns cached mask_comb.
        """
        if self._mask is None or force_update:
           self._mask = self.mask_comb(status=status, neighbors=neighbors, edges=edges, center=center, calib=calib, umask=umask, dtype=dtype, **kwa)
        return self._mask


if __name__ == "__main__":
    import sys
    sys.exit('See example in test_%s if available...' % sys.argv[0].split('/')[-1])

# EOF
