"""
Utilities for mask
==================

Usage::
  from psana.detector.UtilsMask import *

  statmgd = merge_status(status, grinds=(0,1,2,3,4), dtype=DTYPE_STATUS)
         # merges status.shape=(7, 16, 352, 384) to statmgd.shape=(16, 352, 384) of dtype
         # grinds list stands for gain ranges for 'FH','FM','FL','AHL-H','AML-M'
  m = mask_neighbors(mask, allnbrs=True, dtype=DTYPE_MASK)
  m = mask_edges(mask, edge_rows=1, edge_cols=1, dtype=DTYPE_MASK)
  mask_merged = merge_masks(mask1=None, mask2=None, dtype=DTYPE_MASK)

2021-01-25 created by Mikhail Dubrovin
"""
import numpy as np
import psana.pscalib.calib.CalibConstants as CC
DTYPE_MASK   = CC.dic_calib_type_to_dtype[CC.PIXEL_MASK]   # np.uint8
DTYPE_STATUS = CC.dic_calib_type_to_dtype[CC.PIXEL_STATUS] # np.uint64

from psana.detector.NDArrUtils import shape_nda_as_3d # shape_as_3d# info_ndarr, shape_as_3d


def merge_masks(mask1=None, mask2=None, dtype=DTYPE_MASK):
    """Merging masks using np.logical_and rule: (0,1,0,1)^(0,0,1,1) = (0,0,0,1)
    """
    assert mask1.size == mask2.size, 'Mask sizes should be equal'

    if mask1 is None: return mask2
    if mask2 is None: return mask1

    if mask1.shape != mask2.shape:
        if mask1.ndim > mask2.ndim: mask2.shape = mask1.shape
        else                      : mask1.shape = mask2.shape

    cond = np.logical_and(mask1, mask2)
    return np.asarray(np.select((cond,), (1,), default=0), dtype=dtype)
    #return mask if dtype==np.bool else np.asarray(mask, dtype)


def merge_mask_for_grinds(mask, grinds=(0,1,2,3,4), dtype=DTYPE_MASK):
    """Merges mask bits over gain range index.
       grinds list(uint) - list of gain range inices in array mask[i,:]
       grinds=(0,1,2,3,4) for epix10ka/quad/2m mask array mask.shape=(7, <num-segments>, 352, 384) merging to (<num-segments>, 352, 384)
       grinds=(0,1,2) for Jungfrau mask array mask.shape=(3, <num-segments>, 512, 512) merging to (<num-segments>, 512, 512)
    """
    if mask.ndim < 4: return mask # ignore 3-d arrays
    _mask = mask.astype(dtype)
    mask1 = np.copy(_mask[grinds[0],:])
    for i in grinds[1:]:
        if i<mask.shape[0]:
            cond = np.logical_and(mask1, _mask[i,:]) #, out=mask1)
            mask1 = np.asarray(np.select((cond,), (1,), default=0), dtype=dtype)
    return mask1


def merge_status_for_grinds(status, grinds=(0,1,2,3,4), dtype=DTYPE_STATUS):
    """Merges status bits over gain range index.
       Originaly intended for epix10ka(quad/2m) status array status.shape=(7, 16, 352, 384) merging to (16, 352, 384)
       Also can be used with Jungfrau status array status.shape=(7, 8, 512, 512) merging to (8, 512, 512)
       option "indexes" contains a list of status[i,:] indexes to combine status
    """
    if status.ndim < 2: return status # ignore 1-d arrays
    _status = status.astype(dtype)
    st1 = np.copy(_status[grinds[0],:])
    for i in grinds[1:]: # range(1,status.shape[0]):
        if i<status.shape[0]: # boundary check for index
            np.bitwise_or(st1, _status[i,:], out=st1)
    return st1


def mask_neighbors(mask, rad=5, ptrn='r'):
    """In mask array increase region of masked pixels around bad by radial paramerer rad.
       Parameters:
       -----------
       - mask (np.ndarray) - input mask array ndim >=2
       - rad (int) - radial parameter of masked region
       - ptrn (char) - pattern of the masked region, for now ptrn='r'-rhombus, ptrn='c'-circle,
                       othervise square [-rad,+rad] in rows and columns.

       Time on psanagpu109 for img shape:(2203, 2299)
       rad=4: 0.5s
       rad=9: 2.5s
    """
    #t0_sec = time()
    assert isinstance(mask, np.ndarray)
    assert mask.ndim>1
    mmask = np.array(mask)
    rows, cols = mask.shape[-2],mask.shape[-1]
    for dr in range(-rad, rad+1):
      r1b, r1e = max(dr, 0), min(rows, rows+dr)
      r2b, r2e = max(-dr, 0), min(rows, rows-dr)
      for dc in range(-rad, rad+1):
        if ptrn=='r' and (abs(dr)+abs(dc) > rad): continue
        elif ptrn=='c' and (dr*dr + dc*dc > rad*rad ): continue
        c1b, c1e = max(dc, 0), min(cols, cols+dc)
        c2b, c2e = max(-dc, 0), min(cols, cols-dc)
        if mask.ndim==2:
          mmask[r1b:r1e,c1b:c1e] = merge_masks(mmask[r1b:r1e,c1b:c1e], mask[r2b:r2e,c2b:c2e])
        else:
          mmask[:,r1b:r1e,c1b:c1e] = merge_masks(mmask[:,r1b:r1e,c1b:c1e], mask[:,r2b:r2e,c2b:c2e])
    #logger.info('mask_neighbors(rad=%d) time = %.3f sec' % (rad, time()-t0_sec))
    return mmask


def mask_edges(mask, width=0, edge_rows=1, edge_cols=1, dtype=DTYPE_MASK):
    """Returns mask with a requested number of row and column pixels masked - set to 0.

       Parameters

       - mask : np.ndarray, dtype=uint - n-dimensional (n>1) array with input mask
       - width: int - width of edge for rows and colsumns to mask, if width>0 it overrides edge_rows and edge_cols
       - edge_rows: int - number of edge rows to mask
       - edge_cols: int - number of edge columns to mask

       Returns

       - np.ndarray - mask array with masked edges of all panels.
    """

    #assert isinstance(mask, np.ndarray), 'input mask should be numpy array'
    if not isinstance(mask, np.ndarray):
            logger.debug('input mask is not np.ndarray - return None')
            return None

    erows = width if width>0 else edge_rows
    ecols = width if width>0 else edge_cols

    sh = mask.shape
    assert mask.ndim>1

    mask_out = np.asarray(mask, dtype)

    if mask.ndim == 2:
        rows, cols = sh

        if erows > rows:
          raise ValueError('Requested number of edge rows=%d to mask exceeds 2-d, shape=%s' % (erows, str(sh)))

        if ecols > cols:
          raise ValueError('Requested number of edge columns=%d to mask exceeds 2-d, shape=%s' % (ecols, str(sh)))

        if erows>0:
          # mask edge rows
          mask_rows = np.zeros((erows,cols), dtype=mask.dtype)
          mask_out[:erows ,:] = mask_rows
          mask_out[-erows:,:] = mask_rows

        if ecols>0:
          # mask edge colss
          mask_cols = np.zeros((rows,ecols), dtype=mask.dtype)
          mask_out[:,:ecols ] = mask_cols
          mask_out[:,-ecols:] = mask_cols

    else: # shape > 2
        mask_out.shape = shape_nda_as_3d(mask)

        segs, rows, cols = mask_out.shape

        if erows > rows:
          raise ValueError('Requested number of edge rows=%d to mask exceeds 2-d, shape=%s' % (erows, str(sh)))

        if ecols > cols:
          raise ValueError('Requested number of edge columns=%d to mask exceeds 2-d, shape=%s' % (ecols, str(sh)))

        if erows>0:
          # mask edge rows
          mask_rows = np.zeros((segs,erows,cols), dtype=mask.dtype)
          mask_out[:, :erows ,:] = mask_rows
          mask_out[:, -erows:,:] = mask_rows

        if ecols>0:
          # mask edge colss
          mask_cols = np.zeros((segs,rows,ecols), dtype=mask.dtype)
          mask_out[:, :,:ecols ] = mask_cols
          mask_out[:, :,-ecols:] = mask_cols

        mask_out.shape = sh

    return mask_out


def status_as_mask(status, mstcode=0xffff, dtype=DTYPE_MASK, **kwa):
    """Returns per-pixel array of mask generated from pixel_status.

       Parameters

       - status  : np.array - pixel_status calibration constants
       - mstcode : bitword for mask status codes
       - dtype : mask np.array dtype

       Returns

       - np.array - mask generated from calibration type pixel_status (1/0 for status 0/>0, respectively).
    """
    if not isinstance(status, np.ndarray):
            logger.debug('status is not np.ndarray - return None')
            return None

    from psana.detector.NDArrUtils import info_ndarr
    print(info_ndarr(status, 'status'))
    cond = (status & mstcode)>0
    return np.asarray(np.select((cond,), (0,), default=1), dtype=dtype)

# EOF

