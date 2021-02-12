"""
Utilities for mask
==================

Usage::
  from psana.detector.UtilsMask import *

  statmgd = merge_status(status, grinds=(0,1,2,3,4), dtype=DTYPE_STATUS)
         # merges status.shape=(7, 16, 352, 384) to statmgd.shape=(16, 352, 384) of dtype
         # grinds list stands for gain ranges for 'FH','FM','FL','AHL-H','AML-M'
  m = mask_neighbors(mask, allnbrs=True, dtype=DTYPE_MASK)
  m = mask_edges(mask, mrows=1, mcols=1, dtype=DTYPE_MASK)
  mask_merged = merge_masks(mask1=None, mask2=None, dtype=DTYPE_MASK)

2021-01-25 created by Mikhail Dubrovin
"""
import numpy as np
import psana.pscalib.calib.CalibConstants as CC
DTYPE_MASK   = CC.dic_calib_type_to_dtype[CC.PIXEL_MASK]   # np.uint8
DTYPE_STATUS = CC.dic_calib_type_to_dtype[CC.PIXEL_STATUS] # np.uint64

from psana.pyalgos.generic.NDArrUtils import shape_nda_as_3d # shape_as_3d# info_ndarr, shape_as_3d

def merge_status(stnda, grinds=(0,1,2,3,4), dtype=DTYPE_STATUS): # indexes stand gain ranges for 'FH','FM','FL','AHL-H','AML-M'
    """Merges status bits over gain range index.
       Originaly intended for epix10ka(quad/2m) status array stnda.shape=(7, 16, 352, 384) merging to (16, 352, 384)
       Also can be used with Jungfrau status array stnda.shape=(7, 8, 512, 512) merging to (8, 512, 512)
       option "indexes" contains a list of stnda[i,:] indexes to combine status
    """
    if stnda.ndim < 2: return stnda # ignore 1-d arrays
    _stnda = stnda.astype(dtype)
    st1 = np.copy(_stnda[grinds[0],:])
    for i in grinds[1:]: # range(1,stnda.shape[0]):
        if i<stnda.shape[0]: # boundary check for index
            np.bitwise_or(st1, _stnda[i,:], out=st1)
    return st1
    #print(info_ndarr(st1,    'XXX st1   '))
    #print(info_ndarr(_stnda, 'XXX stnda '))


def mask_neighbors(mask, allnbrs=True, dtype=DTYPE_MASK):
    """Return mask with masked eight neighbor pixels around each 0-bad pixel in input mask.
       mask   : int - n-dimensional (n>1) array with input mask
       allnbrs: bool - False/True - masks 4/8 neighbor pixels.
    """
    shape_in = mask.shape
    if mask.ndim < 2:
        raise ValueError('Input mask has less then 2-d, shape = %s' % str(shape_in))

    mask_out = np.copy(mask, dtype) # np.asarray(mask, dtype)

    if mask.ndim == 2:
        # mask nearest neighbors
        mask_out[0:-1,:] = np.logical_and(mask_out[0:-1,:], mask[1:,  :])
        mask_out[1:,  :] = np.logical_and(mask_out[1:,  :], mask[0:-1,:])
        mask_out[:,0:-1] = np.logical_and(mask_out[:,0:-1], mask[:,1:  ])
        mask_out[:,1:  ] = np.logical_and(mask_out[:,1:  ], mask[:,0:-1])
        if allnbrs:
          # mask diagonal neighbors
          mask_out[0:-1,0:-1] = np.logical_and(mask_out[0:-1,0:-1], mask[1:  ,1:  ])
          mask_out[1:  ,0:-1] = np.logical_and(mask_out[1:  ,0:-1], mask[0:-1,1:  ])
          mask_out[0:-1,1:  ] = np.logical_and(mask_out[0:-1,1:  ], mask[1:  ,0:-1])
          mask_out[1:  ,1:  ] = np.logical_and(mask_out[1:  ,1:  ], mask[0:-1,0:-1])

    else: # mask.ndim > 2

        mask_out.shape = mask.shape = shape_nda_as_3d(mask)

        # mask nearest neighbors
        mask_out[:, 0:-1,:] = np.logical_and(mask_out[:, 0:-1,:], mask[:, 1:,  :])
        mask_out[:, 1:,  :] = np.logical_and(mask_out[:, 1:,  :], mask[:, 0:-1,:])
        mask_out[:, :,0:-1] = np.logical_and(mask_out[:, :,0:-1], mask[:, :,1:  ])
        mask_out[:, :,1:  ] = np.logical_and(mask_out[:, :,1:  ], mask[:, :,0:-1])
        if allnbrs:
          # mask diagonal neighbors
          mask_out[:, 0:-1,0:-1] = np.logical_and(mask_out[:, 0:-1,0:-1], mask[:, 1:  ,1:  ])
          mask_out[:, 1:  ,0:-1] = np.logical_and(mask_out[:, 1:  ,0:-1], mask[:, 0:-1,1:  ])
          mask_out[:, 0:-1,1:  ] = np.logical_and(mask_out[:, 0:-1,1:  ], mask[:, 1:  ,0:-1])
          mask_out[:, 1:  ,1:  ] = np.logical_and(mask_out[:, 1:  ,1:  ], mask[:, 0:-1,0:-1])

        mask_out.shape = mask.shape = shape_in

    return mask_out


def mask_edges(mask, edge_rows=1, edge_cols=1, dtype=DTYPE_MASK):
    """Return mask with a requested number of row and column pixels masked - set to 0.
       mask : int - n-dimensional (n>1) array with input mask
       edge_rows: int - number of edge rows to mask
       edge_cols: int - number of edge columns to mask
    """

    assert isinstance(mask, np.ndarray), 'input mask should be numpy array'

    erows = edge_rows
    ecols = edge_cols

    sh = mask.shape
    if mask.ndim < 2:
        raise ValueError('Input mask has less then 2-d, shape = %s' % str(sh))

    mask_out = np.asarray(mask, dtype)

    # print 'shape:', sh

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


def merge_masks(mask1=None, mask2=None, dtype=DTYPE_MASK):
    """Merging masks using np.logical_and rule: (0,1,0,1)^(0,0,1,1) = (0,0,0,1)
    """
    assert mask1.size == mask2.size, 'Mask sizes should be equal'

    if mask1 is None: return mask2
    if mask2 is None: return mask1

    if mask1.shape != mask2.shape:
        if mask1.ndim > mask2.ndim: mask2.shape = mask1.shape
        else                      : mask1.shape = mask2.shape

    mask = np.logical_and(mask1, mask2)
    return mask if dtype==np.bool else np.asarray(mask, dtype)

# EOF

