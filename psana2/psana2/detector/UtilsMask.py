
"""
Utilities for mask
==================

Usage::
  from psana2.detector.UtilsMask import *

  # Test: python <path>/lcls2/psana/psana/detector/testman/test_UtilsMask.py <test-number>

  m = merge_masks(mask1=None, mask2=None, dtype=DTYPE_MASK)
  m = merge_mask_for_grinds(mask, gain_range_inds=(0,1,2,3,4), dtype=DTYPE_MASK)
  s = merge_status_for_grinds(status, gain_range_inds=(0,1,2,3,4), dtype=DTYPE_STATUS)
         # merges status.shape=(7, 16, 352, 384) to s.shape=(16, 352, 384) of dtype
         # gain_range_inds list stands for gain ranges for 'FH','FM','FL','AHL-H','AML-M'
  m = mask_neighbors(mask, rad=5, ptrn='r')
  m = mask_edges(mask, edge_rows=1, edge_cols=1, dtype=DTYPE_MASK)
  m = status_as_mask(status, status_bits=(1<<64)-1, dtype=DTYPE_MASK, **kwa)

  # 2023-02-10 add converters from mask2d to mask3d
  m = convert_mask2d_to_ndarray_using_pixel_coord_indexes(mask2d, ix, iy)
  m = convert_mask2d_to_ndarray_using_geo(mask2d, geo, **kwargs) # kwargs passed to geo.get_pixel_coord_indexes(**kwargs)
  m = convert_mask2d_to_ndarray_using_geometry_file(mask2d, gfname, **kwargs) # kwargs passed to geo.get_pixel_coord_indexes(**kwargs)

  # 2023-02-23 add methods to generate masks for shape parameters
  r = cart2r(x, y)  # converts numpy arrays for carthesian x,y to r
  rows, cols = meshgrids(shape)
  m = mask_circle(shape, center_row, center_col, radius, dtype=DTYPE_MASK)
  m = mask_ring(shape, center_row, center_col, radius_min, radius_max, dtype=DTYPE_MASK)
  m = mask_rectangle(shape, cmin, rmin, cols, rows, dtype=DTYPE_MASK)
  m = mask_poly(shape, colrows, dtype=DTYPE_MASK)
  m = mask_halfplane(shape, r1, c1, r2, c2, rm, cm, dtype=DTYPE_MASK)
  m = mask_arc(shape, cx, cy, ro, ri, ao, ai, dtype=DTYPE_MASK)

2021-01-25 created by Mikhail Dubrovin
"""
import logging
logger = logging.getLogger(__name__)
import numpy as np
import psana2.pscalib.calib.CalibConstants as CC
DTYPE_MASK   = CC.dic_calib_type_to_dtype[CC.PIXEL_MASK]   # np.uint8
DTYPE_STATUS = CC.dic_calib_type_to_dtype[CC.PIXEL_STATUS] # np.uint64
from psana2.detector.NDArrUtils import info_ndarr, shape_nda_as_3d, reshape_to_3d # shape_as_3d, shape_as_3d


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


def merge_mask_for_grinds(mask, gain_range_inds=(0,1,2,3,4), dtype=DTYPE_MASK):
    """Merges mask bits over gain range index.
       gain_range_inds list(uint) - list of gain range inices in array mask[i,:]
       gain_range_inds=(0,1,2,3,4) for epix10ka/quad/2m mask array mask.shape=(7, <num-segments>, 352, 384) merging to (<num-segments>, 352, 384)
       gain_range_inds=(0,1,2) for Jungfrau mask array mask.shape=(3, <num-segments>, 512, 512) merging to (<num-segments>, 512, 512)
    """
    if mask.ndim < 4: return mask # ignore 3-d arrays
    _mask = mask.astype(dtype)
    mask1 = np.copy(_mask[gain_range_inds[0],:])
    for i in gain_range_inds[1:]:
        if i<mask.shape[0]:
            cond = np.logical_and(mask1, _mask[i,:]) #, out=mask1)
            mask1 = np.asarray(np.select((cond,), (1,), default=0), dtype=dtype)
    return mask1


def merge_status_for_grinds(status, gain_range_inds=(0,1,2,3,4), dtype=DTYPE_STATUS):
    """Merges status bits over gain range index.
       Originaly intended for epix10ka(quad/2m) status array status.shape=(7, 16, 352, 384) merging to (16, 352, 384)
       Also can be used with Jungfrau status array status.shape=(7, 8, 512, 512) merging to (8, 512, 512)
       option "indexes" contains a list of status[i,:] indexes to combine status
    """
    if status.ndim < 2: return status # ignore 1-d arrays
    _status = status.astype(dtype)
    st1 = np.copy(_status[gain_range_inds[0],:])
    for i in gain_range_inds[1:]: # range(1,status.shape[0]):
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


def status_as_mask(status, status_bits=(1<<64)-1, dtype=DTYPE_MASK, **kwa):
    """Returns per-pixel array of mask generated from pixel_status.

       Parameters

       - status  : np.array - pixel_status/status_extra/etc calibration constants
       - status_bits : bitword for mask status codes
       - dtype : mask np.array dtype

       Returns

       - np.array - mask generated from calibration type pixel_status/status_extra (1/0 for status 0/status_bits>0, respectively).
    """
    if not isinstance(status, np.ndarray):
            logger.debug('status is not np.ndarray - return None')
            return None

    from psana2.detector.NDArrUtils import info_ndarr
    logger.debug(info_ndarr(status, 'status'))
    cond = (status & status_bits)>0
    return np.asarray(np.select((cond,), (0,), default=1), dtype=dtype)


def convert_mask2d_to_ndarray_using_pixel_coord_indexes(mask2d, rows, cols):
    """Converts 2-d (np.ndarray) image-like mask2d to
       (np.ndarray) shaped as input pixel index arrays rows and cols.
       NOTE: arrays rows and cols should be exactly the same as used to construct mask2d as image.
    """
    from psana2.pscalib.geometry.GeometryAccess import convert_mask2d_to_ndarray
    return convert_mask2d_to_ndarray(mask2d, rows, cols, dtype=DTYPE_MASK)


def convert_mask2d_to_ndarray_using_geo(mask2d, geo, **kwargs):
    """Converts 2-d (np.ndarray) image-like mask2d to 3-d (np.ndarray) shaped as raw data.
       geo (GeometryAccess) is a geometry object.
       **kwargs - keyword arguments passed to geo.get_pixel_coord_indexes().
    """
    from psana2.pscalib.geometry.GeometryAccess import GeometryAccess
    assert isinstance(geo, GeometryAccess)
    ir, ic = geo.get_pixel_coord_indexes(**kwargs)
    reshape_to_3d(ir)
    reshape_to_3d(ic)
    assert ir.shape[0]>1, 'number of segments should be more than one, ir.shape=%s' % str(ir.shape)
    assert ic.shape[0]>1, 'number of segments should be more than one, ic.shape=%s' % str(ic.shape)
    return convert_mask2d_to_ndarray_using_pixel_coord_indexes(mask2d, ir, ic)


def convert_mask2d_to_ndarray_using_geometry_file(mask2d, gfname, **kwargs):
    """Converts 2-d (np.ndarray) image-like mask2d to 3-d (np.ndarray) shaped as raw data.
       gfname (str) is a geometry file name.
       **kwargs - keyword arguments passed to geo.get_pixel_coord_indexes().
    """
    from psana2.pscalib.geometry.GeometryAccess import GeometryAccess
    assert isinstance(gfname, str)
    assert os.path.exists(gfname)
    geo = GeometryAccess(gfname)
    return convert_mask2d_to_ndarray_using_geo(mask2d, geo, **kwargs)


def cart2r(x, y):
    return np.sqrt(x*x + y*y)


def meshgrids(shape):
    """returns np.meshgrid arrays of cols, rows for specified shape"""
    assert len(shape)==2
    return np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))


def mask_circle(shape, center_row, center_col, radius, dtype=DTYPE_MASK):
    c, r = meshgrids(shape)
    rad = cart2r(r-center_row, c-center_col)
    return np.select([rad>radius,], [0,], default=1).astype(dtype)


def mask_ring(shape, center_row, center_col, radius_min, radius_max, dtype=DTYPE_MASK):
    c, r = meshgrids(shape)
    rad = cart2r(r-center_row, c-center_col)
    return np.select([rad<radius_min, rad>radius_max,], [0, 0,], default=1).astype(dtype)


def mask_rectangle(shape, cmin, rmin, cols, rows, dtype=DTYPE_MASK):
    """cmin, rmin (int) - minimal coordinates of the rectangle corner
       cols, rows (int) - width and height of the rectangle
    """
    c, r = meshgrids(shape)
    return np.select([c<cmin, c>cmin+cols, r<rmin, r>rmin+rows], [False, False, False, False], default=True).astype(dtype)


def mask_poly(shape, colrows, dtype=DTYPE_MASK):
    """colrows (list) - list of vertex coordinate pairs as (row,col)
    """
    c, r = meshgrids(shape)
    cr = list(zip(c.ravel(), r.ravel())) # or np.vstack((x,y)).T
    from matplotlib.path import Path
    mask = np.array(Path(colrows).contains_points(cr), dtype=dtype)
    mask.shape = shape
    return mask


def mask_halfplane(shape, r1, c1, r2, c2, rm, cm, dtype=DTYPE_MASK):
    """Half-plane contains the boarder line through the points (r1, c1) and (r2, c2)
       Off-line point (rm, cm) picks the half-plane marked with ones.
    """
    f = 0 if c1 == c2 else (r2-r1)/(c2-c1)
    c, r = meshgrids(shape)
    signgt = rm > r1+f*(cm-c1)
    cond = (r > r1 if rm < r1 else r < r1) if r1 == r2 else\
           (c > c1 if cm < c1 else c < c1) if c1 == c2 else\
           ((r > r1+f*(c-c1)) if signgt else (r < r1+f*(c-c1)))
    #rm, cm = int(rm), int(cm)
    #if not cond[rm, cm]: cond = ~cond
    return np.select([cond,], [0,], default=1).astype(dtype)


def mask_arc(shape, cx, cy, ro, ri, ao, ai, dtype=DTYPE_MASK):
    """Returns arc mask for ami2 ArcROI set of parameters. Ones in the arc zeroes outside.
       Carthesian (x,y) emulated by returned *.T as in ami.
       cx, cy - arc center (col, row)
       ro, ri - radii of the outer and inner arc corner points
       ao, ai - angles of the outer and arc angular size from outer to inner corner points
    """
    logger.debug('shape:%s  cx:%.2f  cy:%.2f  ro:%.2f  ri:%.2f  ao:%.2f  ai:%.2f' % (str(shape), cx, cy, ro, ri, ao, ai))
    from math import radians, sin, cos  # floor, ceil
    assert ro>ri, 'outer radius %d shold be greater than inner %d' % (ro, ri)
    assert ai>0, 'arc span angle %.2f deg > 0' % ai
    #assert ao>0, 'outer arc corner angle %.2f deg > 0' % ao
    row1, col1 = cy, cx
    mring = mask_ring(shape, row1, col1, ri, ro, dtype=dtype)
    ao_rad = radians(ao)
    ai_rad = radians(ao + ai)
    delta = -0.1 # radian
    row2, col2 = row1 + ro * sin(ao_rad), col1 + ro * cos(ao_rad)
    rm, cm = row1 + ro * sin(ao_rad+delta), col1 + ro * cos(ao_rad+delta)
    mhpo = mask_halfplane(shape, row1, col1, row2, col2, rm, cm, dtype=dtype)
    row2, col2 = row1 + ri * sin(ai_rad), col1 + ri * cos(ai_rad)
    rm, cm = row1 + ri * sin(ai_rad-delta), col1 + ri * cos(ai_rad-delta)
    mhpi = mask_halfplane(shape, row1, col1, row2, col2, rm, cm, dtype=dtype)
    mhro = merge_masks(mask1=mring, mask2=mhpo, dtype=dtype)
    mhri = merge_masks(mask1=mring, mask2=mhpi, dtype=dtype)
    return (np.bitwise_and(mhro, mhri) if ai<180 else np.bitwise_or(mhro, mhri)).T

# EOF

