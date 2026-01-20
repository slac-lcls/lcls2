"""
Utilities for area detector
===========================

Usage::

  from psana.detector.UtilsAreaDetector import dict_from_arr3d, arr3d_from_dict
  from psana.detector.UtilsAreaDetector import  *

  a = arr_rot_n90(arr, rot_ang_n90=0)
  sec, nsec = sec_nsec_from_tstamp(ts)
  d = dict_from_arr3d(a)
  a = arr3d_from_dict(d, keys=None)
  img_sta, multinds, nentries = statistics_of_pixel_arrays(rows, cols, rc_tot_max=None)
  img = img_from_pixel_arrays(rows, cols, weight=1.0, dtype=np.float32, vbase=0, rc_tot_max=None)
  img = img_multipixel_max(img, weight, dict_pix_to_img_idx)
  img_multipixel_max(img, weight, dict_pix_to_img_idx)
  img_multipixel_mean(img, weight, dict_pix_to_img_idx, dict_imgidx_numentries)
  size = size_for_shape(shape)
  a = ascending_index_array_for_shape(shape, dtype=np.int32)
  shape = image_shape(arr_rows, arr_cols, rc_tot_max=None) # rc_tot_max - maximal row and column for entire detector image
  img = image_of_pixel_array_ascending_index(rows, cols, img_shape=None, dtype=np.int32, rc_tot_max=None)
  img = image_of_pixel_seg_row_col(img_ascend_pix_ind, arr_shape, dtype=np.int32)
  img = image_of_holes(busy_img_bins)
  inds = hole_inds_ravel(img_holes)
  rows_cols = hole_rows_cols(img_holes)
  fill_holes(img, hrows, hcols)
  statistics_of_holes(rows, cols, **kwa)
  img = img_default(arr)

  #TBD init_interpolation_parameters(rows, cols, x, y, **kwa) # kwa['rc_tot_max'] - maximal row and column
  #TBD img = img_interpolated(data, interpol_pars, **kwa)

2020-11-06 created by Mikhail Dubrovin
"""

import logging
logger = logging.getLogger(__name__)

import numpy as np
from psana.detector.NDArrUtils import info_ndarr
from time import time

def arr_rot_n90(arr, rot_ang_n90=0):
    if   rot_ang_n90==  0: return arr
    elif rot_ang_n90== 90: return np.flipud(arr.T)
    elif rot_ang_n90==180: return np.flipud(np.fliplr(arr))
    elif rot_ang_n90==270: return np.fliplr(arr.T)
    else                 : return arr

def sec_nsec_from_tstamp(ts):
   nsec = ts & 0xffffffff
   sec = (ts >> 32) & 0xffffffff
   return sec, nsec

def dict_from_arr3d(a):
    """Converts 3d array of shape=(n0,n1,n2) to dict {k[0:n0-1] : nd.array.shape(n1,n2)}
       Consumes 25us for epix10ka2m array shape:(16, 352, 384) size:2162688 dtype:float32
    """
    assert isinstance(a, np.ndarray)
    assert a.ndim == 3
    return {k:a[k,:,:] for k in range(a.shape[0])}

def arr3d_from_list(lst):
    assert isinstance(lst, list)
    return np.stack(lst)

def arr3d_from_dict(d, keys=None):
    """Converts dict {k[0:n0-1] : nd.array.shape(n1,n2)} to 3d array of shape=(n0,n1,n2)
       Consumes 7ms for epix10ka2m array shape:(16, 352, 384) size:2162688 dtype:float32
    """
    assert isinstance(d, dict)
    _keys = sorted(d.keys() if keys is None else keys)
    return np.stack([d[k] for k in _keys])

def statistics_of_pixel_arrays(rows, cols, rc_tot_max=None):
    """Returns:
       - 2-d image shaped numpy array with statistics of overlapped data pixels,
       - dict for multiple entries: {<pixel-index-in-data-array> : <pixel-index-on-image>} for ravel arrays
       - dict with number of entries: {<pixel-index-on-image> : <number-of-entries gt.1>} for ravel image array
    """
    assert isinstance(rows, np.ndarray)
    assert isinstance(cols, np.ndarray)
    assert rows.size == cols.size

    img_shape = nrows, ncols = image_shape(rows, cols, rc_tot_max)

    zipped_rows_cols = zip(rows.ravel(), cols.ravel())

    t0_sec = time()
    img_sta = np.zeros(img_shape, dtype=np.uint16)
    for r,c in zipped_rows_cols: img_sta[r,c]+=1 # 1.8 sec
    # DOES NOT WORK FOR MULTI-ENTRIES... img_sta[rows.ravel(),cols.ravel()] += np.ones(rows.size, dtype=np.uint16) # 11ms
    logger.debug('statistics_of_pixel_arrays consumed time (sec) = %.6f' % (time()-t0_sec)) # 1.8 sec
    logger.debug('np.bincount(img_sta): %s' % str(np.bincount(img_sta.ravel(), minlength=10)))

    t0_sec = time()
    cond = img_sta>1
    #inds_img_multiple = np.extract(cond.ravel(), np.arange(img_sta.size))
    #print(info_ndarr(inds_img_multiple, 'inds_img_multiple:'))

    nrows, ncols = img_sta.shape
    multinds = {i:int(r*ncols+c) for i,(r,c) in enumerate(zipped_rows_cols) if cond[r,c]}
    #logger.debug('XXX dict multinds production time (sec) = %.6f' % (time()-t0_sec)) # 170us
    #exit('TEST EXIT') #############
    s = '\n multiple mapping of pixels to image:'
    for k,v in multinds.items(): s += '\n  pix:%06d img:%06d' % (k,v)
    logger.debug(s)

    # count number of entries in overlapping image pixels for epix10kaquad (4, 352, 384)
    # image bin scale size 100 - 11 multiple pixels
    # image bin scale size 101 - 10426 multiple pixels
    # image bin scale size 110 - 84571 multiple pixels

    from collections import Counter
    nentries = Counter(multinds.values())

    s = '\n number of multiple entries to image index:'
    for i,(k,n) in enumerate(nentries.items()): s += '\n  %03d img_ind:%06d entries:%d' % (i+1,k,n)
    logger.debug(s)

    return img_sta, multinds, nentries

def img_from_pixel_arrays(rows, cols, weight=None, dtype=np.float32, vbase=0, rc_tot_max=None):
    """Returns image from rows, cols index arrays and associated weights W.
       Methods like matplotlib imshow(img) plot 2-d image array oriented as matrix(rows,cols).
    """
    assert isinstance(rows, np.ndarray)
    assert isinstance(cols, np.ndarray)
    assert(isinstance(weight, (np.ndarray,float)))

    if rows.size != cols.size \
    or (weight is not None and rows.size != weight.size):
        msg = 'img_from_pixel_arrays(): input array sizes are different;' \
            + ' rows.size=%d, cols.size=%d, W.size=%d' % (rows.size, cols.size, weight.size)
        logger.warning(msg)
        return img_default(np.ones_like(rows, dtype=dtype))

    img_shape = image_shape(rows, cols, rc_tot_max)
    _weight = weight*np.ones_like(rows, dtype=dtype) if isinstance(weight, float) else\
              weight

    img = np.ones(img_shape, dtype=dtype)*vbase if vbase else\
         np.zeros(img_shape, dtype=dtype)

    t0_sec = time()

    img[rows.ravel(),cols.ravel()] = _weight.ravel()

    #for r,c,v in zip(rows.ravel(), cols.ravel(), _weight): img[r,c] = max(img[r,c],v)
    logger.debug('TIME img_from_pixel_arrays consumed time (sec) = %.6f' % (time()-t0_sec))

    return img

def img_multipixel_max(img, weight, dict_pix_to_img_idx):
    imgrav = img.ravel() # ravel() does not copy like ravel()
    if isinstance(dict_pix_to_img_idx, (tuple, list)) and len(dict_pix_to_img_idx) == 2:
        pix_idx, img_idx = dict_pix_to_img_idx
        if pix_idx.size:
            np.maximum.at(imgrav, img_idx, weight.ravel()[pix_idx])
        return
    for ia,i in dict_pix_to_img_idx.items(): imgrav[i] = max(imgrav[i], weight.ravel()[ia])

    if logger.getEffectiveLevel()<=logging.DEBUG: #logger.level
        s = '\n  == img_multipixel_max cross-check'
        for ia,i in dict_pix_to_img_idx.items():
            s += '\n  inds in img:%06d in pixarr:%06d value: %.1f' % (i, ia, weight.ravel()[ia])
        s += '\n  == img_multipixel_max result:'
        for i in sorted(set(dict_pix_to_img_idx.values())):
            s += '\n  inds in img:%06d max: %.1f' % (i, imgrav[i])
        logger.debug(s)
    #return img

def img_multipixel_mean(img, weight, dict_pix_to_img_idx, dict_imgidx_numentries):
    imgrav = img.ravel()
    if (
        isinstance(dict_pix_to_img_idx, (tuple, list))
        and len(dict_pix_to_img_idx) == 2
        and isinstance(dict_imgidx_numentries, (tuple, list))
        and len(dict_imgidx_numentries) == 2
    ):
        pix_idx, img_idx = dict_pix_to_img_idx
        imgidx, nentries = dict_imgidx_numentries
        if imgidx.size:
            imgrav[imgidx] = 0
            np.add.at(imgrav, img_idx, weight.ravel()[pix_idx])
            imgrav[imgidx] /= nentries
        return
    imgidx = list(dict_imgidx_numentries.keys())
    imgrav[imgidx] = 0                                               # initialization
    for ia,i in dict_pix_to_img_idx.items(): imgrav[i] += weight.ravel()[ia] # accumulation
    imgrav[imgidx] /= list(dict_imgidx_numentries.values())          # normalization

    if logger.getEffectiveLevel()<=logging.DEBUG: #logger.level
        s = '\n  == img_multipixel_mean cross-check'
        for ia,i in dict_pix_to_img_idx.items():
            s += '\n  inds in img:%06d in pixarr:%06d value: %.1f' % (i, ia, weight.ravel()[ia])
        s += '\n  == img_multipixel_mean result:'
        for i in imgidx:
            s += '\n  inds in img:%06d mean: %.1f for %d entries' % (i, imgrav[i], dict_imgidx_numentries[i])
        logger.debug(s)
    #return img

def size_for_shape(shape): return np.empty(shape).size

def ascending_index_array_for_shape(shape, dtype=np.int32):
    """returns ascending index [0,size-1] array of given shape"""
    a = np.arange(size_for_shape(shape), dtype=dtype)
    a.shape = shape
    return a

def image_shape(rows, cols, rc_tot_max=None):
    """defines image shape from appays of pixel rows and cols in image"""
    rmax, cmax = (rows.max(), cols.max()) if rc_tot_max is None else rc_tot_max
    return int(rmax)+1, int(cmax)+1

def image_of_pixel_array_ascending_index(rows, cols, img_shape=None, dtype=np.int32, rc_tot_max=None):
    """ returns image-shaped array containing pixel array ascending index
        rows and cols - pixel arrays shaped as data, i.e. for epix10ks... (<n-segments>, 352, 384)
        consumed time 7ms
        NOTE: some indices may be missing due to overlap - last index is retained in image
    """
    shape = image_shape(rows, cols, rc_tot_max) if img_shape is None else img_shape
    img = -np.ones(shape, dtype=dtype)
    img[rows.ravel(), cols.ravel()] = np.arange(rows.size, dtype=dtype) # 7ms
    #for i,(r,c) in enumerate(zip(rows.ravel(), cols.ravel())): img[r,c]=i # 250ms
    return img

def image_of_pixel_seg_row_col(img_ascend_pix_ind, arr_shape, dtype=np.int32):
    """returns image size array of pixel (seg,row,col), shape = (<image-size>,3) # 47ms"""
    imgind_to_seg_row_col = -np.ones((img_ascend_pix_ind.size,3), dtype=dtype)
    inds = img_ascend_pix_ind[img_ascend_pix_ind>-1] # img indexes of non-empty bins
    imgind_to_seg_row_col[inds] = np.array(np.unravel_index(inds.ravel(), arr_shape)).T
    return imgind_to_seg_row_col

def image_of_holes(busy_img_bins):
    """Works with image-shaped arrays.
       Returns image of holes marked by True values.
    """
    nonem = busy_img_bins
    nrows, ncols = shape = nonem.shape
    empty = np.logical_not(nonem)   #  80us
    empty[0:nrows-1,:] = np.logical_and(empty[0:nrows-1,:], nonem[1:nrows,:])   # 340us
    empty[1:nrows,:]   = np.logical_and(empty[1:nrows,:],   nonem[0:nrows-1,:]) # 440us
    empty[:,0:ncols-1] = np.logical_and(empty[:,0:ncols-1], nonem[:,1:ncols])   # 626us
    empty[:,1:ncols]   = np.logical_and(empty[:,1:ncols],   nonem[:,0:ncols-1]) # 810us
    return empty

def hole_inds_ravel(img_holes):
    return np.where(img_holes.ravel())[0] # [0] because np.where returns tuple for all axes

def hole_rows_cols(img_holes):
    """Returns hole rows and cols on image shaped array"""
    return np.where(img_holes)

def fill_holes(img, hrows, hcols):
    img[hrows, hcols] = np.minimum(\
             np.minimum(img[hrows-1, hcols], img[hrows+1, hcols]),
             np.minimum(img[hrows, hcols-1], img[hrows, hcols+1]))
    #return img

def statistics_of_holes(rows, cols, **kwa):
    """generates and returns a few useful arrays
       img_pix_ascend_ind - image-shaped [int32] contains ravel index in pixel (data) array
       img_holes - image-shaped [bool]
       hole_rows, hole_cols - arrays of hole rows and cols in image
    """
    assert isinstance(rows, np.ndarray)
    assert isinstance(cols, np.ndarray)
    assert rows.size == cols.size
    rc_tot_max=kwa.get('rc_tot_max', None)
    arr_shape = rows.shape
    img_shape = nrows, ncols = image_shape(rows, cols, rc_tot_max)
    img_size = nrows * ncols
    # make img_pix_ascend_ind mapping [r,c] to index in pixelarray
    t0_sec = time()
    img_pix_ascend_ind = image_of_pixel_array_ascending_index(rows, cols, img_shape, np.int32, rc_tot_max)
    logger.debug('statistics_of_holes.image_of_pixel_array_ascending_index time (sec) = %.6f' % (time()-t0_sec)) # 8ms
    logger.debug(info_ndarr(img_pix_ascend_ind, ' img_pix_ascend_ind:'))

    busy_img_bins = img_pix_ascend_ind>-1
    n_img_pixbins = np.sum(busy_img_bins) #np.count_nonzero(busy_img_bins)
    fr_noon_empty = float(n_img_pixbins)/busy_img_bins.size
    logger.debug('statistics_of_holes busy image bins/total: %d/%d = %.3f' %(n_img_pixbins, busy_img_bins.size, fr_noon_empty))

    t0_sec = time()
    img_holes = image_of_holes(busy_img_bins)
    logger.debug('statistics_of_holes hole finding time (sec) = %.6f' % (time()-t0_sec))
    logger.debug('statistics_of_holes number of holes = %d' % np.sum(img_holes))

    hole_inds1d = hole_inds_ravel(img_holes) #np.where(img_holes.ravel())[0] # [0] because np.where returns tuple
    logger.debug('statistics_of_holes hole indexes ravel = %s' % str(hole_inds1d))
    logger.debug(info_ndarr(hole_inds1d, 'hole_inds1d:'))

    hole_rows, hole_cols = hole_rows_cols(img_holes)
    logger.debug(info_ndarr(hole_rows, 'statistics_of_holes hole hrows:',last=10))
    logger.debug(info_ndarr(hole_cols, 'statistics_of_holes hole hcols:',last=10))

    return img_pix_ascend_ind, img_holes, hole_rows, hole_cols, hole_inds1d

def img_default(arr):
    med = np.median(arr)
    spr = np.median(np.abs(arr-med))
    amin, amax = med-1*spr, med+3*spr
    if amin == amax: amax = amin + 1
    #logger.debug('XXXX amin:%.1f, amax::%.1f' % (amin, amax))
    a = np.arange(amin, amax, (amax-amin)/12., dtype=np.float32)
    a.shape =(3,4)
    return a

def init_interpolation_parameters(rows, cols, x, y, **kwa):
    """TBD: currently returns image of ascending index in data array
       kwa['rc_tot_max'] - maximal row and column for entire detector image
    """
    assert isinstance(rows, np.ndarray)
    assert isinstance(cols, np.ndarray)
    assert rows.size == cols.size
    rc_tot_max = kwa.get('rc_tot_max', None)
    #address_table_4 = -np.ones((nrows,ncols,4), dtype=np.int32)
    #weights_table_4 = np.zeros((nrows,ncols,4), dtype=np.int32)
    img_shape = image_shape(rows, cols, rc_tot_max=kwa.get('rc_tot_max', None))
    return image_of_pixel_array_ascending_index(rows, cols, img_shape, np.int32, rc_tot_max)

def img_interpolated(data, interpol_pars, **kwa):
    """Image inperpolation.
       For each element of the uniform image matrix
       use the 1, x, y, x*y weights of 4 real neighbor pixels.
       Interpolation algorithm assumes the 4-node formula:

                 Term                   Weight
       f(x,y) =  f00                    1
              + (f10-f00)*x             x
              + (f01-f00)*y             y
              + (f11+f00-f10-f01)*x*y   x*y
    """
    return interpol_pars # img_default(data)

# EOF
