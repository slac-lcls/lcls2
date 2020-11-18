"""
Utilities for area detector.

from psana.detector.UtilsAreaDetector import dict_from_arr3d, arr3d_from_dict

2020-11-06 created by Mikhail Dubrovin
"""

#from psana.pyalgos.generic.NDArrUtils import shape_as_2d, shape_as_3d, reshape_to_2d, reshape_to_3d
import logging
logger = logging.getLogger(__name__)

import numpy as np
from psana.pyalgos.generic.NDArrUtils import info_ndarr
from time import time

def arr_rot_n90(arr, rot_ang_n90=0) :
    if   rot_ang_n90==  0 : return arr
    elif rot_ang_n90== 90 : return np.flipud(arr.T)
    elif rot_ang_n90==180 : return np.flipud(np.fliplr(arr))
    elif rot_ang_n90==270 : return np.fliplr(arr.T)
    else                  : return arr


def dict_from_arr3d(a):
    """Converts 3d array of shape=(n0,n1,n2) to dict {k[0:n0-1] : nd.array.shape(n1,n2)}
       Consumes 25us for epix10ka2m array shape:(16, 352, 384) size:2162688 dtype:float32
    """
    assert isinstance(a, np.ndarray)
    assert a.ndim == 3
    return {k:a[k,:,:] for k in range(a.shape[0])}


def arr3d_from_dict(d, keys=None):
    """Converts dict {k[0:n0-1] : nd.array.shape(n1,n2)} to 3d array of shape=(n0,n1,n2)
       Consumes 7ms for epix10ka2m array shape:(16, 352, 384) size:2162688 dtype:float32
    """
    assert isinstance(d, dict)
    _keys = sorted(d.keys() if keys is None else keys)
    return np.stack([d[k] for k in _keys])


def statistics_of_pixel_arrays(rows, cols):
    """Returns:
       - 2-d image shaped numpy array with statistics of overlapped data pixels,
       - dict for multiple entries: {<pixel-index-in-data-array> : <pixel-index-on-image>} for flatten arrays
       - dict with number of entries: {<pixel-index-on-image> : <number-of-entries gt.1>} for flatten image array
    """
    assert isinstance(rows, np.ndarray)
    assert isinstance(cols, np.ndarray)
    assert rows.size == cols.size

    rowsfl = rows.flatten()
    colsfl = cols.flatten()

    rsize = int(rowsfl.max())+1 
    csize = int(colsfl.max())+1
         
    t0_sec = time()
    img_sta = np.zeros((rsize+1,csize+1), dtype=np.uint16)
    for r,c in zip(rowsfl, colsfl): img_sta[r,c]+=1
    dt_sec = time()-t0_sec
    logger.info('XXX statistics_of_pixel_arrays consumed time (sec) = %.6f' % dt_sec)
    # DOES NOT WORK: img_sta[rowsfl,colsfl] += 1 
    logger.info('XXX np.bincount(img_sta): %s' % str(np.bincount(img_sta.flatten(), minlength=10)))

    cond = img_sta>1
    #inds_img_multiple = np.extract(cond.flatten(), np.arange(img_sta.size))
    #print(info_ndarr(inds_img_multiple, 'inds_img_multiple:'))
    #print(info_ndarr(rowsfl, 'rowsfl:'))
    #print(info_ndarr(colsfl, 'colsfl:'))

    nrows, ncols = img_sta.shape
    multinds = {i:int(r*ncols+c) for i,(r,c) in enumerate(zip(rowsfl, colsfl)) if cond[r,c]}

    s = '\n multiple mapping of pixels to image:'
    for k,v in multinds.items(): s += '\n  pix:%06d img:%06d' % (k,v)
    logger.info(s)

    # count number of entries in overlapping image pixels for epix10kaquad (4, 352, 384)
    # image bin scale size 100 - 11 multiple pixels
    # image bin scale size 101 - 10426 multiple pixels
    # image bin scale size 110 - 84571 multiple pixels

    from collections import Counter
    nentries = Counter(multinds.values())

    s = '\n number of multiple entries to image index:'
    for i,(k,n) in enumerate(nentries.items()): s += '\n  %03d img_ind:%06d entries:%d' % (i+1,k,n)
    logger.info(s)

    return img_sta, multinds, nentries


def img_from_pixel_arrays(rows, cols, weight=1.0, dtype=np.float32, vbase=0):
    """Returns image from rows, cols index arrays and associated weights W.
       Methods like matplotlib imshow(img) plot 2-d image array oriented as matrix(rows,cols).
    """
    assert isinstance(rows, np.ndarray)
    assert isinstance(cols, np.ndarray)
    assert (isinstance(weight, (np.ndarray,float)))

    if rows.size != cols.size \
    or (weight is not None and rows.size !=  weight.size):
        msg = 'img_from_pixel_arrays(): input array sizes are different;' \
            + ' rows.size=%d, cols.size=%d, W.size=%d' % (rows.size, cols.size, W.size)
        logger.warning(msg)
        return img_default()

    rowsrav = rows.ravel() # ravel does not copy...
    colsrav = cols.ravel()

    rsize = int(rowsrav.max())+1 
    csize = int(colsrav.max())+1

    _weight = weight*np.ones_like(rowsrav, dtype=dtype) if isinstance(weight, float) else\
              weight.flatten()

    img = np.ones((rsize+1,csize+1), dtype=dtype)*vbase if vbase else\
         np.zeros((rsize+1,csize+1), dtype=dtype)

    t0_sec = time()

    img[rowsrav,colsrav] = _weight

    #for r,c,v in zip(rowsrav, colsrav, _weight): img[r,c] = max(img[r,c],v)
    dt_sec = time()-t0_sec
    logger.info('TIME img_from_pixel_arrays consumed time (sec) = %.6f' % dt_sec)

    return img


def img_multipixel_max(img, weight, dict_pix_to_img_idx):
    imgrav = img.ravel() # ravel() does not copy like flatten()
    for ia,i in dict_pix_to_img_idx.items(): imgrav[i] = max(imgrav[i], weight.ravel()[ia])

    if logger.getEffectiveLevel()<=logging.DEBUG: #logger.level
        s = '\n  == img_multipixel_max cross-check'
        for ia,i in dict_pix_to_img_idx.items(): 
            s += '\n  inds in img:%06d in pixarr:%06d value: %.1f' % (i, ia, weight.ravel()[ia])
        s += '\n  == img_multipixel_max result:'
        for i in sorted(set(dict_pix_to_img_idx.values())):
            s += '\n  inds in img:%06d max: %.1f' % (i, imgrav[i])
        logger.info(s)

    return img


def img_multipixel_mean(img, weight, dict_pix_to_img_idx, dict_imgidx_numentries):
    imgrav = img.ravel()
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
        logger.info(s)

    return img
    

def img_interpolated(**kwa):
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
    return np.array(range(12), shape(3,4), dtype=np.float32)

#----

