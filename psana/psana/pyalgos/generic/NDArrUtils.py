"""
:py:class:`NDArrUtils` - a set of utilities working on numpy arrays
=====================================================================

Usage::

    # test: python psana/psana/pyalgos/generic/NDArrUtils.py 12

    # assuming that $PYTHONPATH=.../lcls2/psana
    # Import
    import psana.pyalgos.generic.NDArrUtils as gu

    # Methods
    #resp = gu.<method(pars)>

    gu.print_ndarr(nda, name='', first=0, last=5)
    s = gu.info_ndarr(nda, name='', first=0, last=5)

    shape = (32,185,388)
    size  = size_from_shape(shape) # returns 32*185*388
    shp2d = shape_as_2d(shape) # returns (32*185,388)

    shape = (4,8,185,388)
    shp3d = shape_as_3d(shape) # returns (32,185,388)

    shp2d = gu.shape_nda_as_2d(nda)
    shp3d = gu.shape_nda_as_3d(nda)

    arr2d = gu.reshape_to_2d(nda)
    arr3d = gu.reshape_to_3d(nda)

    arot  = arr_rot_n90(arr, rot_ang_n90=0)

    mmask = gu.merge_masks(mask1=None, mask2=None, dtype=np.uint8)
    mask  = gu.mask_neighbors(mask_in, allnbrs=True, dtype=np.uint8)
    mask  = gu.mask_neighbors_in_radius(mask, rad=5, ptrn='r')
    mask  = gu.mask_edges(mask, mrows=1, mcols=1, dtype=np.uint8)
    mask  = gu.mask_2darr_edges(shape=(185,388), width=2)
    mask  = gu.mask_3darr_edges(shape=(32,185,388), width=2)
    res   = gu.divide_protected(num, den, vsub_zero=0)


    # Make mask n-d numpy array using shape and windows
    # =================================================
    mask_xy_max = locxymax(data, order=1, mode='clip')

    shape = (2,185,388)
    w = 1
    winds = [(s, w, 185-w, w, 388-w) for s in (0,1)]
    mask = mask_from_windows(shape, winds)

    # Background subtraction
    # ======================
    # Example for cspad, assuming all nda_*.shape = (32,185,388)
    winds_bkgd = [(s, 10, 100, 270, 370) for s in (4,12,20,28)] # use part of segments 4,12,20, and 28 to subtract bkgd
    nda = subtract_bkgd(nda_data, nda_bkgd, mask=nda_mask, winds=winds_bkgd, pbits=0)

    gu.set_file_access_mode(fname, mode=0o664)
    gu.save_2darray_in_textfile(nda, fname, fmode, fmt, umask=0o0, group='ps-users')
    gu.save_ndarray_in_textfile(nda, fname, fmode, fmt, umask=0o0, group='ps-users')

See:
    - :py:class:`Utils`
    - :py:class:`NDArrUtils`
    - :py:class:`NDArrGenerators`

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created: 2018-01-25 by Mikhail Dubrovin
Adopted for LCLS2 on 2018-02-02
"""

import os
import sys

#from time import localtime, strftime, time

import numpy as np

from psana.pscalib.calib.NDArrIO import save_txt


import logging
logger = logging.getLogger('NDArrUtils')


def str_formatted(nda, first=0, last=5, vfmt='%0.6f', spa=' '):
    if vfmt is None:
        return '%s%s' % (str(nda.ravel()[first:last]).rstrip(']'), '...]' if nda.size>last else ']')
    #str(nda.ravel()[first:last])
    s = spa.join([vfmt % v for v in nda.ravel()[first:last]])
    suffix = ' ...]' if nda.size>last else ']'
    return '[%s%s' % (s.lstrip('\n'),suffix)


def info_ndarr(nda, name='', first=0, last=5, vfmt=None, spa=' ', sfmt=None): #' mean:%.3f median:%.3f min:%.3f max:%.3f'):
    """ optional: sfmt=' mean:%.3f median:%.3f min:%.3f max:%.3f'
    """
    _name = '%s '%name if name!='' else name
    s = ''
    gap = '\n' if (last-first)>10 else ' '
    if nda is None: s = '%sNone' % _name
    elif isinstance(nda, tuple): s += info_ndarr(np.array(nda), 'ndarray from tuple: %s' % name)
    elif isinstance(nda, list):  s += info_ndarr(np.array(nda), 'ndarray from list: %s' % name)
    elif not isinstance(nda, np.ndarray):
        s = '%s%s' % (_name, type(nda))
    else:
        a = '' if last == 0 else\
            str_formatted(nda, first=first, last=last, vfmt=vfmt, spa=spa)
        #    '%s%s' % (str(nda.ravel()[first:last]).rstrip(']'), '...]' if nda.size>last else ']')
        sstat = '' if sfmt is None else sfmt % (np.mean(nda), np.median(nda), nda.min(), nda.max())
        s = '%sshape:%s size:%d%s dtype:%s%s%s' %\
            (_name, str(nda.shape), nda.size, sstat, nda.dtype, gap, a)
    return s


def print_ndarr(nda, name=' ', first=0, last=5):
    print(info_ndarr(nda, name, first, last))


def size_from_shape(shape):
    """Returns size from the shape sequence
    """
    size=1
    for d in shape: size*=d
    return size


def size_from_shape_v2(arrsh):
    """Returns size from the shape sequence (list, tuple, np.array)
    """
    return np.prod(arrsh, axis=None, dtype=np.int32)


def shape_as_2d(sh):
    """Returns 2-d shape for n-d shape if ndim != 2, otherwise returns unchanged shape.
    """
    ndim = len(sh)
    if ndim>2: return (int(size_from_shape(sh)/sh[-1]), sh[-1])
    if ndim<2: return (1, sh[0])
    return sh


def shape_as_3d(sh):
    """Returns 3-d shape for n-d shape if ndim != 3, otherwise returns unchanged shape.
    """
    ndim = len(sh)
    if ndim >3: return (int(size_from_shape(sh)/sh[-1]/sh[-2]), sh[-2], sh[-1])
    if ndim==2: return (1, sh[0], sh[1])
    if ndim==1: return (1, 1, sh[0])
    return sh


def shape_nda_as_2d(arr):
    """Return shape of np.array to reshape to 2-d
    """
    return arr.shape if arr.ndim==2 else shape_as_2d(arr.shape)


def shape_nda_as_3d(arr):
    """Return shape of np.array to reshape to 3-d
    """
    return arr.shape if arr.ndim==3 else shape_as_3d(arr.shape)


def reshape_to_2d(arr):
    """Reshape np.array to 2-d if arr.ndim != 2.
    """
    if arr.ndim != 2: arr.shape = shape_nda_as_2d(arr)
    return arr


def reshape_to_3d(arr):
    """Reshape np.array to 3-d if arr.ndim != 3.
    """
    if arr.ndim != 3: arr.shape = shape_nda_as_3d(arr)
    return arr


def reshape_2d_to_3d(arr):
    """Reshape np.array from 2-d to 3-d. Accepts 2d arrays only.
    """
    if arr.ndim==2:
        sh = arr.shape
        arr.shape = (1,sh[-2],sh[-1])
    return arr


def arr_rot_n90(arr, rot_ang_n90=0):
    if   rot_ang_n90==  0: return arr
    elif rot_ang_n90== 90: return np.flipud(arr.T)
    elif rot_ang_n90==180: return np.flipud(np.fliplr(arr))
    elif rot_ang_n90==270: return np.fliplr(arr.T)
    else                 : return arr


#Aliases for compatability
reshape_nda_to_2d = reshape_to_2d
reshape_nda_to_3d = reshape_to_3d


def merge_masks(mask1=None, mask2=None, dtype=np.uint8):
    """Merging masks using np.logical_and rule: (0,1,0,1)^(0,0,1,1) = (0,0,0,1)
    """
    if mask1 is None: return mask2
    if mask2 is None: return mask1

    shape1 = mask1.shape
    shape2 = mask2.shape

    if shape1 != shape2:
        if len(shape1) > len(shape2): mask2.shape = shape1
        else                        : mask1.shape = shape2

    mask = np.logical_and(mask1, mask2)
    return mask if dtype==np.bool else np.asarray(mask, dtype)


def mask_neighbors(mask, allnbrs=True, dtype=np.uint8):
    """Return mask with masked eight neighbor pixels around each 0-bad pixel in input mask.

       mask   : int - n-dimensional (n>1) array with input mask
       allnbrs: bool - False/True - masks 4/8 neighbor pixels.
    """
    shape_in = mask.shape
    if len(shape_in) < 2:
        raise ValueError('Input mask has less then 2-d, shape = %s' % str(shape_in))

    mask_out = np.asarray(mask, dtype)

    if len(shape_in) == 2:
        # mask nearest neighbors
        mask_out[0:-1,:] = np.logical_and(mask_out[0:-1,:], mask[1:,  :])
        mask_out[1:,  :] = np.logical_and(mask_out[1:,  :], mask[0:-1,:])
        mask_out[:,0:-1] = np.logical_and(mask_out[:,0:-1], mask[:,1:  ])
        mask_out[:,1:  ] = np.logical_and(mask_out[:,1:  ], mask[:,0:-1])
        if allnbrs :
          # mask diagonal neighbors
          mask_out[0:-1,0:-1] = np.logical_and(mask_out[0:-1,0:-1], mask[1:  ,1:  ])
          mask_out[1:  ,0:-1] = np.logical_and(mask_out[1:  ,0:-1], mask[0:-1,1:  ])
          mask_out[0:-1,1:  ] = np.logical_and(mask_out[0:-1,1:  ], mask[1:  ,0:-1])
          mask_out[1:  ,1:  ] = np.logical_and(mask_out[1:  ,1:  ], mask[0:-1,0:-1])

    else : # shape>2

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


def mask_neighbors_in_radius(mask, rad=5, ptrn='r'):
    """In mask array increase region of masked pixels around bad by radial paramerer rad.
       Parameters:
       -----------
       - mask (np.ndarray) - input mask array ndim >=2
       - rad (int) - radial parameter of masked region
       - ptrn (char) - pattern of the masked region, for now ptrn='r' -rhombus, othervise square [-rad,+rad] in rows and columns.

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
        c1b, c1e = max(dc, 0), min(cols, cols+dc)
        c2b, c2e = max(-dc, 0), min(cols, cols-dc)
        if mask.ndim==2:
          mmask[r1b:r1e,c1b:c1e] = merge_masks(mmask[r1b:r1e,c1b:c1e], mask[r2b:r2e,c2b:c2e])
        else:
          mmask[:,r1b:r1e,c1b:c1e] = merge_masks(mmask[:,r1b:r1e,c1b:c1e], mask[:,r2b:r2e,c2b:c2e])
    #logger.info('mask_neighbors(rad=%d) time = %.3f sec' % (rad, time()-t0_sec))
    return mmask


def mask_edges(mask, mrows=1, mcols=1, dtype=np.uint8):
    """Return mask with a requested number of row and column pixels masked - set to 0.
       mask : int - n-dimensional (n>1) array with input mask
       mrows: int - number of edge rows to mask
       mcols: int - number of edge columns to mask
    """
    sh = mask.shape
    if len(sh) < 2:
        raise ValueError('Input mask has less then 2-d, shape = %s' % str(sh))

    mask_out = np.asarray(mask, dtype)

    # print('shape:', sh)

    if len(sh) == 2:
        rows, cols = sh

        if mrows > rows:
          raise ValueError('Requested number of edge rows=%d to mask exceeds 2-d, shape=%s' % (mrows, str(sh)))

        if mcols > cols:
          raise ValueError('Requested number of edge columns=%d to mask exceeds 2-d, shape=%s' % (mcols, str(sh)))

        if mrows>0:
          # mask edge rows
          mask_rows = np.zeros((mrows,cols), dtype=mask.dtype)
          mask_out[:mrows, :] = mask_rows
          mask_out[-mrows:,:] = mask_rows

        if mcols>0:
          # mask edge colss
          mask_cols = np.zeros((rows,mcols), dtype=mask.dtype)
          mask_out[:,:mcols]  = mask_cols
          mask_out[:,-mcols:] = mask_cols

    else: # shape>2
        mask_out.shape = shape_nda_as_3d(mask)

        segs, rows, cols = mask_out.shape

        if mrows > rows:
          raise ValueError('Requested number of edge rows=%d to mask exceeds 2-d, shape=%s' % (mrows, str(sh)))

        if mcols > cols:
          raise ValueError('Requested number of edge columns=%d to mask exceeds 2-d, shape=%s' % (mcols, str(sh)))

        if mrows>0:
          # mask edge rows
          mask_rows = np.zeros((segs,mrows,cols), dtype=mask.dtype)
          mask_out[:, :mrows,:] = mask_rows
          mask_out[:,-mrows:,:] = mask_rows

        if mcols>0:
          # mask edge colss
          mask_cols = np.zeros((segs,rows,mcols), dtype=mask.dtype)
          mask_out[:,:,:mcols]  = mask_cols
          mask_out[:,:,-mcols:] = mask_cols

        mask_out.shape = sh

    return mask_out


def mask_2darr_edges(shape=(185,388), width=2):
    """Returns mask with masked width rows/colunms of edge pixels for 2-d numpy array.
    """
    s, w = shape, width
    mask = np.zeros(shape, dtype=np.uint16)
    mask[w:-w,w:-w] = np.ones((s[0]-2*w,s[1]-2*w), dtype=np.uint16)
    return mask


def mask_3darr_edges(shape=(32,185,388), width=2):
    """Returns mask with masked width rows/colunms of edge pixels for 3-d numpy array.
    """
    s, w = shape, width
    mask = np.zeros(shape, dtype=np.uint16)
    mask[:,w:-w,w:-w] = np.ones((s[0],s[1]-2*w,s[2]-2*w), dtype=np.uint16)
    return mask


def divide_protected(num, den, vsub_zero=0):
    """Returns result of devision of numpy arrays num/den with substitution of value vsub_zero for zero den elements.
    """
    if num is None or den is None: return None
    pro_num = np.select((den!=0,), (num,), default=vsub_zero)
    pro_den = np.select((den!=0,), (den,), default=1)
    return pro_num / pro_den


def mask_from_windows(ashape=(32,185,388), winds=None):
    """Makes mask as 2-d or 3-d numpy array defined by the shape with ones in windows.
       N-d shape for N>3 is converted to 3-d.
       - param shape - shape of the output numpy array with mask.
       - param winds - list of windows, each window is a sequence of 5 parameters (segment, rowmin, rowmax, colmin, colmax)
    """
    ndim = len(ashape)

    if ndim<2:
        print('ERROR in mask_from_windows(...):',\
              ' Wrong number of dimensions %d in the shape=%s parameter. Allowed ndim>1.' % (ndim, str(shape)))
        return None

    shape = ashape if ndim<4 else shape_as_3d(ashape)

    seg1 = np.ones((shape[-2], shape[-1]), dtype=np.uint16) # shaped as last two dimensions
    mask = np.zeros(shape, dtype=np.uint16)

    if ndim == 2:
        for seg,rmin,rmax,cmin,cmax in winds: mask[rmin:rmax,cmin:cmax] =  seg1[rmin:rmax,cmin:cmax]
        return mask

    elif ndim == 3:
        for seg,rmin,rmax,cmin,cmax in winds: mask[seg,rmin:rmax,cmin:cmax] =  seg1[rmin:rmax,cmin:cmax]
        return mask


def list_of_windarr(nda, winds=None):
    """Converts 2-d or 3-d numpy array in the list of 2-d numpy arrays for windows
       - param nda - input 2-d or 3-d numpy array
    """
    ndim = len(nda.shape)
    #print('len(nda.shape): ', ndim)

    if ndim == 2:
        return [nda] if winds is None else \
               [nda[rmin:rmax, cmin:cmax] for (s, rmin, rmax, cmin, cmax) in winds]

    elif ndim == 3:
        return [nda[s,:,:] for s in range(ndim.shape[0])] if winds is None else \
               [nda[s, rmin:rmax, cmin:cmax] for (s, rmin, rmax, cmin, cmax) in winds]

    else:
        print('ERROR in list_of_windarr (with winds): Unexpected number of n-d array dimensions: ndim = %d' % ndim)
        return []


def mean_of_listwarr(lst_warr):
    """Evaluates the mean value of the list of 2-d arrays.
       - lst_warr - list of numpy arrays to evaluate per pixel mean intensity value.
    """
    s1, sa = 0., 0.
    for warr in lst_warr:
        sa += np.sum(warr, dtype=np.float64)
        s1 += warr.size
    return sa/s1 if s1 > 0 else 1


def subtract_bkgd(data, bkgd, mask=None, winds=None, pbits=0):
    """Subtracts numpy array of bkgd from data using normalization in windows for good pixels in mask.
       Shapes of data, bkgd, and mask numpy arrays should be the same.
       Each window is specified by 5 parameters: (segment, rowmin, rowmax, colmin, colmax)
       For 2-d arrays segment index is not used, but still 5 parameters needs to be specified.

       Parameters

       - data - numpy array for data.
       - bkgd - numpy array for background.
       - mask - numpy array for mask.
       - winds - list of windows, each window is a sequence of 5 parameters.
       - pbits - print control bits; =0 - print nothing, !=0 - normalization factor.
    """
    mdata = data if mask is None else data*mask
    mbkgd = bkgd if mask is None else bkgd*mask

    lwdata = list_of_windarr(mdata, winds)
    lwbkgd = list_of_windarr(mbkgd, winds)

    mean_data = mean_of_listwarr(lwdata)
    mean_bkgd = mean_of_listwarr(lwbkgd)

    frac = mean_data/mean_bkgd
    if pbits: print('subtract_bkgd, fraction = %10.6f' % frac)

    return data - bkgd*frac


# See: http://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.argrelmax.html#scipy.signal.argrelmax
def locxymax(nda, order=1, mode='clip'):
    """For 2-d or 3-d numpy array finds mask of local maxima in x and y axes (diagonals are ignored)
       using scipy.signal.argrelmax and return their product.

       - param nda - input ndarray
       - param order - range to search for local maxima along each dimension
       - param mode - parameter of scipy.signal.argrelmax of how to treat the boarder
    """
    shape = nda.shape
    size  = nda.size
    ndim = len(shape)

    if ndim< 2 or ndim>3:
        msg = 'ERROR: locxymax nda shape %s should be 2-d or 3-d' % (shape)
        sys.exit(msg)

    ext_cols = argrelmax(nda, -1, order, mode)
    ext_rows = argrelmax(nda, -2, order, mode)

    indc = np.array(ext_cols, dtype=np.uint16)
    indr = np.array(ext_rows, dtype=np.uint16)

    msk_ext_cols = np.zeros(shape, dtype=np.uint16)
    msk_ext_rows = np.zeros(shape, dtype=np.uint16)

    if ndim == 2:
        icr = indc[0,:]
        icc = indc[1,:]
        irr = indr[0,:]
        irc = indr[1,:]

        msk_ext_cols[icr,icc] = 1
        msk_ext_rows[irr,irc] = 1

    elif ndim == 3:
        ics = indc[0,:]
        icr = indc[1,:]
        icc = indc[2,:]
        irs = indr[0,:]
        irr = indr[1,:]
        irc = indr[2,:]

        msk_ext_cols[ics,icr,icc] = 1
        msk_ext_rows[irs,irr,irc] = 1

    #print('nda.size:',   nda.size)
    #print('indc.shape:', indc.shape)
    #print('indr.shape:', indr.shape)

    return msk_ext_rows * msk_ext_cols


def set_file_access_mode(fname, mode=0o664):
    os.chmod(fname, mode)


def change_file_ownership(fname, user=None, group='ps-users'):
    """change file ownership"""
    import grp
    import pwd
    gid = os.getgid() if group is None else grp.getgrnam(group).gr_gid
    uid = os.getuid() if user is None else pwd.getpwnam(user).pw_uid
    logger.debug('change_file_ownership uid:%d gid:%d' % (uid, gid)) # uid:5269 gid:10000
    os.chown(fname, uid, gid) # for non-default user - OSError: [Errno 1] Operation not permitted


def save_2darray_in_textfile(nda, fname, fmode, fmt, umask=0o0, group='ps-users', logmethod=logger.debug):
    os.umask(umask)
    fexists = os.path.exists(fname)
    np.savetxt(fname, nda, fmt=fmt)
    if not fexists:
        set_file_access_mode(fname, fmode)
        change_file_ownership(fname, user=None, group=group)
    logmethod('saved:  %s' % fname)


def save_ndarray_in_textfile(nda, fname, fmode, fmt, umask=0o0, group='ps-users'):
    os.umask(umask)
    fexists = os.path.exists(fname)
    save_txt(fname=fname, arr=nda, fmt=fmt)
    if not fexists:
        set_file_access_mode(fname, fmode)
        change_file_ownership(fname, user=None, group=group)
    logger.debug('saved: %s fmode: %s fmt: %s' % (fname, oct(fmode), fmt))


if __name__ == "__main__":
    sys.exit('See examples/test_NDArrUtils.py')

# EOF
