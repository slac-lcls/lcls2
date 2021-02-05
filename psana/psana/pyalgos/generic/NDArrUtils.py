"""
:py:class:`NDArrUtils` - a set of utilities working on numpy arrays
=================================================================

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

See:
    - :py:class:`Utils`
    - :py:class:`NDArrUtils`
    - :py:class:`NDArrGenerators`

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created: 2018-01-25 by Mikhail Dubrovin
Adopted for LCLS2 on 2018-02-02
"""

#import os
#from time import localtime, strftime, time

import numpy as np

#----

import logging
logger = logging.getLogger('NDArrUtils')

#----

def info_ndarr(nda, name=' ', first=0, last=5):
    s = ''
    gap = '\n' if last>10 else ' '
    if nda is None: s = '%sNone' % name
    elif isinstance(nda, tuple): s += info_ndarr(np.array(nda), 'ndarray from tuple: %s' % name)
    elif isinstance(nda, list) : s += info_ndarr(np.array(nda), 'ndarray from list: %s' % name)
    elif not isinstance(nda, np.ndarray):
                     s = '%s%s' % (name, type(nda))
    else: s = '%sshape:%s size:%d dtype:%s%s%s%s'%\
               (name, str(nda.shape), nda.size, nda.dtype, gap, str(nda.flatten()[first:last]).rstrip(']'),\
                ('...]' if nda.size>last else ']'))
    return s


def print_ndarr(nda, name=' ', first=0, last=5):
    print(info_ndarr(nda, name, first, last))

#    if nda is None: print('%s: %s', name, nda)
#    elif isinstance(nda, tuple): print_ndarr(np.array(nda), 'ndarray from tuple: %s' % name)
#    elif isinstance(nda, list) : print_ndarr(np.array(nda), 'ndarray from list: %s' % name)
#    elif not isinstance(nda, np.ndarray):
#                     print('%s: %s' % (name, type(nda)))
#    else          : print('%s:  shape:%s  size:%d  dtype:%s %s...'%\
#                           (name, str(nda.shape), nda.size, nda.dtype, nda.flatten()[first:last]))


def size_from_shape(shape):
    """Returns size from the shape sequence 
    """
    size=1
    for d in shape: size*=d
    return size


def size_from_shape_v2(arrsh):
    """Returns size from the shape sequence (list, tuple, np.array)
    """
    return np.prod(arrsh, axis=None, dtype=np.int)


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

#------------------------------
#----------- TEST -------------
#------------------------------

if __name__ == "__main__":

  def test_01():
    from psana.pyalgos.generic.NDArrGenerators import random_standard

    print('%s\n%s\n' % (80*'_','Test method subtract_bkgd(...):'))
    shape1 = (32,185,388)
    winds = [(s, 10, 155, 20, 358) for s in (0,1)]
    data = random_standard(shape=shape1, mu=300, sigma=50)
    bkgd = random_standard(shape=shape1, mu=100, sigma=10)
    cdata = subtract_bkgd(data, bkgd, mask=None, winds=winds, pbits=0o377)


  def test_02():
    from psana.pyalgos.generic.NDArrGenerators import random_standard
    shape1 = (32,185,388)
    data = random_standard(shape=shape1, mu=300, sigma=50)
    print(info_ndarr(data, 'test_02: info_ndarr', first=0, last=3))
    print(info_ndarr(shape1, 'test_02: info_ndarr'))


  def test_08():
    import psana.pyalgos.generic.Graphics as gg
    from psana.pyalgos.generic.NDArrGenerators import random_standard
    from psana.pyalgos.generic.NDArrUtils import reshape_to_2d

    print('%s\n%s\n' % (80*'_','Test method locxymax(nda, order, mode):'))
    #data = random_standard(shape=(32,185,388), mu=0, sigma=10)
    data = random_standard(shape=(2,185,388), mu=0, sigma=10)
    t0_sec = time()
    mask = locxymax(data, order=1, mode='clip')
    print('Consumed t = %10.6f sec' % (time()-t0_sec))

    if True:
      img = data if len(data.shape)==2 else reshape_to_2d(data)
      msk = mask if len(mask.shape)==2 else reshape_to_2d(mask)

      ave, rms = img.mean(), img.std()
      amin, amax = ave-2*rms, ave+2*rms
      gg.plotImageLarge(img, amp_range=(amin, amax), title='random')
      gg.plotImageLarge(msk, amp_range=(0, 1), title='mask loc max')
      gg.show()

 
  def test_mask_neighbors_2d(allnbrs=True):

    randexp = random_exponential(shape=(40,60), a0=1)
    fig  = gr.figure(figsize=(16,7), title='Random 2-d mask')
    axim1 = gr.add_axes(fig, axwin=(0.05,  0.05, 0.40, 0.91))
    axcb1 = gr.add_axes(fig, axwin=(0.452, 0.05, 0.01, 0.91))

    axim2 = gr.add_axes(fig, axwin=(0.55,  0.05, 0.40, 0.91))
    axcb2 = gr.add_axes(fig, axwin=(0.952, 0.05, 0.01, 0.91))

    mask = np.select((randexp>6,), (0,), default=1)
    mask_nbrs = mask_neighbors(mask, allnbrs)
    img1 = mask # mask # randexp
    img2 = mask_nbrs # mask # randexp
    
    imsh1, cbar1 = gr.imshow_cbar(fig, axim1, axcb1, img1, amin=0, amax=10, orientation='vertical', cmap='jet')
    imsh2, cbar2 = gr.imshow_cbar(fig, axim2, axcb2, img2,  amin=0, amax=10, orientation='vertical', cmap='jet')
    gr.show(mode=None)
    

  def test_mask_neighbors_3d(allnbrs=True):

    #randexp = random_exponential(shape=(2,2,30,80), a0=1)
    randexp = random_exponential(shape=(2,30,80), a0=1)

    fig  = gr.figure(figsize=(16,7), title='Random > 2-d mask')
    axim1 = gr.add_axes(fig, axwin=(0.05,  0.05, 0.40, 0.91))
    axcb1 = gr.add_axes(fig, axwin=(0.452, 0.05, 0.01, 0.91))

    axim2 = gr.add_axes(fig, axwin=(0.55,  0.05, 0.40, 0.91))
    axcb2 = gr.add_axes(fig, axwin=(0.952, 0.05, 0.01, 0.91))

    mask = np.select((randexp>6,), (0,), default=1)
    mask_nbrs = mask_neighbors(mask, allnbrs)

    img1 = reshape_to_2d(mask)
    img2 = reshape_to_2d(mask_nbrs)
    
    imsh1, cbar1 = gr.imshow_cbar(fig, axim1, axcb1, img1, amin=0, amax=10, orientation='vertical', cmap='jet')
    imsh2, cbar2 = gr.imshow_cbar(fig, axim2, axcb2, img2, amin=0, amax=10, orientation='vertical', cmap='jet')
    gr.show(mode=None)
    

  def test_mask_edges_2d(mrows=1, mcols=1):

    fig  = gr.figure(figsize=(8,7), title='Mask edges 2-d')
    axim1 = gr.add_axes(fig, axwin=(0.05,  0.05, 0.87, 0.91))
    axcb1 = gr.add_axes(fig, axwin=(0.922, 0.05, 0.01, 0.91))

    mask = np.ones((20,30))
    mask_out = mask_edges(mask, mrows, mcols)

    img1 = mask_out
    imsh1, cbar1 = gr.imshow_cbar(fig, axim1, axcb1, img1, amin=0, amax=10, orientation='vertical', cmap='jet')
    gr.show(mode=None)
    

  def test_mask_edges_3d(mrows=1, mcols=1):

    fig  = gr.figure(figsize=(8,7), title='Mask edges 2-d')
    axim1 = gr.add_axes(fig, axwin=(0.05,  0.05, 0.87, 0.91))
    axcb1 = gr.add_axes(fig, axwin=(0.922, 0.05, 0.01, 0.91))

    #mask = np.ones((2,2,20,30))
    mask = np.ones((2,20,30))
    mask_out = mask_edges(mask, mrows, mcols)

    img1 = reshape_to_2d(mask_out)
    imsh1, cbar1 = gr.imshow_cbar(fig, axim1, axcb1, img1, amin=0, amax=10, orientation='vertical', cmap='jet')
    gr.show(mode=None)
    

  def do_test():

    from psana.pyalgos.generic.NDArrGenerators import random_exponential; global random_exponential 
    import psana.pyalgos.generic.Graphics as gr; global gr

    print(80*'_')
    tname = sys.argv[1] if len(sys.argv)>1 else '1'
    if   tname == '1': test_mask_neighbors_2d(allnbrs = False)
    elif tname == '2': test_mask_neighbors_2d(allnbrs = True)
    elif tname == '3': test_mask_neighbors_3d(allnbrs = False)
    elif tname == '4': test_mask_neighbors_3d(allnbrs = True)
    elif tname == '5': test_mask_edges_2d(mrows=5, mcols=1)
    elif tname == '6': test_mask_edges_2d(mrows=0, mcols=5)
    elif tname == '7': test_mask_edges_3d(mrows=1, mcols=2)
    elif tname == '8': test_mask_edges_3d(mrows=5, mcols=0)
    elif tname =='12': test_02()
    else: sys.exit ('Not recognized test name: "%s"    Try tests 1-8' % tname)

#----

if __name__ == "__main__":
    import sys; global sys
    logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s',\
                        datefmt='%m-%d-%Y %H:%M:%S',\
                        level=logging.DEBUG)
                        #filename='example.log', filemode='w'
    do_test()
    sys.exit('\nEnd of test')

#----
