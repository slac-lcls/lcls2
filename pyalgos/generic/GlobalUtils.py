#!/usr/bin/env python
##-----------------------------
"""
:py:class:`GlobalUtils` contains collection of global utilities with a single call algorithms
=============================================================================================

Usage::

    # Import
    # ==============
    from pyimgalgos.GlobalUtils import subtract_bkgd, mask_from_windows #, ...
    from pyimgalgos.NDArrGenerators import random_standard_array

    # Background subtraction
    # ======================
    # Example for cspad, assuming all nda_*.shape = (32,185,388)
    winds_bkgd = [ (s, 10, 100, 270, 370) for s in (4,12,20,28)] # use part of segments 4,12,20, and 28 to subtract bkgd
    nda = subtract_bkgd(nda_data, nda_bkgd, mask=nda_mask, winds=winds_bkgd, pbits=0)

    # Operations with numpy array shape
    # =================================
    shape = (32,185,388)
    size = size_from_shape(shape) # returns 32*185*388   
    shape_2d = shape_as_2d(shape) # returns (32*185,388)
    arr_2d = reshape_to_2d(nda)   # returns re-shaped ndarray

    shape = (4,8,185,388)
    shape_3d = shape_as_3d(shape) # returns (32,185,388)
    arr_3d = reshape_to_3d(nda)   # returns re-shaped ndarray

    # Make mask n-d numpy array using shape and windows
    # =================================================
    shape = (2,185,388)
    w = 1
    winds = [(s, w, 185-w, w, 388-w) for s in (0,1)]
    mask = mask_from_windows(shape, winds)

    # Make mask as 2,3-d numpy array for a few(width) rows/cols of pixels 
    # ===================================================================
    mask2d = mask_2darr_edges(shape=(185,388), width=2)
    mask3d = mask_3darr_edges(shape=(32,185,388), width=5)

    # Make mask of local maximal intensity pixels in x-y (ignoring diagonals)
    # ===================================================================
    # works for 2-d and 3-d arrays only - reshape if needed.

    data = random_standard_array(shape=(32,185,388), mu=0, sigma=10)

    mask_xy_max = locxymax(data, order=1, mode='clip')

    # Get string with time stamp, ex: 2016-01-26T10:40:53
    # ===================================================================
    ts = str_tstamp(fmt='%Y-%m-%dT%H:%M:%S', time_sec=None)

    # Converters for Cheetah
    # ======================
    runnum, tstamp, tsec, fid = convertCheetahEventName('LCLS_2015_Feb22_r0169_022047_197f7', fmtts='%Y-%m-%dT%H:%M:%S')

    table8x8 = table_from_cspad_ndarr(nda_cspad) 
    nda_cspad = cspad_ndarr_from_table(table8x8)

    nda_32x185x388 = cspad_psana_from_cctbx(nda_64x185x194)
    nda_64x185x194 = cspad_cctbx_from_psana(nda_32x185x388)
    cross_check_cspad_psana_cctbx(nda_32x185x388, nda_64x185x194)

    # Safe math
    # =========
    res = divide_protected(num, den, vsub_zero=0)

    # Single line printed for np.array
    # ================================
    print_ndarr(nda, name='', first=0, last=5)

    # Save image in file
    # ==================
    image = random_standard()
    save_image_tiff(image, fname='image.tiff', verb=True) # 16-bit tiff
    save_image_file(image, fname='image.png', verb=True) # gif, pdf, eps, png, jpg, jpeg, tiff (8-bit only)

    # Create directory
    # ==================
    create_directory('work-dir')

    # Test
    # ======================
    # is implemented for test numbers from 1 to 9. Command example
    # python pyimgalgos/src/GlobalUtils.py 1

See :py:class:`GlobalUtils`

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Created by Mikhail Dubrovin
"""

##-----------------------------

import os
import sys
import math
import numpy as np
from scipy.signal import argrelmax
from time import time, strptime, strftime, localtime, mktime

#------------------------------

class Storage :
    """Store for shared parameters."""
    def __init__(self) :
        self.Image     = None
        self.scim      = None
        self.time      = None
        self.localtime = None
        self.strftime  = None

##-----------------------------

sp = Storage()

##-----------------------------

def size_from_shape(shape) :
    """Returns size from the shape sequence 
    """
    size=1
    for d in shape : size*=d
    return size

##-----------------------------

def shape_as_2d(sh) :
    """Returns 2-d shape for n-d shape if n>2, otherwise returns unchanged shape.
    """
    if len(sh)<3 : return sh
    return (size_from_shape(sh)/sh[-1], sh[-1])

##-----------------------------

def shape_as_3d(sh) :
    """Returns 3-d shape for n-d shape if n>3, otherwise returns unchanged shape.
    """
    if len(sh)<4 : return sh
    return (size_from_shape(sh)/sh[-1]/sh[-2], sh[-2], sh[-1])

##-----------------------------

def reshape_to_2d(arr) :
    """Returns n-d re-shaped to 2-d
    """
    arr.shape = shape_as_2d(arr.shape)
    return arr

##-----------------------------

def reshape_to_3d(arr) :
    """Returns n-d re-shaped to 3-d
    """
    arr.shape = shape_as_3d(arr.shape)
    return arr

##-----------------------------

def mask_2darr_edges(shape=(185,388), width=2) :
    """Returns mask with masked width rows/colunms of edge pixels for 2-d numpy array.
    """
    s, w = shape, width
    mask = np.zeros(shape, dtype=np.uint16)
    mask[w:-w,w:-w] = np.ones((s[0]-2*w,s[1]-2*w), dtype=np.uint16)
    return mask

##-----------------------------

def mask_3darr_edges(shape=(32,185,388), width=2) :
    """Returns mask with masked width rows/colunms of edge pixels for 3-d numpy array.
    """
    s, w = shape, width
    mask = np.zeros(shape, dtype=np.uint16)
    mask[:,w:-w,w:-w] = np.ones((s[0],s[1]-2*w,s[2]-2*w), dtype=np.uint16)
    return mask

##-----------------------------

def mask_from_windows(ashape=(32,185,388), winds=None) :
    """Makes mask as 2-d or 3-d numpy array defined by the shape with ones in windows.
       N-d shape for N>3 is converted to 3-d.
       @param shape - shape of the output numpy array with mask.
       @param winds - list of windows, each window is a sequence of 5 parameters (segment, rowmin, rowmax, colmin, colmax)     
    """
    ndim = len(ashape)
        
    if ndim<2 :
        print 'ERROR in mask_from_windows(...):',\
              ' Wrong number of dimensions %d in the shape=%s parameter. Allowed ndim>1.' % (ndim, str(shape))
        return None

    shape = ashape if ndim<4 else shape_as_3d(ashape)

    seg1 = np.ones((shape[-2], shape[-1]), dtype=np.uint16) # shaped as last two dimensions
    mask = np.zeros(shape, dtype=np.uint16)

    if ndim == 2 :
        for seg,rmin,rmax,cmin,cmax in winds : mask[rmin:rmax,cmin:cmax] =  seg1[rmin:rmax,cmin:cmax]
        return mask

    elif ndim == 3 :
        for seg,rmin,rmax,cmin,cmax in winds : mask[seg,rmin:rmax,cmin:cmax] =  seg1[rmin:rmax,cmin:cmax]
        return mask

##-----------------------------

def list_of_windarr(nda, winds=None) :
    """Converts 2-d or 3-d numpy array in the list of 2-d numpy arrays for windows
       @param nda - input 2-d or 3-d numpy array
    """
    ndim = len(nda.shape)
    #print 'len(nda.shape): ', ndim

    if ndim == 2 :
        return [nda] if winds is None else \
               [nda[rmin:rmax, cmin:cmax] for (s, rmin, rmax, cmin, cmax) in winds]

    elif ndim == 3 :
        return [nda[s,:,:] for s in range(ndim.shape[0])] if winds is None else \
               [nda[s, rmin:rmax, cmin:cmax] for (s, rmin, rmax, cmin, cmax) in winds]

    else :
        print 'ERROR in list_of_windarr (with winds): Unexpected number of n-d array dimensions: ndim = %d' % ndim
        return []

##-----------------------------

def mean_of_listwarr(lst_warr) :
    """Evaluates the mean value of the list of 2-d arrays.
       @lst_warr - list of numpy arrays to evaluate per pixel mean intensity value.
    """
    s1, sa = 0., 0. 
    for warr in lst_warr :
        sa += np.sum(warr, dtype=np.float64)
        s1 += warr.size
    return sa/s1 if s1 > 0 else 1

##-----------------------------

def subtract_bkgd(data, bkgd, mask=None, winds=None, pbits=0) :
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
    if pbits : print 'subtract_bkgd, fraction = %10.6f' % frac

    return data - bkgd*frac



##-----------------------------
# See: http://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.argrelmax.html#scipy.signal.argrelmax

def locxymax(nda, order=1, mode='clip') :
    """For 2-d or 3-d numpy array finds mask of local maxima in x and y axes (diagonals are ignored) 
       using scipy.signal.argrelmax and return their product.

       @param nda - input ndarray
       @param order - range to search for local maxima along each dimension
       @param mode - parameter of scipy.signal.argrelmax of how to treat the boarder
    """
    shape = nda.shape
    size  = nda.size
    ndim = len(shape)

    if ndim< 2 or ndim>3 :
        msg = 'ERROR: locxymax nda shape %s should be 2-d or 3-d' % (shape)
        sys.exit(msg)

    ext_cols = argrelmax(nda, -1, order, mode)
    ext_rows = argrelmax(nda, -2, order, mode)
    
    indc = np.array(ext_cols, dtype=np.uint16)
    indr = np.array(ext_rows, dtype=np.uint16)

    msk_ext_cols = np.zeros(shape, dtype=np.uint16)
    msk_ext_rows = np.zeros(shape, dtype=np.uint16)

    if ndim == 2 :
        icr = indc[0,:] 
        icc = indc[1,:]
        irr = indr[0,:]
        irc = indr[1,:]

        msk_ext_cols[icr,icc] = 1
        msk_ext_rows[irr,irc] = 1

    elif ndim == 3 :
        ics = indc[0,:] 
        icr = indc[1,:] 
        icc = indc[2,:]
        irs = indr[0,:]
        irr = indr[1,:]
        irc = indr[2,:]

        msk_ext_cols[ics,icr,icc] = 1
        msk_ext_rows[irs,irr,irc] = 1

    #print 'nda.size:',   nda.size
    #print 'indc.shape:', indc.shape
    #print 'indr.shape:', indr.shape
    
    return msk_ext_rows * msk_ext_cols

##-----------------------------

def str_tstamp(fmt='%Y-%m-%dT%H:%M:%S', time_sec=None) :
    """Returns string timestamp for specified format and time in sec or current time by default
    """
    if sp.localtime is None : from time import localtime, strftime; sp.localtime, sp.strftime = localtime, strftime

    return sp.strftime(fmt, sp.localtime(time_sec))

##-----------------------------

def save_image_tiff(image, fname='image.tiff', verb=False) :
    """Saves image in 16-bit tiff file
    """
    if sp.Image is None : import Image; sp.Image=Image

    if verb : print 'Save image in file %s' % fname
    img = sp.Image.fromarray(image.astype(np.int16))
    img.save(fname)

##-----------------------------

def save_image_file(image, fname='image.png', verb=False) :
    """Saves files with type by extension gif, pdf, eps, png, jpg, jpeg, tiff (8-bit only),
       or txt for any other type
    """
    if sp.scim is None : import scipy.misc as scim; sp.scim=scim 

    fields = os.path.splitext(fname)
    if len(fields)>1 and fields[1] in ['.gif', '.pdf', '.eps', '.png', '.jpg', '.jpeg', '.tiff'] : 
        if verb : print 'Save image in file %s' % fname
        sp.scim.imsave(fname, image) 
    else :
        if verb : print 'Non-supported file extension. Save image in text file %s' % fname
        np.savetxt(fname, image, fmt='%8.1f', delimiter=' ', newline='\n')
        #raise IOError('Unknown file type in extension %s' % fname)

#------------------------------

def convertCheetahEventName(evname, fmtts='%Y-%m-%dT%H:%M:%S') :
    """Converts Cheetah event name like 'LCLS_2015_Feb22_r0169_022047_197f7'
       and returns runnum, tstamp, tsec, fid = 169, '2015-02-22T02:20:47', <tsec>, 197f7
    """
    fields = evname.split('_')
    if len(fields) != 6 :
        raise ValueError('Cheetah event name has unexpected structure (ex: '\
                         'LCLS_2015_Feb22_r0169_022047_197f7) \n number of fields is not 6: %s' % evname)

    s_factory, s_year, s_mon_day, s_run, s_time, s_fid = fields
    
    #fid    = int(s_fid, 16)
    runnum = int(s_run.strip('r').lstrip('0'))
    struct = strptime('%s-%s-%s' % (s_year, s_mon_day, s_time), '%Y-%b%d-%H%M%S')
    tsec   = mktime(struct)
    tstamp = strftime(fmtts, localtime(tsec))
    return runnum, tstamp, tsec, s_fid

#------------------------------

def src_from_rc8x8(row, col) :
    """Converts Cheetah 8x8 ASICs table row and column to seg, row, col coordinates
    """
    qsegs, rows, cols = (8, 185, 388) 
    quad = math.floor(col/cols) # [0,3]
    qseg = math.floor(row/rows) # [0,7]
    s = qsegs*quad + qseg
    c = col%cols if isinstance(col, int) else math.fmod(col, cols)
    r = row%rows if isinstance(row, int) else math.fmod(row, rows)
    return s, r, c

#------------------------------

def print_ndarr(nda, name='', first=0, last=5) :
    if nda is None : print '%s: %s' % (name, nda)
    elif isinstance(nda, tuple) : print_ndarr(np.array(nda), 'ndarray from tuple: %s' % name)
    elif isinstance(nda, list)  : print_ndarr(np.array(nda), 'ndarray from list: %s' % name)
    elif not isinstance(nda, np.ndarray) :
                     print '%s: %s' % (name, type(nda))
    else           : print '%s:  shape:%s  size:%d  dtype:%s %s...' % \
         (name, str(nda.shape), nda.size, nda.dtype, nda.flatten()[first:last])

#------------------------------

def table_from_cspad_ndarr(nda_cspad) :
    """returns table of 2x1s shaped as (8*185, 4*388) in style of Cheetah
       generated from cspad array with size=(32*185*388) ordered as in data, shape does not matter. 
    """
    shape, size = (4, 8*185, 388), 4*8*185*388
    if nda_cspad.size != size :
        raise ValueError('Input array size: %d is not consistent with cspad size: %d' % (nda_cspad.size, size))
    shape_in = nda_cspad.shape # preserve original shape
    nda_cspad.shape = shape    # reshape to what we need
    nda_out = np.hstack([nda_cspad[q,:] for q in range(shape[0])])
    nda_cspad.shape = shape_in # restore original shape
    return nda_out

#------------------------------

def cspad_ndarr_from_table(table8x8) :
    """returns cspad array with shape=(32,185,388)
       generated from table of 2x1s shaped as (8*185, 4*388) in style of Cheetah
    """
    quads, segs, rows, cols = (4,8,185,388)
    size = quads * segs * rows * cols
    shape8x8 = (segs*rows, quads*cols)

    if table8x8.size != size :
        raise ValueError('Input array size: %d is not consistent with cspad size: %d' % (table8x8.size, size))

    if table8x8.shape != shape8x8:
        raise ValueError('Input array shape: %s is not consistent with cspad 8x8 table shape: %s' % (table8x8.shape, shape8x8))

    nda_out = np.array([table8x8[:,q*cols:(q+1)*cols] for q in range(quads)]) # shape:(4, 1480, 388)
    nda_out.shape = (quads*segs, rows, cols)
    return nda_out

#------------------------------

def cspad_psana_from_cctbx(nda_in) :
    """returns cspad array (32, 185, 388) from cctbx array of ASICs (64, 185, 194)
    """
    asics, rows, colsh = shape_in = (64,185,194)
    size = asics * rows * colsh
    segs, cols = asics/2, colsh*2
    
    if nda_in.size != size :
        raise ValueError('Input array size: %d is not consistent with cspad size: %d' % (nda_in.size, size))

    if nda_in.shape != shape_in:
        raise ValueError('Input array shape: %s is not consistent with cspad 8x8 table shape: %s' % (nda_in.shape, shape_in))

    nda_out = np.empty((segs, rows, cols), dtype=nda_in.dtype)
    
    for s in range(segs) :
        a=s*2 # ASIC[0] in segment
        nda_out[s,:,0:colsh]    = nda_in[a,:,:]
        nda_out[s,:,colsh:cols] = nda_in[a+1,:,:]

    return nda_out

#------------------------------

def cspad_cctbx_from_psana(nda_in) :
    """returns cctbx array of ASICs (64, 185, 194) from cspad array (32, 185, 388)
    """
    segs, rows, cols = shape_in = (32,185,388)
    size = segs * rows * cols
    colsh = cols/2

    if nda_in.size != size :
        raise ValueError('Input array size: %d is not consistent with cspad size: %d' % (nda_in.size, size))

    if nda_in.shape != shape_in:
        raise ValueError('Input array shape: %s is not consistent with cspad shape: %s' % (nda_in.shape, shape_in))

    nda_out = np.empty((segs*2, rows, cols/2), dtype=nda_in.dtype)
    for s in range(segs) :
        a=s*2 # ASIC[0] in segment
        nda_out[a,:,:]   = nda_in[s,:,0:colsh]
        nda_out[a+1,:,:] = nda_in[s,:,colsh:cols]
    return nda_out

#------------------------------

def cross_check_cspad_psana_cctbx(nda, arr) :
    """Apply two-way conversions between psana and cctbx cspad arrays and compare.
    """
    from time import time
    t0_sec = time()
    nda_c = cspad_psana_from_cctbx(arr)
    dt1 = time() - t0_sec
    t0_sec = time()
    arr_c = cspad_cctbx_from_psana(nda)
    dt2 = time() - t0_sec
    print 'psana ndarray is equal to converted from cctbx: %s, time = %.6f sec' % (np.array_equal(nda, nda_c), dt1)
    print 'cctbx ndarray is equal to converted from psana: %s, time = %.6f sec' % (np.array_equal(arr, arr_c), dt2)

#------------------------------
#------------------------------
#------------------------------

def divide_protected(num, den, vsub_zero=0) :
    """Returns result of devision of numpy arrays num/den with substitution of value vsub_zero for zero den elements.
    """
    pro_num = np.select((den!=0,), (num,), default=vsub_zero)
    pro_den = np.select((den!=0,), (den,), default=1)
    return pro_num / pro_den

#------------------------------

def create_directory(dir) :
    if dir=='' or (dir is None) : return
    if os.path.exists(dir) :
        print 'Directory exists: %s' % dir
    else :
        os.makedirs(dir)
        print 'Directory created: %s' % dir

#------------------------------

def print_command_line_parameters(parser) :
    """Prints input arguments and optional parameters"""
    (popts, pargs) = parser.parse_args()
    args = pargs                             # list of positional arguments
    opts = vars(popts)                       # dict of options
    defs = vars(parser.get_default_values()) # dict of default options

    print 'Command:\n ', ' '.join(sys.argv)+\
          '\nArgument list: %s\nOptional parameters:\n' % str(args)+\
          '  <key>      <value>              <default>'
    for k,v in opts.items() :
        print '  %s %s %s' % (k.ljust(10), str(v).ljust(20), str(defs[k]).ljust(20))

#------------------------------

def sigma_h(pk) :
    if pk.seg in (0,1,4,5, 10,11,14,15, 16,17,20,21, 26,27,30,31) : return pk.rsigma 
    return pk.csigma

def sigma_v(pk) :
    if pk.seg in (0,1,4,5, 10,11,14,15, 16,17,20,21, 26,27,30,31) : return pk.csigma 
    return pk.rsigma

def sigma_hov(pk) :
    sh = sigma_h(pk)
    sv = sigma_v(pk)
    return sh/sv if sv>0 else 10

#------------------------------
#------------------------------
"""Aliases"""

table8x8_from_cspad_ndarr = table_from_cspad_ndarr
cspad_ndarr_from_table8x8 = cspad_ndarr_from_table
# See tests in Detector/examples/ex_ndarray_from_image.py

##-----------------------------
##-----------------------------
##---------- TEST -------------
##-----------------------------
##-----------------------------

def test_01() :
    from pyimgalgos.NDArrGenerators import random_standard

    print '%s\n%s\n' % (80*'_','Test method subtract_bkgd(...):')
    shape1 = (32,185,388)
    winds = [(s, 10, 155, 20, 358) for s in (0,1)]
    data = random_standard(shape=shape1, mu=300, sigma=50)
    bkgd = random_standard(shape=shape1, mu=100, sigma=10)
    cdata = subtract_bkgd(data, bkgd, mask=None, winds=winds, pbits=0377)

##-----------------------------

def test_02() :
    print '%s\n%s\n' % (80*'_','Test method size_from_shape(shape):')
    shape = (2,3,4,5)
    print '  shape=%s,  size_from_shape(shape)=%d' % (shape, size_from_shape(shape))

##-----------------------------

def test_03() :
    print '%s\n%s\n' % (80*'_','Test method shape_as_2d(shape):')
    shape = (2,3,4,5)
    print '  shape=%s,  shape_as_2d(shape)=%s' % (shape, shape_as_2d(shape))

##-----------------------------

def test_04() :
    print '%s\n%s\n' % (80*'_','Test method shape_as_3d(shape):')
    shape = (2,3,4,5)
    print '  shape=%s,  shape_as_3d(shape)=%s' % (shape, shape_as_3d(shape))

##-----------------------------

def test_05() :
    print '%s\n%s\n' % (80*'_','Test method mask_from_windows(shape, winds):')
    shape = (2,185,388)
    w = 1
    winds = [(s, w, 185-w, w, 388-w) for s in (0,1)]
    mask = mask_from_windows(shape, winds)
    print '  shape=%s \nwinds:\n%s, \nmask_from_windows(shape, winds):\n%s' % (shape, winds, mask_from_windows(shape, winds))

##-----------------------------

def test_06() :
    print '%s\n%s\n' % (80*'_','Test method mask_3darr_edges(shape, width):')
    shape = (32,185,388)
    width = 1
    print '  shape=%s, masking width=%d, mask_3darr_edges(shape, width):\n%s' % (shape, width, mask_3darr_edges(shape, width))

##-----------------------------

def test_07() :
    print '%s\n%s\n' % (80*'_','Test method mask_2darr_edges(shape, width):')
    shape = (185,388)
    width = 2
    print '  shape=%s, masking width=%d, mask_2darr_edges(shape, width):\n%s' % (shape, width, mask_2darr_edges(shape, width))

##-----------------------------

def test_08() :
    from time import time
    import pyimgalgos.GlobalGraphics as gg
    from pyimgalgos.NDArrGenerators import random_standard

    print '%s\n%s\n' % (80*'_','Test method locxymax(nda, order, mode):')
    #data = random_standard(shape=(32,185,388), mu=0, sigma=10)
    data = random_standard(shape=(2,185,388), mu=0, sigma=10)
    t0_sec = time()
    mask = locxymax(data, order=1, mode='clip')
    print 'Consumed t = %10.6f sec' % (time()-t0_sec)

    if True :
      img = data if len(data.shape)==2 else reshape_to_2d(data)
      msk = mask if len(mask.shape)==2 else reshape_to_2d(mask)

      ave, rms = img.mean(), img.std()
      amin, amax = ave-2*rms, ave+2*rms
      gg.plotImageLarge(img, amp_range=(amin, amax), title='random')
      gg.plotImageLarge(msk, amp_range=(0, 1), title='mask loc max')
      gg.show()

##-----------------------------

def test_09() :
    print str_tstamp()

##-----------------------------

def test_10() :
    from pyimgalgos.NDArrGenerators import random_standard

    image = random_standard()
    verbosity=True
    save_image_tiff(image, fname='image.tiff', verb=verbosity)
    save_image_file(image, fname='image.png',  verb=verbosity)
    save_image_file(image, fname='image.xyz',  verb=verbosity)

##-----------------------------

def test_11() :
    eventName = 'LCLS_2015_Feb22_r0169_022047_197f7'
    runnum, tstamp, tsec, fid = convertCheetahEventName(eventName, fmtts='%Y-%m-%dT%H:%M:%S')
    print 'Method convertCheetahEventName converts Cheetah event name %s' % eventName,\
          '\nto runnum: %d  tstamp: %s  tsec: %d  fid: %s' % (runnum, tstamp, tsec, fid)

##-----------------------------
##-----------------------------
##-----------------------------
##-----------------------------
##-----------------------------
##-----------------------------

def usage() : return 'Use command: python %s <test-number>, where <test-number> = 1,2,...,7,...' % sys.argv[0]

##-----------------------------

def main() :    
    print '\n%s\n' % usage()
    if len(sys.argv) != 2 : test_01()
    elif sys.argv[1]=='1' : test_01()
    elif sys.argv[1]=='2' : test_02()
    elif sys.argv[1]=='3' : test_03()
    elif sys.argv[1]=='4' : test_04()
    elif sys.argv[1]=='5' : test_05()
    elif sys.argv[1]=='6' : test_06()
    elif sys.argv[1]=='7' : test_07()
    elif sys.argv[1]=='8' : test_08()
    elif sys.argv[1]=='9' : test_09()
    elif sys.argv[1]=='10': test_10()
    elif sys.argv[1]=='11': test_11()
    else                  : sys.exit ('Test number parameter is not recognized.\n%s' % usage())

##-----------------------------

if __name__ == "__main__" :
    main()
    sys.exit('\nEnd of test')

##-----------------------------
##-----------------------------
##-----------------------------
##-----------------------------

