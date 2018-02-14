####!/usr/bin/env python
#------------------------------
"""
:py:class:`PSUtils` contains collection of global utilities with a single call algorithms
=============================================================================================

Usage::

    # Import
    # ==============
    from pyimgalgos.PSUtils import subtract_bkgd, #, ...
    from pyimgalgos.NDArrGenerators import random_standard_array

    # Converters for Cheetah
    # ======================
    runnum, tstamp, tsec, fid = convertCheetahEventName('LCLS_2015_Feb22_r0169_022047_197f7', fmtts='%Y-%m-%dT%H:%M:%S')

    table8x8 = table_from_cspad_ndarr(nda_cspad) 
    nda_cspad = cspad_ndarr_from_table(table8x8)

    nda_32x185x388 = cspad_psana_from_cctbx(nda_64x185x194)
    nda_64x185x194 = cspad_cctbx_from_psana(nda_32x185x388)
    cross_check_cspad_psana_cctbx(nda_32x185x388, nda_64x185x194)

    # Test
    # ======================
    # is implemented for test numbers from 1 to 9. Command example
    # python pyimgalgos/src/PSUtils.py 1

See :py:class:`PSUtils`

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created by Mikhail Dubrovin
Adopted for LCLS2 on 2018-02-02
"""

#------------------------------

import os
import sys
import math
import numpy as np
from scipy.signal import argrelmax
from time import time, strptime, strftime, localtime, mktime


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
    t0_sec = time()
    nda_c = cspad_psana_from_cctbx(arr)
    dt1 = time() - t0_sec
    t0_sec = time()
    arr_c = cspad_cctbx_from_psana(nda)
    dt2 = time() - t0_sec
    print('psana ndarray is equal to converted from cctbx: %s, time = %.6f sec' % (np.array_equal(nda, nda_c), dt1))
    print('cctbx ndarray is equal to converted from psana: %s, time = %.6f sec' % (np.array_equal(arr, arr_c), dt2))

#------------------------------
#------------------------------

def print_command_line_parameters(parser) :
    """Prints input arguments and optional parameters"""
    (popts, pargs) = parser.parse_args()
    args = pargs                             # list of positional arguments
    opts = vars(popts)                       # dict of options
    defs = vars(parser.get_default_values()) # dict of default options

    print('Command:\n ', ' '.join(sys.argv)+\
          '\nArgument list: %s\nOptional parameters:\n' % str(args)+\
          '  <key>      <value>              <default>')
    for k,v in opts.items() :
        print('  %s %s %s' % (k.ljust(10), str(v).ljust(20), str(defs[k]).ljust(20)))

#------------------------------

"""Aliases"""

table8x8_from_cspad_ndarr = table_from_cspad_ndarr
cspad_ndarr_from_table8x8 = cspad_ndarr_from_table
# See tests in Detector/examples/ex_ndarray_from_image.py

#------------------------------
#------------------------------
#----------- TEST -------------
#------------------------------
#------------------------------

def test_01() :
    from pyimgalgos.NDArrGenerators import random_standard

    print('%s\n%s\n' % (80*'_','Test method subtract_bkgd(...):'))
    shape1 = (32,185,388)
    winds = [(s, 10, 155, 20, 358) for s in (0,1)]
    data = random_standard(shape=shape1, mu=300, sigma=50)
    bkgd = random_standard(shape=shape1, mu=100, sigma=10)
    cdata = subtract_bkgd(data, bkgd, mask=None, winds=winds, pbits=0377)

#------------------------------

def test_08() :
    import pyalgos.generic.Graphics as gg
    from pyalgos.generic.NDArrGenerators import random_standard
    from pyalgos.generic.NDArrUtils import reshape_to_2d

    print('%s\n%s\n' % (80*'_','Test method locxymax(nda, order, mode):'))
    #data = random_standard(shape=(32,185,388), mu=0, sigma=10)
    data = random_standard(shape=(2,185,388), mu=0, sigma=10)
    t0_sec = time()
    mask = locxymax(data, order=1, mode='clip')
    print('Consumed t = %10.6f sec' % (time()-t0_sec))

    if True :
      img = data if len(data.shape)==2 else reshape_to_2d(data)
      msk = mask if len(mask.shape)==2 else reshape_to_2d(mask)

      ave, rms = img.mean(), img.std()
      amin, amax = ave-2*rms, ave+2*rms
      gg.plotImageLarge(img, amp_range=(amin, amax), title='random')
      gg.plotImageLarge(msk, amp_range=(0, 1), title='mask loc max')
      gg.show()


#------------------------------

def test_10() :
    from pyimgalgos.NDArrGenerators import random_standard

    image = random_standard()
    verbosity=True
    save_image_tiff(image, fname='image.tiff', verb=verbosity)
    save_image_file(image, fname='image.png',  verb=verbosity)
    save_image_file(image, fname='image.xyz',  verb=verbosity)

#------------------------------

def test_11() :
    eventName = 'LCLS_2015_Feb22_r0169_022047_197f7'
    runnum, tstamp, tsec, fid = convertCheetahEventName(eventName, fmtts='%Y-%m-%dT%H:%M:%S')
    print('Method convertCheetahEventName converts Cheetah event name %s' % eventName,\
          '\nto runnum: %d  tstamp: %s  tsec: %d  fid: %s' % (runnum, tstamp, tsec, fid))

#------------------------------
#------------------------------
#------------------------------
#------------------------------
#------------------------------
#------------------------------

def usage() : return 'Use command: python %s <test-number>, where <test-number> = 1,2,...,7,...' % sys.argv[0]

#------------------------------

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
    else                  : sys.exit('Test number parameter is not recognized.\n%s' % usage())

#------------------------------

if __name__ == "__main__" :
    main()
    sys.exit('End of test')

#------------------------------
#------------------------------
#------------------------------
#------------------------------

