####!/usr/bin/env python
#------------------------------
"""
:py:class:`PSUtils` contains collection of global utilities with a single call algorithms
=============================================================================================

Usage::

    # Import
    # ==============
    from psana.pyalgos.generic.PSUtils import subtract_bkgd, #, ...
    from psana.pyalgos.generic.NDArrGenerators import random_standard_array

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
    # python lcls2/psana/psana/pyalgos/generic/PSUtils.py 1

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
from time import time, strptime, strftime, localtime, mktime
#from psana.pyalgos.generic.PSNameManager import nm

#------------------------------

def list_of_experiments(direxp=None) : # e.g. '/reg/d/psdm/XPP'
    """ Returns list of experiments in experimental directiory defined through configuration parameters.

    Parameters
    direxp : str - directory of experiments for particular instrument  # e.g. '/reg/d/psdm/XPP'
    """
    #ptrn = cp.instr_name.value().lower() if pattern is None else pattern # e.g. 'xpp'
    #dir  = nm.dir_exp() if direxp is None else direxp
    dir  = direxp
    ptrn = dir.rstrip('/').rsplit('/',1)[1].lower()    # e.g. 'xpp'
    #print('dir: %s  ptrn: %s' % (dir, ptrn))
    ldir = sorted(os.listdir(dir))
    #print('XXX list_of_experiments:', ldir)
    return [e for e in ldir if e[:3] == ptrn]

#------------------------------

def list_of_files_in_dir(dirname) :
    return os.listdir(dirname)

#------------------------------

def list_of_files_in_dir_for_ext(dir, ext='.xtc') :
    """Returns the list of files in the directory for specified extension or None if directory is None."""
    if dir is None : return []
    if not os.path.exists(dir) : return [] 
    return sorted([f for f in os.listdir(dir) if os.path.splitext(f)[1] == ext])
    
#------------------------------

def list_of_pathes_in_dir_for_ext(dir, ext='.xtc') :
    """Returns the list of pathes in the directory for specified extension or None if directory is None."""
    return [os.path.join(dir,f) for f in list_of_files_in_dir_for_ext(dir, ext)]
    
#------------------------------

def list_of_files_in_dir_for_pattern(dir, pattern='-r0022') :
    """Returns the list of files in the directory for specified file name pattern or [] - empty list."""
    if dir is None : return []
    if not os.path.exists(dir) : return []
    return sorted([os.path.join(dir,f) for f in os.listdir(dir) if pattern in f])

#------------------------------

def list_of_int_from_list_of_str(list_in) :
    """Converts  ['0001', '0202', '0203', '0204',...] to [1, 202, 203, 204,...]
    """
    return [int(item.lstrip('0')) for item in list_in]

#------------------------------

def list_of_str_from_list_of_int(list_in, fmt='%04d') :
    """Converts [1, 202, 203, 204,...] to ['0001', '0202', '0203', '0204',...]
    """
    return [fmt % item for item in list_in]

#------------------------------

def list_of_runs_in_xtc_dir(dirxtc) :  # e.g. '/reg/d/psdm/XPP/xpptut13/xtc'
    #dir = nm.dir_xtc() if dirxtc is None else dirxtc
    dir = dirxtc
    xtcfiles = list_of_files_in_dir_for_ext(dir, ext='.xtc')
    runs = [f.split('-')[1].lstrip('r') for f in xtcfiles]
    return set(runs)

#------------------------------

def src_type_alias_from_cfg_key(key) :
    """Returns striped object 'EventKey(type=None, src='DetInfo(CxiDs2.0:Cspad.0)', alias='DsdCsPad')'"""
    return k.src(), k.type(), k.alias()

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

"""Aliases"""

table8x8_from_cspad_ndarr = table_from_cspad_ndarr
cspad_ndarr_from_table8x8 = cspad_ndarr_from_table
# See tests in Detector/examples/ex_ndarray_from_image.py

#------------------------------
#----------- TEST -------------
#------------------------------

if __name__ == "__main__" :

  def test_convertCheetahEventName() :
    eventName = 'LCLS_2015_Feb22_r0169_022047_197f7'
    runnum, tstamp, tsec, fid = convertCheetahEventName(eventName, fmtts='%Y-%m-%dT%H:%M:%S')
    print('Method convertCheetahEventName converts Cheetah event name %s' % eventName,\
          '\nto runnum: %d  tstamp: %s  tsec: %d  fid: %s' % (runnum, tstamp, tsec, fid))

#------------------------------

  def test_list_of_files_in_dir() :
    print('%s:' % sys._getframe().f_code.co_name)
    lfiles = list_of_files_in_dir('/reg/d/psdm/sxr/sxrtut13/xtc/')
    for fname in lfiles : print(fname)

#------------------------------

  def test_list_of_files_in_dir_for_pattern() :
    print('%s:' % sys._getframe().f_code.co_name)
    lfiles = list_of_files_in_dir_for_pattern('/reg/d/psdm/cxi/cxitut13/xtc/', pattern='-r0011')
    for fname in lfiles : print(fname)

#------------------------------

  def test_list_of_files_in_dir_for_ext() :
    print('%s:' % sys._getframe().f_code.co_name)
    lfiles = list_of_files_in_dir_for_ext('/reg/d/psdm/sxr/sxrtut13/xtc/', ext='.xtc')
    for fname in lfiles : print(fname)

#------------------------------

  def test_list_of_str_from_list_of_int() :
    print('%s:' % sys._getframe().f_code.co_name)
    print(list_of_str_from_list_of_int([1, 202, 203, 204], fmt='%04d'))

#------------------------------

  def test_list_of_int_from_list_of_str() :
    print('%s:' % sys._getframe().f_code.co_name)
    print(list_of_int_from_list_of_str(['0001', '0202', '0203', '0204']))

#------------------------------

  def test_list_of_experiments(tname) :
    print('%s:' % sys._getframe().f_code.co_name)

    lexps = list_of_experiments('/reg/d/psdm/XPP')
    s = 'list_of_experiments():\n'
    for i,e in enumerate(lexps) :
        s += '%9s '%e
        if not (i+1)%10 : s += '\n'
    print(s)

#------------------------------

  def test_list_of_runs_in_xtc_dir() :
    print('%s:' % sys._getframe().f_code.co_name)
    print(list_of_runs_in_xtc_dir('/reg/d/psdm/XPP/xpptut13/xtc'))

#------------------------------
#------------------------------
#------------------------------

  def usage() : return 'Use command: python %s <test-number>, where <test-number> = 1,2,...,8,...' % sys.argv[0]

#------------------------------
  def test_all(tname) :
    print('\n%s\n' % usage())
    if len(sys.argv) != 2 : test_01()
    elif tname == '1': test_convertCheetahEventName()
    elif tname == '2': test_list_of_experiments(tname)
    elif tname == '3': test_list_of_str_from_list_of_int() 
    elif tname == '4': test_list_of_int_from_list_of_str()
    elif tname == '5': test_list_of_files_in_dir()
    elif tname == '6': test_list_of_files_in_dir_for_ext()
    elif tname == '7': test_list_of_files_in_dir_for_pattern()
    elif tname == '8': test_list_of_runs_in_xtc_dir()
    else : sys.exit('Test number parameter is not recognized.\n%s' % usage())

#------------------------------

if __name__ == "__main__" :
    import sys; global sys
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    print(50*'_', '\nTest %s' % tname)
    test_all(tname)
    sys.exit('End of Test %s' % tname)

#------------------------------
#------------------------------
#------------------------------
#------------------------------

