####!/usr/bin/env python

"""
:py:class:`PSUtils` contains collection of global utilities with a single call algorithms
=============================================================================================

Usage::

    # Import
    # ==============
    from psana2.pyalgos.generic.PSUtils import subtract_bkgd, #, ...
    from psana2.pyalgos.generic.NDArrGenerators import random_standard_array

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
import logging
logger = logging.getLogger(__name__)

import os
import sys
import math
import numpy as np
from time import time, strptime, strftime, localtime, mktime
from psana2.pyalgos.generic.UtilsFS import *

INSTRUMENT_DIR = os.getenv('SIT_PSDM_DATA', '/sdf/data/lcls/ds/').rstrip('/')  # /sdf/data/lcls/ds/ or /cds/data/psdm/

def dir_exp(expname, dirinstr=INSTRUMENT_DIR):
    assert isinstance(expname, str)
    assert len(expname) in (8,9)
    return os.path.join(dirinstr, expname[:3], expname) # expname[:3].upper()

def dir_xtc(expname, dirinstr=INSTRUMENT_DIR):
    return os.path.join(dir_exp(expname, dirinstr), 'xtc')

def dir_calib(expname, dirinstr=INSTRUMENT_DIR):
    return os.path.join(dir_exp(expname, dirinstr), 'calib')

def list_of_experiments(direxp=None): # e.g. '/reg/d/psdm/XPP'
    """ Returns list of experiments in experimental directiory defined through configuration parameters.

    Parameters
    direxp: str - directory of experiments for particular instrument  # e.g. '/reg/d/psdm/XPP'
    """
    #ptrn = cp.instr_name.value().lower() if pattern is None else pattern # e.g. 'xpp'
    #dir  = nm.dir_exp() if direxp is None else direxp
    dir  = direxp
    ptrn = dir.rstrip('/').rsplit('/',1)[1].lower()    # e.g. 'xpp'
    #print('dir: %s  ptrn: %s' % (dir, ptrn))
    ldir = sorted(os.listdir(dir))
    #print('XXX list_of_experiments:', ldir)
    return [e for e in ldir if e[:3] == ptrn]

def list_of_instruments(dirname=INSTRUMENT_DIR):
    ldir = os.listdir(dirname)
    if len(ldir)>10: return tuple([d for d in ldir if d[:3].isupper()]) + ('asc','prj','mon','ued','txi')
    else:
        logger.info('can not access content of the directory %s\n return hardwired list of instruments' % dirname)
        return ['MEC', 'TMO', 'MOB', 'MFX', 'RIX', 'USR', 'SXR', 'AMO', 'XPP', 'CXI', 'DET', 'DIA', 'TST', 'XCS',\
                'asc', 'prj', 'mon', 'ued', 'txi']

def list_of_int_from_list_of_str(list_in):
    """Converts  ['0001', '0202', '0203', '0204',...] to [1, 202, 203, 204,...]
    """
    return [int(item.lstrip('0')) for item in list_in]

def list_of_str_from_list_of_int(list_in, fmt='%04d'):
    """Converts [1, 202, 203, 204,...] to ['0001', '0202', '0203', '0204',...]
    """
    return [fmt % item for item in list_in]

def list_of_runs_in_xtc_dir(dirxtc, ext='.xtc'):  # e.g. '/reg/d/psdm/XPP/xpptut15/xtc'
    #xtcfiles = list_of_files_in_dir_for_ext(dirxtc, ext)
    xtcfiles = list_of_files_in_dir_for_pattern(dirxtc, ext)
    runs = [f.split('-')[1].lstrip('r') for f in xtcfiles]
    return set(runs)

def src_type_alias_from_cfg_key(key):
    """Returns striped object 'EventKey(type=None, src='DetInfo(CxiDs2.0:Cspad.0)', alias='DsdCsPad')'"""
    return k.src(), k.type(), k.alias()

def convertCheetahEventName(evname, fmtts='%Y-%m-%dT%H:%M:%S'):
    """Converts Cheetah event name like 'LCLS_2015_Feb22_r0169_022047_197f7'
       and returns runnum, tstamp, tsec, fid = 169, '2015-02-22T02:20:47', <tsec>, 197f7
    """
    fields = evname.split('_')
    if len(fields) != 6:
        raise ValueError('Cheetah event name has unexpected structure (ex: '\
                         'LCLS_2015_Feb22_r0169_022047_197f7) \n number of fields is not 6: %s' % evname)

    s_factory, s_year, s_mon_day, s_run, s_time, s_fid = fields

    #fid    = int(s_fid, 16)
    runnum = int(s_run.strip('r').lstrip('0'))
    struct = strptime('%s-%s-%s' % (s_year, s_mon_day, s_time), '%Y-%b%d-%H%M%S')
    tsec   = mktime(struct)
    tstamp = strftime(fmtts, localtime(tsec))
    return runnum, tstamp, tsec, s_fid

def src_from_rc8x8(row, col):
    """Converts Cheetah 8x8 ASICs table row and column to seg, row, col coordinates."""
    qsegs, rows, cols = (8, 185, 388)
    quad = math.floor(col/cols) # [0,3]
    qseg = math.floor(row/rows) # [0,7]
    s = qsegs*quad + qseg
    c = col%cols if isinstance(col, int) else math.fmod(col, cols)
    r = row%rows if isinstance(row, int) else math.fmod(row, rows)
    return s, r, c

def table_from_cspad_ndarr(nda_cspad):
    """returns table of 2x1s shaped as (8*185, 4*388) in style of Cheetah
       generated from cspad array with size=(32*185*388) ordered as in data, shape does not matter.
    """
    shape, size = (4, 8*185, 388), 4*8*185*388
    if nda_cspad.size != size:
        raise ValueError('Input array size: %d is not consistent with cspad size: %d' % (nda_cspad.size, size))
    shape_in = nda_cspad.shape # preserve original shape
    nda_cspad.shape = shape    # reshape to what we need
    nda_out = np.hstack([nda_cspad[q,:] for q in range(shape[0])])
    nda_cspad.shape = shape_in # restore original shape
    return nda_out

def cspad_ndarr_from_table(table8x8):
    """returns cspad array with shape=(32,185,388)
       generated from table of 2x1s shaped as (8*185, 4*388) in style of Cheetah
    """
    quads, segs, rows, cols = (4,8,185,388)
    size = quads * segs * rows * cols
    shape8x8 = (segs*rows, quads*cols)

    if table8x8.size != size:
        raise ValueError('Input array size: %d is not consistent with cspad size: %d' % (table8x8.size, size))

    if table8x8.shape != shape8x8:
        raise ValueError('Input array shape: %s is not consistent with cspad 8x8 table shape: %s' % (table8x8.shape, shape8x8))

    nda_out = np.array([table8x8[:,q*cols:(q+1)*cols] for q in range(quads)]) # shape:(4, 1480, 388)
    nda_out.shape = (quads*segs, rows, cols)
    return nda_out


def cspad_psana_from_cctbx(nda_in):
    """returns cspad array (32, 185, 388) from cctbx array of ASICs (64, 185, 194)."""
    asics, rows, colsh = shape_in = (64,185,194)
    size = asics * rows * colsh
    segs, cols = asics/2, colsh*2

    if nda_in.size != size:
        raise ValueError('Input array size: %d is not consistent with cspad size: %d' % (nda_in.size, size))

    if nda_in.shape != shape_in:
        raise ValueError('Input array shape: %s is not consistent with cspad 8x8 table shape: %s' % (nda_in.shape, shape_in))

    nda_out = np.empty((segs, rows, cols), dtype=nda_in.dtype)

    for s in range(segs):
        a=s*2 # ASIC[0] in segment
        nda_out[s,:,0:colsh]    = nda_in[a,:,:]
        nda_out[s,:,colsh:cols] = nda_in[a+1,:,:]

    return nda_out

def cspad_cctbx_from_psana(nda_in):
    """returns cctbx array of ASICs (64, 185, 194) from cspad array (32, 185, 388)."""
    segs, rows, cols = shape_in = (32,185,388)
    size = segs * rows * cols
    colsh = cols/2

    if nda_in.size != size:
        raise ValueError('Input array size: %d is not consistent with cspad size: %d' % (nda_in.size, size))

    if nda_in.shape != shape_in:
        raise ValueError('Input array shape: %s is not consistent with cspad shape: %s' % (nda_in.shape, shape_in))

    nda_out = np.empty((segs*2, rows, cols/2), dtype=nda_in.dtype)
    for s in range(segs):
        a=s*2 # ASIC[0] in segment
        nda_out[a,:,:]   = nda_in[s,:,0:colsh]
        nda_out[a+1,:,:] = nda_in[s,:,colsh:cols]
    return nda_out

def cross_check_cspad_psana_cctbx(nda, arr):
    """Apply two-way conversions between psana and cctbx cspad arrays and compare."""
    t0_sec = time()
    nda_c = cspad_psana_from_cctbx(arr)
    dt1 = time() - t0_sec
    t0_sec = time()
    arr_c = cspad_cctbx_from_psana(nda)
    dt2 = time() - t0_sec
    print('psana ndarray is equal to converted from cctbx: %s, time = %.6f sec' % (np.array_equal(nda, nda_c), dt1))
    print('cctbx ndarray is equal to converted from psana: %s, time = %.6f sec' % (np.array_equal(arr, arr_c), dt2))

def table_nxm_cspad2x1_from_ndarr(nda):
    """returns table of cspad2x1 panels shaped as (nxm)
       generated from cspad array shaped as (N,185,388) in data.
    """
    segsize = 185*388
    a = np.array(nda) # make a copy

    if a.size == segsize:
       a.shape = (185,388)
       return a

    elif a.size == 2*segsize:
       if a.shape[-1]==2:
           from psana2.pscalib.geometry.GeometryObject import data2x2ToTwo2x1 # ,two2x1ToData2x2
           a = data2x2ToTwo2x1(a)
       a.shape = (2*185,388)
       return a

    elif a.size == 8*segsize:
       sh = a.shape = (2,4*185,388)
       return np.hstack([a[q,:] for q in range(sh[0])])

    elif a.size == 32*segsize:
       sh = a.shape = (4,8*185,388)
       return np.hstack([a[q,:] for q in range(sh[0])])

    else:
       from psana2.pyalgos.generic.NDArrUtils import reshape_to_2d
       return reshape_to_2d(a)

def table_nxm_jungfrau_from_ndarr(nda):
    """returns table of jungfrau panels shaped as (nxn)
       generated from jungfrau array shaped as (N, 512, 1024) in data.
    """
    segsize = 512*1024
    a = np.array(nda) # make a copy

    if a.size == segsize:
       a.shape = (512,1024)
       return a

    elif a.size == 2*segsize:
       logger.warning('jungfrau1m panels are stacked as [1,0]')
       sh = a.shape = (2,512,1024)
       return np.vstack([a[q,:] for q in (1,0)])

    elif a.size == 8*segsize:
       logger.warning('jungfrau4m panels are stacked as [(7,3), (6,2), (5,1), (4,0)]')
       sh = a.shape = (8,512,1024)
       return np.hstack([np.vstack([a[q,:] for q in (7,6,5,4)]),\
                         np.vstack([a[q,:] for q in (3,2,1,0)])])
    else:
       from psana2.pyalgos.generic.NDArrUtils import reshape_to_2d
       return reshape_to_2d(a)

def table_nxn_epix10ka_from_ndarr(nda, gapv=20):
    """returns table of epix10ka/epixhr panels shaped as (nxn)
       generated from epix10ka/epixhr array shaped as (N, 352, 384)/(N, 288, 384) in data.
    """
    pcols = nda.shape[-1]  # 384
    prows = nda.shape[-2]  # 352 or 288

    segsize = prows*pcols
    a = np.array(nda) # make a copy

    if a.size == segsize:
       a.shape = (prows, pcols)
       return a

    elif a.size == 4*segsize:
       logger.warning('quad panels are stacked as [(3,2),(1,0)]')
       sh = a.shape = (4, prows, pcols)
       return np.vstack([np.hstack([a[3],a[2]]), np.hstack([a[1],a[0]])])
       #sh = a.shape = (2,2*prows,pcols)
       #return np.hstack([a[q,:] for q in range(sh[0])])

    elif a.size == 16*segsize:
       sh = a.shape = (4, 4*prows, pcols)
       return np.hstack([a[q,:] for q in range(sh[0])])

    elif a.size == 7*4*segsize:
       logger.warning('quad panels are stacked as [(3,2),(1,0)]')
       agap = np.zeros((gapv, 2*pcols))
       sh = a.shape = (7, 4, prows, pcols)
       return np.vstack([np.vstack([np.hstack([a[g,3],a[g,2]]), np.hstack([a[g,1],a[g,0]]), agap]) for g in range(7)])
       #sh = a.shape = (7,2,2*prows,pcols)
       #return np.vstack([np.vstack([np.hstack([a[g,q,:] for q in range(2)]), agap]) for g in range(7)])

    elif a.size == 7*16*segsize:
       agap = np.zeros((gapv, 4*pcols))
       sh = a.shape = (7, 4, 4*prows, pcols)
       return np.vstack([np.vstack([np.hstack([a[g,q,:] for q in range(4)]), agap]) for g in range(7)])

    else:
       from psana2.pyalgos.generic.NDArrUtils import reshape_to_2d
       return reshape_to_2d(a)

"""Aliases"""

table8x8_from_cspad_ndarr = table_from_cspad_ndarr
cspad_ndarr_from_table8x8 = cspad_ndarr_from_table
# See tests in Detector/examples/ex_ndarray_from_image.py

def env_time(env):
    return 1585724400

#----------- TEST -------------

if __name__ == "__main__":

  def test_convertCheetahEventName():
    eventName = 'LCLS_2015_Feb22_r0169_022047_197f7'
    runnum, tstamp, tsec, fid = convertCheetahEventName(eventName, fmtts='%Y-%m-%dT%H:%M:%S')
    print('Method convertCheetahEventName converts Cheetah event name %s' % eventName,\
          '\nto runnum: %d  tstamp: %s  tsec: %d  fid: %s' % (runnum, tstamp, tsec, fid))

  def test_directory():
    d = '%s/XPP/xpptut15/xtc/' % INSTRUMENT_DIR
    print('test directory: %s' % d)
    return d

  def test_01():
    print('empty %s' % sys._getframe().f_code.co_name)

  def test_list_of_files_in_dir():
    print('%s:' % sys._getframe().f_code.co_name)
    lfiles = list_of_files_in_dir(test_directory())
    for i,fname in enumerate(lfiles):
        print(fname)
        if i>10:
            print('...')
            break

  def test_list_of_files_in_dir_for_pattern():
    print('%s:' % sys._getframe().f_code.co_name)
    lfiles = list_of_files_in_dir_for_pattern(test_directory(), pattern='-r0059')
    for i,fname in enumerate(lfiles): print(fname)

  def test_list_of_files_in_dir_for_ext():
    print('%s:' % sys._getframe().f_code.co_name)
    lfiles = list_of_files_in_dir_for_ext(test_directory(), ext='.xtc')
    for i,fname in enumerate(lfiles):
        print(fname)
        if i>10:
            print('...')
            break

  def test_list_of_str_from_list_of_int():
    print('%s:' % sys._getframe().f_code.co_name)
    print(list_of_str_from_list_of_int([1, 202, 203, 204], fmt='%04d'))

  def test_list_of_int_from_list_of_str():
    print('%s:' % sys._getframe().f_code.co_name)
    print(list_of_int_from_list_of_str(['0001', '0202', '0203', '0204']))

  def test_list_of_experiments(tname):
    print('%s:' % sys._getframe().f_code.co_name)

    lexps = list_of_experiments('%s/XPP' % INSTRUMENT_DIR)
    s = 'list_of_experiments():\n'
    for i,e in enumerate(lexps):
        s += '%9s '%e
        if not (i+1)%10: s += '\n'
    print(s + '\n...')

  def test_list_of_runs_in_xtc_dir():
    print('%s:' % sys._getframe().f_code.co_name)
    d = test_directory()
    print(list_of_runs_in_xtc_dir(d))

  def usage(): return 'Use command: python %s <test-number>, where <test-number> = 1,2,...,8,...' % sys.argv[0]

  def test_all(tname):
    print('\n%s\n' % usage())
    if len(sys.argv) != 2: test_01()
    elif tname == '1': test_convertCheetahEventName()
    elif tname == '2': test_list_of_experiments(tname)
    elif tname == '3': test_list_of_str_from_list_of_int()
    elif tname == '4': test_list_of_int_from_list_of_str()
    elif tname == '5': test_list_of_files_in_dir()
    elif tname == '6': test_list_of_files_in_dir_for_ext()
    elif tname == '7': test_list_of_files_in_dir_for_pattern()
    elif tname == '8': test_list_of_runs_in_xtc_dir()
    else: sys.exit('Test number parameter is not recognized.\n%s' % usage())

if __name__ == "__main__":
    import sys; global sys
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    print(50*'_', '\nTest %s' % tname)
    test_all(tname)
    sys.exit('End of Test %s' % tname)

# EOF

