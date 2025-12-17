#!/usr/bin/env python

"""
developed from: detector/testman/test-scaling-mpi-epix10ka.py

> s3dflogin
> psana
srun --partition milano --account lcls:prjdat21 -n 1 --time=05:00:00 --exclusive --pty /bin/bash --exclude sdfmilan022
srun --partition milano --account lcls:prjdat21 -n 1 --time=05:00:00 --exclusive --pty /bin/bash

in other window

> s3dflogin
ssh -Y sdfmilan216
cd LCLS/con-lcls2/lcls2
. set_env...

For time comparison:
  ../lcls2/psana/psana/pycalgos/test_cpo

mpirun -n 1  python ./lcls2/psana/psana/detector/testman/test-scaling-mpi-jungfrau.py -t2
mpirun -n 80 python ./lcls2/psana/psana/detector/testman/test-scaling-mpi-jungfrau.py -t2
mpirun -n 6  python ./lcls2/psana/psana/detector/testman/test-scaling-mpi-jungfrau.py -t2 -n400
python ./lcls2/psana/psana/detector/testman/test-scaling-mpi-jungfrau.py 99

python ../lcls2/psana/psana/detector/testman/test-scaling-mpi-jungfrau.py 50
mpirun -n 80 python ../lcls2/psana/psana/detector/testman/test-scaling-mpi-jungfrau.py 50

python detector/testman/test-scaling-mpi-jungfrau.py <Test-number>

About MPI:
https://docs.oracle.com/cd/E19356-01/820-3176-10/ExecutingPrograms.html#50413574_76503

--byslot (default)
mpirun -n 80 --bynode python ./lcls2/psana/psana/detector/testman/test-scaling-mpi-jungfrau.py 2
mpirun -np 80 --oversubscribe python ./lcls2/psana/psana/detector/testman/test-scaling-mpi-jungfrau.py 2
"""

import os
import sys
from ctypes import *

import logging
import numpy as np
from time import time
import psana.detector.NDArrUtils as ndau
import psana.detector.UtilsJungfrau as uj
import psana.detector.UtilsGraphics as ug
import psana.detector.Utils as ut
import psana.pycalgos.utilsdetector as ud
gr = ug.gr
up, BW1, BW2, BW3, MSK = uj.up, uj.BW1, uj.BW2, uj.BW3, uj.MSK
is_true = ut.is_true
info_ndarr = ndau.info_ndarr

global STRLOGLEV # sys.argv[2] if len(sys.argv)>2 else 'INFO'
global INTLOGLEV # logging._nameToLevel[STRLOGLEV]
SCRNAME = sys.argv[0].rsplit('/')[-1]
logger = logging.getLogger(__name__)

# CALIBMET options
CALIB_STD       = 0
CALIB_STD_LOCAL = 1
CALIB_LOCAL_V2  = 2
CALIB_CPP_V0    = 3
CALIB_CPP_V1    = 4
CALIB_CPP_V2    = 5
CALIB_CPP_V3    = 6
CALIB_CPP_V4    = 7
CALIB_CPP_V5    = 8

dic_calibmet = {CALIB_STD:       'CALIB_STD',\
                CALIB_STD_LOCAL: 'CALIB_STD_LOCAL',\
                CALIB_LOCAL_V2:  'CALIB_LOCAL_V2',\
                CALIB_CPP_V0:    'CALIB_CPP_V0',\
                CALIB_CPP_V1:    'CALIB_CPP_V1',\
                CALIB_CPP_V2:    'CALIB_CPP_V2',\
                CALIB_CPP_V3:    'CALIB_CPP_V3',\
                CALIB_CPP_V4:    'CALIB_CPP_V4',\
                CALIB_CPP_V5:    'CALIB_CPP_V5'}

def calib_jungfrau_local(det_raw, evt, **kwa): # cmpars=(7,3,200,10),
    """
    Taken from UtilsJungfraiu calib_jungfrau

    improved performance, reduce time and memory consumption, use peds-offset constants
    Returns calibrated jungfrau data

    - gets constants
    - gets raw data
    - evaluates (code - pedestal - offset)
    - applys common mode correction if turned on
    - apply gain factor

    Parameters

    - det_raw (run.Detector) - Detector object
    - evt (psana.Event)    - Event object
    - cmpars (tuple) - common mode parameters
        - cmpars[0] - algorithm # 7-for jungfrau
        - cmpars[1] - control bit-word 1-in rows, 2-in columns
        - cmpars[2] - maximal applied correction
    - **kwa - used here and passed to det_raw.mask_v2 or det_raw.mask_comb
      - nda_raw - if not None, substitutes evt.raw()
      - mbits - DEPRECATED parameter of the det_raw.mask_comb(...)
      - mask - user defined mask passed as optional parameter
    """

    nda_raw = kwa.get('nda_raw', None)

    arr = det_raw.raw(evt) if nda_raw is None else nda_raw # shape:(<npanels>, 512, 1024) dtype:uint16

    if is_true(arr is None, 'det_raw.raw(evt) and nda_raw are None, return None',\
               logger_method = logger.warning): return None

    odc = det_raw._odc # cache.detcache_for_detname(det_raw._det_name)
    first_entry = odc is None

    if first_entry:
        det_raw._odc = odc = uj.DetCache(det_raw, evt, **kwa) # cache.add_detcache(det_raw, evt, **kwa)
        logger.info('calib_jungfrau **kwa: %s' % str(kwa))
        logger.info(det_raw._info_calibconst()) # is called in AreaDetector

    if odc.poff is None: return arr

    if kwa != odc.kwa:
        logger.warning('IGNORED ATTEMPT to call det_raw.calib/image with different **kwargs (due to caching)'\
                       + '\n  **kwargs at first entry: %s' % str(odc.kwa)\
                       + '\n  **kwargs at this entry: %s' % str(kwa)\
                       + '\n  MUST BE FIXED - please consider to use the same **kwargs during the run in all calls to det.calib/image.')
    # 4d pedestals + offset shape:(3, 1, 512, 1024) dtype:float32

    poff, gfac, mask, cmps, inds =\
        odc.poff, odc.gfac, odc.mask, odc.cmps, odc.inds

    if first_entry:
        logger.debug('\n  ====================== det.name: %s' % det_raw._det_name\
                   +info_ndarr(arr,  '\n  calib_jungfrau first entry:\n    arr ')\
                   +info_ndarr(poff, '\n    peds+off')\
                   +info_ndarr(gfac, '\n    gfac')\
                   +info_ndarr(mask, '\n    mask')\
                   +'\n    inds: segment indices: %s' % str(inds)\
                   +'\n    common mode parameters: %s' % str(cmps)\
                   +'\n    loop over segments: %s' % odc.loop_banks)

    #nsegs = arr.shape[0]
    shseg = arr.shape[-2:] # (512, 1024)
    outa = np.zeros_like(arr, dtype=np.float32)

    #print('XXX inds:', inds)
    #print('XXX _sorted..., _segment_numbers:', det._sorted_segment_inds , det._segment_numbers)
    for iraw,i in enumerate(inds):
        arr1  = arr[iraw,:]

        #print('XXX i:', i)
        #print(info_ndarr(mask, 'XXX mask:'))

        mask1 = None if mask is None else mask[i,:] if i<mask.shape[0] else mask[0,:]
        gfac1 = None if gfac is None else gfac[:,i,:,:]
        poff1 = None if poff is None else poff[:,i,:,:]
        arr1.shape  = (1,) + shseg
        if mask1 is not None: mask1.shape = (1,) + shseg
        if gfac1 is not None: gfac1.shape = (3,1,) + shseg
        if poff1 is not None: poff1.shape = (3,1,) + shseg
        out1 = uj.calib_jungfrau_single_panel(arr1, gfac1, poff1, mask1, cmps)

        logger.debug('segment index %d arrays:' % i\
            + info_ndarr(arr1,  '\n  arr1 ')\
            + info_ndarr(poff1, '\n  poff1')\
            + info_ndarr(out1,  '\n  out1 '))
        outa[iraw,:] = out1[0,:]
    #logger.debug(info_ndarr(outa, '     outa '))
    return outa


def calib_jungfrau_single_panel(arr, gfac, poff, mask, cmps):
    """ example for 8-panel detector
    arr:  shape:(1, 512, 1024) size:524288 dtype:uint16 [2906 2945 2813 2861 3093...]
    poff: shape:(3, 1, 512, 1024) size:1572864 dtype:float32 [2922.283 2938.098 2827.207 2855.296 3080.415...]
    gfac: shape:(3, 1, 512, 1024) size:1572864 dtype:float32 [0.02490437 0.02543429 0.02541406 0.02539831 0.02544083...]
    mask: shape:(1, 512, 1024) size:524288 dtype:uint8 [1 1 1 1 1...]
    cmps: shape:(16,) size:16 dtype:float64 [  7.   1. 100.   0.   0....]
    """

    #t0_sec = time()

    # Define bool arrays of ranges
    #gr0 = arr <  BW1
    #gr1 =(arr >= BW1) & (arr<BW2)
    #gr2 = arr >= BW2
    #bad =(arr >= BW2) & (arr<BW3) # 10 - badly frozen pixel
    # time for gr0, gr1, gr2 170-200 us

    #gr0 = arr & BW3 == 0                                   # 00
    #gr1 = np.logical_and(arr & BW1 == BW1, arr & BW2 == 0) # 01
    #gr2 = arr & BW2 == BW2                                 # 10 or 11
    # time for gr0, gr1, gr2 320-370 us

    gbits = np.array(arr>>14, dtype=np.uint8) # 00/01/11 - gain bits for mode 0,1,2
    gr0, gr1, gr2 = gbits==0, gbits==1, gbits==3
    # time for gr0, gr1, gr2 140-200 us

    #print('XXXX gain range def time %.3f us' % 1e6*(time() - t0_sec))

    factor = np.select((gr0, gr1, gr2), (gfac[0,:], gfac[1,:], gfac[2,:]), default=0) # 2msec
    pedoff = np.select((gr0, gr1, gr2), (poff[0,:], poff[1,:], poff[2,:]), default=0)
    # time for gr0, gr1, gr2, factor and pedoff 2-10 ms

    #print('XXXX gain range def time %.3f us' % 1e6*(time() - t0_sec))

    # Subtract offset-corrected pedestals
    arrf = (np.array(arr & MSK, dtype=np.float32) - pedoff) * factor
    #arrf -= pedoff
    #arrf *= factor
    return arrf if mask is None else arrf * mask



def calib_jungfrau_compare(det_raw, evt, **kwa): # cmpars=(7,3,200,10),
    """the same as calib_jungfrau_local, but
       - add switch for panel processing in python or C++
    """
    nda_raw = kwa.get('nda_raw', None)
    size_blk = kwa.get('size_blk', 512*1024) # single panel size

    arr = det_raw.raw(evt) if nda_raw is None else nda_raw # shape:(<npanels>, 512, 1024) dtype:uint16

    if is_true(arr is None, 'det_raw.raw(evt) and nda_raw are None, return None',\
               logger_method = logger.warning): return None

    odc = det_raw._odc # cache.detcache_for_detname(det_raw._det_name)
    first_entry = odc is None

    if first_entry:
        det_raw._odc = odc = uj.DetCache(det_raw, evt, **kwa) # cache.add_detcache(det_raw, evt, **kwa)
        logger.info('calib_jungfrau **kwa: %s' % str(kwa))
        logger.info(det_raw._info_calibconst()) # is called in AreaDetector

    if odc.poff is None: return arr

    if kwa != odc.kwa:
        logger.warning('IGNORED ATTEMPT to call det_raw.calib/image with different **kwargs (due to caching)'\
                       + '\n  **kwargs at first entry: %s' % str(odc.kwa)\
                       + '\n  **kwargs at this entry: %s' % str(kwa)\
                       + '\n  MUST BE FIXED - please consider to use the same **kwargs during the run in all calls to det.calib/image.')
    # 4d pedestals + offset shape:(3, 1, 512, 1024) dtype:float32

    ccons, poff, gfac, mask, cmps, inds, outa =\
        odc.ccons, odc.poff, odc.gfac, odc.mask, odc.cmps, odc.inds, odc.outa

    if first_entry:
        logger.info('\n  ====================== det.name: %s' % det_raw._det_name\
                   +info_ndarr(arr,  '\n  calib_jungfrau first entry:\n    arr ')\
                   +info_ndarr(poff, '\n    peds+off')\
                   +info_ndarr(gfac, '\n    gfac')\
                   +info_ndarr(mask, '\n    mask')\
                   +info_ndarr(outa, '\n    outa')\
                   +info_ndarr(ccons, '\n   ccons (peds+off, gain*mask)', last=8, vfmt='%0.3f')\
                   +'\n    inds: segment indices: %s' % str(inds)\
                   +'\n    common mode parameters: %s' % str(cmps)\
                   +'\n    loop over segments: %s' % odc.loop_banks)
        print(uj.info_gainbits_statistics(arr))
        print(uj.info_gainrange_statistics(arr))
        print(uj.info_gainrange_fractions(arr))

    if CALIBMET in (CALIB_LOCAL_V2, CALIB_CPP_V0):
      shseg = arr.shape[-2:] # (512, 1024)
      # loop over segments here in python
      outa.shape = arr.shape
      for iraw,i in enumerate(inds):
        arr1  = arr[iraw,:]
        mask1 = None if mask is None else mask[i,:] if i<mask.shape[0] else mask[0,:]
        gfac1 = None if gfac is None else gfac[:,i,:,:]
        poff1 = None if poff is None else poff[:,i,:,:]
        outa1 = outa[i,:]
        arr1.shape  = (1,) + shseg
        if mask1 is not None: mask1.shape = (1,) + shseg
        if gfac1 is not None: gfac1.shape = (3,1,) + shseg
        if poff1 is not None: poff1.shape = (3,1,) + shseg
        #out1 = uj.calib_jungfrau_single_panel(arr1, gfac1, poff1, mask1, cmps)  if CALIBMET == CALIB_LOCAL_V2 else\
        out1 = calib_jungfrau_single_panel(arr1, gfac1, poff1, mask1, cmps)  if CALIBMET == CALIB_LOCAL_V2 else\
               ud.calib_jungfrau_v0       (arr1, poff1, gfac1, mask1, outa1) if CALIBMET == CALIB_CPP_V0 else\
               None # raw, peds, gain, mask, databits, out

        logger.debug('segment index %d arrays:' % i\
            + info_ndarr(arr1,  '\n  arr1 ')\
            + info_ndarr(poff1, '\n  poff1')\
            + info_ndarr(out1,  '\n  out1 '))
        if CALIBMET == CALIB_LOCAL_V2: outa[iraw,:] = out1[0,:]

    elif CALIBMET == CALIB_CPP_V1:
        # full processing in C++ reshaped constants (<NPIXELS>,8)
        #print('XXXXX COMENT OUT calib_jungfrau_v1, in test %s' % dic_calibmet[CALIBMET])
        #print('XXXXX PASS FROM PARSER size_blk, in test %d' % size_blk)
        #print(info_ndarr(arr,   '    arr',   first=1000, last=1008))
        #print(info_ndarr(ccons, '    ccons', first=1000, last=1008))
        #print(info_ndarr(outa,  '    outa',  first=1000, last=1008))
        ud.calib_jungfrau_v1(arr, ccons, size_blk, outa)

    elif CALIBMET == CALIB_CPP_V2:
        # full processing in C++ reshaped constants (8,<NPIXELS>)
        #logger.info(info_ndarr(ccons, 'XXX CHANGE TO v2    ccons', first=1000, last=1008))
        ud.calib_jungfrau_v2(arr, ccons, size_blk, outa)

    elif CALIBMET == CALIB_CPP_V3:
        # full processing in C++ reshaped constants (4, <NPIXELS>, 2)
        #logger.info(info_ndarr(ccons, 'XXX CHANGE TO v2    ccons', first=1000, last=1008))
        ud.calib_jungfrau_v3(arr, ccons, size_blk, outa)

    elif CALIBMET == CALIB_CPP_V4:
        return ud.calib_jungfrau_v4_empty()

    elif CALIBMET == CALIB_CPP_V5:
        return ud.calib_jungfrau_v5_empty(arr, ccons, size_blk, outa)

    return outa



def test_event_loop(calibmet, **kwargs):
    """
       optimization of det.raw.calib for mpi
       datinfo -k exp=uedcom103,run=812 -d epixquad
       https://confluence.slac.stanford.edu/display/LCLSIIData/psana#psana-PublicPracticeData
    """
    global CALIBMET

    CALIBMET = calibmet
    str_dskwargs = kwargs.get('dskwargs', 'exp=mfx100848724,run=51')
    detname      = kwargs.get('detname', 'jungfrau')
    rank_test    = kwargs.get('rank_test', 0)
    cmpars       = kwargs.get('cmpars', None)
    events       = kwargs.get('events', 100)
    plot_img     = kwargs.get('plot_img', False)
    fname_prefix = kwargs.get('fname_prefix', 'summary')
    #cversion     = kwargs.get('cversion', 1)
    kwargs.setdefault('logmet_init', logger.info)

    arrts = np.zeros(events, dtype=float) if calibmet in (CALIB_LOCAL_V2, CALIB_CPP_V0, CALIB_CPP_V1, CALIB_CPP_V2, CALIB_CPP_V3) else\
            np.zeros(events, dtype=float) if calibmet in (CALIB_STD, CALIB_STD_LOCAL) else\
            np.zeros(events, dtype=float)

    # see ~/LCLS/con-py3/Detector/examples/test-scaling-mpi-epix10ka2m.py
    #from psana import MPIDataSource # lcls1
    from psana import DataSource
    from psana.detector.Utils import get_hostname
    import psutil

    #if do_mpi : # VVVVV for MPI
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    hostname = get_hostname()
    cpu_num = psutil.Process().cpu_num()
    s_rsc = 'rank:%03d/%03d cpu:%03d' % (rank, size, cpu_num)
    print(s_rsc)
    # ^^^^^ for MPI

    dskwargs = up.datasource_kwargs_from_string(str_dskwargs)
    dskwargs['max_events'] = events
    dskwargs['batch_size'] = 1
    ds = DataSource(**dskwargs)

    flimg = None
    counter = 0
    break_loop = False
    runnum, expt = None, None
    for nrun,orun in enumerate(ds.runs()):
      det = orun.Detector(detname)
      runnum, expt = orun.runnum, orun.expt
      #if size==1 and break_loop:
      #    print('break run for %s' % s_rsc)
      #    break
      print('  det.raw._shape_as_daq():', det.raw._shape_as_daq())
      print(info_ndarr(det.raw._pedestals(), '\n  peds ', first=1000, last=1005))

      #for nstep,step in enumerate(orun.steps()):
      #  if size==1 and break_loop:
      #      print('break step for %s' % s_rsc)
      #      break
      #  for nevt,evt in enumerate(step.events()):

      for nevt,evt in enumerate(orun.events()):

          if nevt<2:
              print('\n=============== evt: %d' % nevt)
              #print('\n=============== step %d evt: %d config gain mode: %s' % (nstep, nevt, ue.find_gain_mode(det.raw, evt)))

          #if size==1 and nevt>events-1:
          #   break_loop = True
          #   print('\n break event loop for rank: %03d' % rank)
          #   break

          if nevt>events-1:
          #    print('  evt: %d continue event loop for rank: %03d' % (nevt, rank))
              continue

          s = '== evt %04d %s' % (nevt, s_rsc)
          t0_sec = time()
          raw = det.raw.raw(evt)
          dt_sec_raw = time()-t0_sec
          s += ' ' + dic_calibmet[calibmet]
          t0_sec = time()

          if calibmet == CALIB_STD:
            calib = det.raw.calib(evt, **kwargs)   # calib_std(det, evt, cmpars=None)

          elif calibmet in (CALIB_CPP_V4, CALIB_CPP_V5):
            dt_us_cpp, dt_sec_cy = calib_jungfrau_compare(det.raw, evt, **kwargs)
            calib = raw
            s += ' dt cpp: %.3f msec, cython: %.3f msec' % (dt_us_cpp*0.001, dt_sec_cy*1000)

          elif calibmet in (CALIB_LOCAL_V2, CALIB_CPP_V0, CALIB_CPP_V1, CALIB_CPP_V2, CALIB_CPP_V3):
            calib = calib_jungfrau_compare(det.raw, evt, **kwargs)

          elif calibmet == CALIB_STD_LOCAL:
            calib = calib_jungfrau_local(det.raw, evt, **kwargs) # cmpars=cmpars)

          dt_sec_calib = time()-t0_sec

          if calib is None: continue
          arrts[nevt] = dt_sec_calib

          #s += info_ndarr(nda, '\n  calib', first=1000, last=1005)
          s += ' dt raw: %.3f msec, calib: %.3f msec' % (dt_sec_raw*1000, dt_sec_calib*1000)
          print(s)
          #print(info_ndarr(calib, (12*' ')+'calib'))

          if plot_img:
            nda = calib
            if nda is None: continue
            img = det.raw.image(evt, nda=nda)

            if flimg is None:
               flimg = ug.fleximage(img, arr=nda, h_in=10, w_in=11.2)
            gr.set_win_title(flimg.fig, titwin='Event %d' % nevt)
            flimg.update(img, arr=nda) #, amin=0, amax=60000)
            gr.show(mode='DO NOT HOLD')

    #if rank == rank_test:
    if True: #calibmet in (CALIB_STD, CALIB_STD_LOCAL, CALIB_LOCAL_V2, CALIB_CPP_V0, CALIB_CPP_V1, CALIB_CPP_V2, CALIB_CPP_V3):
        dt = 1000*arrts[1:]
        print(info_ndarr(dt, name='%s times(msec):' % s_rsc, first=0, last=100, vfmt='%0.3f'))
        dt_sel = dt[dt>0]
        med_dt = np.median(dt_sel) if dt_sel.size > 0 else 0
        s = '*** %s evts: %4d median time for calib %5.1f msec or %7.3f Hz' % (s_rsc, dt_sel.size, med_dt, (1000./med_dt if med_dt>0 else 0))
        print(s)
        with open('summary.txt', 'a') as f:
          f.write('\n'+s)
          #f.close()


    if plot_img: gr.show()

    print('END OF TEST %s'% s_rsc)


#datinfo -k exp=mfx100848724,run=51 -d jungfrau # det.raw.raw(evt) shape:(32, 512, 1024)
#pedestals    from exp:mfx100848724 run:0049 shape:(3, 32, 512, 1024)     in step0: Event 17871


def argument_parser():
    from argparse import ArgumentParser
    d_tname = '0'
    d_dskwargs = 'exp=mfx100848724,run=51'  # None
    d_detname  = 'jungfrau' # None
    d_events   = 10
    d_loglevel = 'INFO' # 'DEBUG'
    d_plot_img = 0
    d_cmpars   = None
    d_size_blk = 1024 # 512*1024
#    d_cversion = 1

    h_tname    = '(str) test name, usually numeric, default = %s' % d_tname
    h_dskwargs = '(str) dataset kwargs for DataSource(**kwargs), default = %s' % d_dskwargs
    h_detname  = 'detector name, default = %s' % d_detname
    h_events   = 'number of events to process, default = %d' % d_events
    h_loglevel = 'logging level, one of %s, default = %s' % (', '.join(tuple(logging._nameToLevel.keys())), d_loglevel)
    h_plot_img = 'image bitword to plot images, default = %d' % d_plot_img
    h_cmpars   = '(str) list of common mode parameters, i.g. (7,7,200,10), default = %s' % d_cmpars
    h_size_blk = '(int) block size (number of pixels) to split entire array for processing, default = %d' % d_size_blk
#    h_cversion = '(int) array ordering version for calibration constants, default = %d' % d_cversion

    parser = ArgumentParser(description='%s tests of jungfrau calib berformance with mpi' % SCRNAME, usage=usage())
    parser.add_argument('-t', '--tname',    default=d_tname,    type=str, help=h_tname)
    parser.add_argument('-k', '--dskwargs', default=d_dskwargs, type=str, help=h_dskwargs)
    parser.add_argument('-d', '--detname',  default=d_detname,  type=str, help=h_detname)
    parser.add_argument('-n', '--events',   default=d_events,   type=int, help=h_events)
    parser.add_argument('-L', '--loglevel', default=d_loglevel, type=str, help=h_loglevel)
    parser.add_argument('-p', '--plot_img', default=d_plot_img, type=int, help=h_plot_img)
    parser.add_argument('-c', '--cmpars',   default=d_cmpars,   type=str, help=h_cmpars)
    parser.add_argument('-s', '--size_blk', default=d_size_blk, type=int, help=h_size_blk)
#    parser.add_argument('-C', '--cversion', default=d_cversion, type=int, help=h_cversion)
    return parser

def usage():
    import inspect
    return '\n  dataset test: datinfo -k exp=mfx100848724,run=51 -d jungfrau'\
        +'\n\n  %s -t <tname> [other kwargs]\n' % sys.argv[0].split('/')[-1]\
    + '\n'.join([s for s in inspect.getsource(selector).split('\n') if "tname ==" in s or "tnum in" in s])

def selector():
    if len(sys.argv) < 2:
        print(usage())
        sys.exit('EXIT due to MISSING PARAMETERS')

    parser = argument_parser()
    args = parser.parse_args()
    kwargs = vars(args)

    cmpars = kwargs['cmpars']
    kwargs['cmpars'] = None if cmpars is None else eval(kwargs['cmpars'])

    #print('parser.parse_args()', args)
    print(ut.info_parser_arguments(parser, title='parser parameters:'))

    STRLOGLEV = args.loglevel
    INTLOGLEV = logging._nameToLevel[STRLOGLEV]
    logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d %(filename)s: %(message)s', level=INTLOGLEV)

    tname = args.tname  # sys.argv[1] if len(sys.argv)>1 else '0'
    tnum = int(tname)

    if   tname ==  '0': test_event_loop(CALIB_STD, cversion=0, **kwargs)      # det.raw.calib - current
    elif tname ==  '1': test_event_loop(CALIB_STD_LOCAL, **kwargs)            # the same as det.raw.calib but in this module
    elif tname ==  '2': test_event_loop(CALIB_LOCAL_V2,  **kwargs)            # version for comparison in python
    elif tname ==  '3': test_event_loop(CALIB_CPP_V0,    **kwargs)            # version for comparison in C++
    elif tname ==  '4': test_event_loop(CALIB_CPP_V1, cversion=1, **kwargs)   # version for comparison in C++ reshaped constants (<NPIXELS>,8)
    elif tname ==  '5': test_event_loop(CALIB_CPP_V2, cversion=2, **kwargs)   # version for comparison in C++ reshaped constants (8,<NPIXELS>)
    elif tname ==  '6': test_event_loop(CALIB_CPP_V3, cversion=3, **kwargs)   # version for comparison in C++ reshaped constants (4,<NPIXELS>,2)
    elif tname ==  '7': test_event_loop(CALIB_CPP_V4, cversion=3, **kwargs)   # version for empty call to cython-C++ WITHOUT PARAMETERS
    elif tname ==  '8': test_event_loop(CALIB_CPP_V5, cversion=3, **kwargs)   # version for empty call to cython-C++ WITH PARAMETERS
    elif tname == '10': test_event_loop(CALIB_STD, cversion=0, **kwargs)      # det.raw.calib - python original
    elif tname == '11': test_event_loop(CALIB_STD, cversion=1, **kwargs)      # C++ constants (<NPIXELS>,8)
    elif tname == '12': test_event_loop(CALIB_STD, cversion=2, **kwargs)      # C++ constants (8,<NPIXELS>)
    elif tname == '13': test_event_loop(CALIB_STD, cversion=3, **kwargs)      # C++ constants (4,<NPIXELS>,2)
    elif tname == '14': test_event_loop(CALIB_STD,             **kwargs)      # C++ constants (4,<NPIXELS>,2) DEFAULT

    else:
        print(usage())
        exit('\nTEST "%s" IS NOT IMPLEMENTED'%tname)
    sys.exit(0)


if __name__ == "__main__":
    selector()

# EOF
