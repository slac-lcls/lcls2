#!/usr/bin/env python

"""
LCLS1 anolog: ~/LCLS/con-py3/Detector/examples/test-scaling-mpi-epix10ka2m.py

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


mpirun -n 1  python ./lcls2/psana/psana/detector/testman/test-scaling-mpi-epix10ka.py 2
mpirun -n 80 python ./lcls2/psana/psana/detector/testman/test-scaling-mpi-epix10ka.py 2
mpirun -n 5  python ./lcls2/psana/psana/detector/testman/test-scaling-mpi-epix10ka.py 2
mpirun -n 5  python ./lcls2/psana/psana/detector/testman/test-scaling-mpi-epix10ka.py 80
python ./lcls2/psana/psana/detector/testman/test-scaling-mpi-epix10ka.py 99

python ../lcls2/psana/psana/detector/testman/test-scaling-mpi-epix10ka.py 50
mpirun -n 80 python ../lcls2/psana/psana/detector/testman/test-scaling-mpi-epix10ka.py 50

python detector/testman/test-scaling-mpi-epix10ka2m.py <Test-number>

About MPI:
https://docs.oracle.com/cd/E19356-01/820-3176-10/ExecutingPrograms.html#50413574_76503

--byslot (default)
mpirun -n 80 --bynode python ./lcls2/psana/psana/detector/testman/test-scaling-mpi-epix10ka.py 2
mpirun -np 80 --oversubscribe python ./lcls2/psana/psana/detector/testman/test-scaling-mpi-epix10ka.py 2
"""

import os
import sys
from ctypes import *

import logging
import numpy as np
from time import time
import psana.detector.UtilsEpix10ka as ue
import psana.detector.UtilsGraphics as ug
import psana.detector.Utils as ut
gr = ug.gr

global STRLOGLEV # sys.argv[2] if len(sys.argv)>2 else 'INFO'
global INTLOGLEV # logging._nameToLevel[STRLOGLEV]
SCRNAME = sys.argv[0].rsplit('/')[-1]
logger = logging.getLogger(__name__)

# CALIBMET options
CALIB_STD      = 0
CALIB_LOCAL    = 1
CALIB_LOCAL_V2 = 2
CALIB_LOCAL_V3 = 3
CALIB_V2       = 4
SIM0          = 80
SIM1          = 81
SIM2          = 82
SIM3          = 83
SIM4          = 84
SIM5          = 85
SIMS = (SIM0, SIM1, SIM2, SIM3, SIM4, SIM5)

from psana.detector.ArrayIterator import ArrayIterator, test_ArrayIterator

def calib_epix10ka_any_local(det_raw, evt, cmpars=None, **kwa): #cmpars=(7,2,100)):
    """
    the same as det.raw.calib(evt, cmpars=None)

    Algorithm
    ---------
    - gets constants
    - gets raw data
    - evaluates (code - pedestal - offset)
    - applys common mode correction if turned on
    - apply gain factor

    Parameters
    ----------
    - det_raw (psana.Detector.raw) - Detector.raw object
    - evt (psana.Event)    - Event object
    - cmpars (tuple) - common mode parameters
          = None - use pars from calib directory
          = cmpars=(<alg>, <mode>, <maxcorr>)
            alg is not used
            mode =0-correction is not applied, =1-in rows, =2-in cols-WORKS THE BEST
            i.e: cmpars=(7,0,100) or (7,2,100) or (7,7,100)
    - **kwa - used here and passed to det_raw.mask_comb
      - nda_raw - substitute for det_raw.raw(evt)
      - mbits - parameter of the det_raw.mask_comb(...)
      - mask - user defined mask passed as optional parameter

    Returns
    -------
      - calibrated epix10ka data
    """

    #print('XXXX calib_epix10ka_any_local kwa:', kwa)

    nda_raw = kwa.get('nda_raw', None)
    raw = det_raw.raw(evt) if nda_raw is None else nda_raw # shape:(352, 384) or suppose to be later (<nsegs>, 352, 384) dtype:uint16
    if ue.cond_msg(raw is None, msg='raw is None'): return None

    gmaps = ue.gain_maps_epix10ka_any(det_raw, evt) #tuple: 7 x shape:(4, 352, 384)
    if ue.cond_msg(gmaps is None, msg='gmaps is None'): return None

    store = det_raw._store_ = ue.Storage(det_raw, cmpars=cmpars, **kwa) if det_raw._store_ is None else det_raw._store_  #perpix=True
    store.counter += 1
    #if not store.counter%100:
    if store.counter < 1: ue.print_gmaps_info(gmaps)

    factor = ue.event_constants_for_gmaps(gmaps, store.gfac, default=1)  # 3d gain factors
    pedest = ue.event_constants_for_gmaps(gmaps, store.peds, default=0)  # 3d pedestals

    arrf = np.array(raw & det_raw._data_bit_mask, dtype=np.float32)
    if pedest is not None: arrf -= pedest

    if store.cmpars is not None:
        ue.common_mode_epix_multigain_apply(arrf, gmaps, store)

    logger.debug(ue.info_ndarr(arrf,  'arrf:'))

    if ue.cond_msg(factor is None, msg='factor is None - substitute with 1', output_meth=logger.warning): factor = 1

    mask = store.mask
    return arrf * factor if mask is None else arrf * factor * mask # gain correction





def calib_epix10ka_any_local_v2(det_raw, evt, **kwa):
    """ v2: add time points, get rid of common mode correction
    """
    t0 = time()
    nda_raw = kwa.get('nda_raw', None)
    raw = det_raw.raw(evt) if nda_raw is None else nda_raw # shape:(352, 384) or suppose to be later (<nsegs>, 352, 384) dtype:uint16
    if ue.cond_msg(raw is None, msg='raw is None'): return None

    t1 = time()
    gmaps = ue.gain_maps_epix10ka_any(det_raw, evt) #tuple: 7 x shape:(4, 352, 384)
    if ue.cond_msg(gmaps is None, msg='gmaps is None'): return None
    #print(ue.info_ndarr(gmaps,  'XXX gmaps:'))

    t2 = time()
    store = det_raw._store_ = ue.Storage(det_raw, **kwa) if det_raw._store_ is None else det_raw._store_  #perpix=True
    store.counter += 1
    if store.counter < 1: ue.print_gmaps_info(gmaps)

    t3 = time()
    factor = ue.event_constants_for_gmaps(gmaps, store.gfac, default=1)  # 3d gain factors
    pedest = ue.event_constants_for_gmaps(gmaps, store.peds, default=0)  # 3d pedestals

    t4 = time()
    arrf = np.array(raw & det_raw._data_bit_mask, dtype=np.float32)

    t5 = time()
    if pedest is not None: arrf -= pedest

    logger.debug(ue.info_ndarr(arrf, 'arrf:'))
    if ue.cond_msg(factor is None, msg='factor is None - substitute with 1', output_meth=logger.warning): factor = 1

    t6 = time()
    if store.cmpars is not None:
        print('IT DOES CM')
        ue.common_mode_epix_multigain_apply(arrf, gmaps, store)

    t7 = time()
    mask = store.mask
    res = arrf * factor if mask is None else arrf * factor * mask # gain correction

    t8 = time()
    return res, (t0, t1, t2, t3, t4, t5, t6, t7, t8)









def calib_epix10ka_any_local_v3(det_raw, evt, **kwa):
    """ v2: add time points, get rid of common mode correction
    """
    t0 = time()
    nda_raw = kwa.get('nda_raw', None)
    raw = det_raw.raw(evt) if nda_raw is None else nda_raw # shape:(352, 384) or suppose to be later (<nsegs>, 352, 384) dtype:uint16
    if ue.cond_msg(raw is None, msg='raw is None'): return None

    t1 = time()
    gmaps = ue.gain_maps_epix10ka_any(det_raw, evt) #tuple: 7 x shape:(4, 352, 384)
    if ue.cond_msg(gmaps is None, msg='gmaps is None'): return None

    t2 = time()
    store = det_raw._store_ = ue.Storage(det_raw, **kwa) if det_raw._store_ is None else det_raw._store_  #perpix=True
    store.counter += 1
    if store.counter < 1: ue.print_gmaps_info(gmaps)

    t3 = time()
    factor = ue.event_constants_for_gmaps(gmaps, store.gfac, default=1)  # 3d gain factors
    pedest = ue.event_constants_for_gmaps(gmaps, store.peds, default=0)  # 3d pedestals

    t4 = time()
    arrf = np.array(raw & det_raw._data_bit_mask, dtype=np.float32)

    #t5 = time()
    if pedest is not None: arrf -= pedest

    logger.debug(ue.info_ndarr(arrf, 'arrf:'))
    if ue.cond_msg(factor is None, msg='factor is None - substitute with 1', output_meth=logger.warning): factor = 1

    #t6 = time()
    #if store.cmpars is not None:
    #    print('IT DOES CM')
    #    ue.common_mode_epix_multigain_apply(arrf, gmaps, store)

    #t7 = time()
    mask = store.mask
    res = arrf * factor if mask is None else arrf * factor * mask # gain correction

    t5 = t6 = t7 = t8 = time()
    #t8 = time()
    return res, (t0, t1, t2, t3, t4, t5, t6, t7, t8)




def test_event_loop(*args, **kwargs):
    """
       optimization of det.raw.calib for mpi
       @sdfiana002
       epixquad shape:(4, 352, 384)
       datinfo -k exp=uedcom103,run=812 -d epixquad
       https://confluence.slac.stanford.edu/display/LCLSIIData/psana#psana-PublicPracticeData
       uedcom103, ueddaq02 epix10ka
       /sdf/data/lcls/drpsrcf/ffb/rix/rixc00121/xtc
       /sdf/data/lcls/drpsrcf/ffb/ued/*/xtc/
       /sdf/data/lcls/ds/ued/uedcom103/xtc/
            runs
            812-dark
            811 - FM - 1 steps x 1755 events
            796-scan FL [0-15] 16 steps x 1080 evts
            646-dark -          5 steps x 2000 events
            590-scan FH [0-15] 16 steps x 1080 evts
            509-scan FL         1 steps x 8652 evts
            470-scan FM [0-15] 16 steps x 1080 evts
            460-scan FL [0-15] 16 steps x 1080 evts
            422-scan FL [0-15] 16 steps x 1080 evts
            419-scan FL [0-15] 16 steps x 1080 evts
             95-scan FH [0-15] 16 steps x 6000 evts
             79-scan FL [0-39] 40 steps x 1000 evts
       ffb: uedc00104/  0 uedcom103/  0 ueddaq02/  0 uedpsdm02/  0 uedtst088

    """
    CALIBMET     = kwargs.get('CALIBMET', CALIB_STD)
    rank_test    = kwargs.get('rank_test', 0)
    cmpars       = kwargs.get('cmpars', None)
    nevents      = kwargs.get('nevents', 100)
    do_image     = kwargs.get('do_image', False)
    fname_prefix = kwargs.get('fname_prefix', 'summary')
    arrts = np.zeros((nevents,9), dtype=float) if CALIBMET in (CALIB_LOCAL_V2, CALIB_LOCAL_V3) else\
            np.zeros(nevents, dtype=float)     if CALIBMET == CALIB_LOCAL else\
            np.zeros(nevents, dtype=float)

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

    ds = DataSource(exp='uedcom103',run=95, max_events=nevents if size==1 else 96000) #nevents*size) # dark run
    #ds = DataSource(exp='uedcom103',run=812)

    flimg = None
    counter = 0
    break_loop = False
    runnum, expt = None, None
    for nrun,orun in enumerate(ds.runs()):
      det = orun.Detector('epixquad')
      runnum, expt = orun.runnum, orun.expt
      #if size==1 and break_loop:
      #    print('break run for %s' % s_rsc)
      #    break
      print('  det.raw._shape_as_daq():', det.raw._shape_as_daq())
      print(ue.info_ndarr(det.raw._pedestals(), '\n  peds ', first=1000, last=1005))

      #for nstep,step in enumerate(orun.steps()):
      #  if size==1 and break_loop:
      #      print('break step for %s' % s_rsc)
      #      break
      #  for nevt,evt in enumerate(step.events()):
      for nevt,evt in enumerate(orun.events()):

          if nevt<2:
              print('\n=============== evt: %d config gain mode: %s' % (nevt, ue.find_gain_mode(det.raw, evt)))
              #print('\n=============== step %d evt: %d config gain mode: %s' % (nstep, nevt, ue.find_gain_mode(det.raw, evt)))

          #if size==1 and nevt>nevents-1:
          #   break_loop = True
          #   print('\n break event loop for rank: %03d' % rank)
          #   break

          if nevt>nevents-1:
          #    print('  evt: %d continue event loop for rank: %03d' % (nevt, rank))
              continue

          s = '== evt %04d %s' % (nevt, s_rsc)
          t0_sec = time()
          raw = det.raw.raw(evt)
          dt_sec_raw = time()-t0_sec

          t0_sec = time()
          if CALIBMET == CALIB_STD:
            s += ' CALIB_STD'
            calib = det.raw.calib(evt, cmpars=None)   # calib_std(det, evt, cmpars=None)
            if calib is None: continue
            dt_sec = time()-t0_sec
            arrts[nevt] = dt_sec

          elif CALIBMET == CALIB_LOCAL:
            s += ' CALIB_LOCAL'
            calib = calib_epix10ka_any_local(det.raw, evt, cmpars=cmpars)
            if calib is None: continue
            arrts[nevt] = time()-t0_sec

          elif CALIBMET == CALIB_LOCAL_V2:
            s += ' CALIB_LOCAL_V2'
            calib, times = calib_epix10ka_any_local_v2(det.raw, evt, cmpars=cmpars)
            if calib is None: continue
            if nevt<nevents:
                dt_sec = times[-1] - times[0]
                if isinstance(times, tuple) and len(times)==arrts.shape[1]:
                    counter += 1
                    arrts[nevt] = times

          elif CALIBMET == CALIB_LOCAL_V3:
            s += ' CALIB_LOCAL_V3'
            calib, times = calib_epix10ka_any_local_v3(det.raw, evt, cmpars=cmpars)
            if calib is None: continue
            if nevt<nevents:
                dt_sec = times[-1] - times[0]
                if isinstance(times, tuple) and len(times)==arrts.shape[1]:
                    counter += 1
                    arrts[nevt] = times

          nda = calib
          dt_sec_calib = time()-t0_sec
          #s += ue.info_ndarr(raw, '\n  raw  ', first=1000, last=1005)
          #s += ue.info_ndarr(nda, '\n  calib', first=1000, last=1005)
          s += ' dt raw: %.3f msec, calib: %.3f msec' % (dt_sec_raw*1000, dt_sec_calib*1000)
          print(s)

          if do_image:
            if nda is None: continue
            img = det.raw.image(evt, nda=nda)

            if flimg is None:
               flimg = ug.fleximage(img, arr=nda, h_in=10, w_in=11.2)
            gr.set_win_title(flimg.fig, titwin='Event %d' % nevt)
            flimg.update(img, arr=nda) #, amin=0, amax=60000)
            gr.show(mode='DO NOT HOLD')

    #if rank == rank_test:
    if True:
            dt = arrts[1:,-1] - arrts[1:,0] if CALIBMET in (CALIB_LOCAL_V2, CALIB_LOCAL_V3) else\
                 arrts[1:]
            #print(ue.info_ndarr(dt, 'rank:%03d times:' % rank, last=nevents))
            #tit = '%s rank %03d of %03d cpu_num %03d' % (hostname, rank, size, cpu_num)

            #print('XXX dt=', str(dt))

            arrts = arrts[1:,:] # exclude 1st event
            arrts[:,1:] -= arrts[:,0:-1] # evaluate dt for each step
            arrts[:,0]   = dt   # elt 0: total time per event
            arrts[:,:]  *= 1000 # convert sec > msec

            if CALIBMET in (CALIB_LOCAL_V2, CALIB_LOCAL_V3):
               #print('XXX', fname_prefix, expt, runnum, size)
               fname = '%s-%s-r%04d-ncpu-%03d.txt' % (fname_prefix, expt, runnum, size)
               if rank == 0:
                   s = title(arrts)
                   print('save summary in file: %s\n%s' % (fname, s))
                   ut.save_textfile(s+'\n', fname, mode='a')

               print_summary(arrts, show_arrts=False, cmt='%s number of recs: %d' % (s_rsc, counter), fname=fname)

    if do_image: gr.show()

    #print('END OF TEST %s'% s_rsc)



def title(arrts):
    ntpoints = arrts.shape[1]
    s = '     '.join(['  t%02d'%i for i in range(0, ntpoints)])
    return 'dt, ms:%s' % s


def print_summary(arrts, show_arrts=False, cmt='', fname='summary.txt'):
    #print(ue.info_ndarr(arrts, 'arrts[msec]:', last=100))
    ntpoints = arrts.shape[1]
    fmt = ntpoints*' %9.4f'
    if show_arrts:
      for n,r in enumerate(range(arrts.shape[0])):
        times = arrts[r,:]
        if n%100 == 0:
          print('%03d '%n, fmt % tuple(times))

    #tmed = np.median(arrts[:,1:], axis=0)
    tmed = np.median(arrts[:,:], axis=0)
    s = 'medi:' + (fmt % tuple(tmed)) + ' ' + cmt
    print(s)
    ut.save_textfile(s+'\n', fname, mode='a')


def parse_summary(*args, **kwargs):
    fname = kwargs.get('fname', '')
    ptrn = kwargs.get('ptrn', '100')
    s = ut.load_textfile(fname)
    recs = s.split('\n')
    arrts = []
    for r in recs:
        print(r)
        f = r.split()
        if isinstance(f, list) and len(f)>1 and f[-1]==ptrn:
            arrts.append(tuple([float(v) for v in f[1:9]]))
    arrts = np.array(arrts)
    print('mean:    %s for %d fully loaded cpus' % ('    '.join(['%.4f'%v for v in np.mean(arrts, axis=0)]), arrts.shape[0]))



#def test_sim_(*args, **kwargs):

def myfunc(a, p, g, m):
    return ((a & M14) - p) * g * m

uf = np.frompyfunc(myfunc, 3, 1)

vf = np.vectorize(myfunc) # , otypes=np.float32, signature='(1),(1),(1)->(1)')
#vf = np.vectorize(myfunc, otypes=[np.float32], signature='(n),(n),(n)->(n)')

def test_simulation(*args, **kwargs):
    import psana.pyalgos.generic.NDArrGenerators as ag
    import psana.pycalgos.utilsdetector as ud
    #import psana.pscalib.calib.CalibConstants import as cc

    CALIBMET  = kwargs.get('CALIBMET', SIM0)
    nloops    = kwargs.get('nloops', 500)
    rank_test = kwargs.get('rank_test', 0)
    rank = 0

    if False:
        #if do_mpi : # do it for MPI
        from psana.detector.Utils import get_hostname
        import psutil

        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        hostname = get_hostname()
        cpu_num = psutil.Process().cpu_num()
        s_rsc = 'rank:%03d/%03d cpu:%03d' % (rank, size, cpu_num)
        print(s_rsc)

    DTYPE_RAWD = np.uint16
    DTYPE_PEDS = np.float32
    DTYPE_STAT = np.uint64
    DTYPE_GAIN = np.float32
    DTYPE_MASK = np.uint8
    DTYPE_BKGD = np.float32
    DTYPE_CMOD = np.float64
    DTYPE_REST = np.float32
    M14 = 0x3fff
    B14 = 0x4000

    arrts = None
    sh = (16, 352, 384)

    tgen0 = time()

    rawa = np.empty((nloops,) + sh, dtype=DTYPE_RAWD)
    for i in range(nloops):
        rawa[i,:] = ag.random_standard(shape=sh, mu=1000, sigma=100, dtype=DTYPE_RAWD)

    #raw  = ag.random_standard(shape=sh, mu=1000, sigma=100, dtype=DTYPE_RAWD)
    mask = ag.random_0or1(shape=sh, p1=0.90, dtype=DTYPE_MASK)
    peds = ag.random_standard(shape=sh, mu=1000, sigma=100, dtype=DTYPE_PEDS)
    gain = ag.random_standard(shape=sh, mu=5, sigma=1,      dtype=DTYPE_GAIN)
    arrf = np.empty(sh, dtype=DTYPE_REST)
    #arrf = np.array(raw & M14, dtype=DTYPE_REST)

    if rank == rank_test:
        print(ue.info_ndarr(rawa,  'rawa:'))
        print(ue.info_ndarr(peds,  'peds:'))
        print(ue.info_ndarr(gain,  'gain:'))
        print(ue.info_ndarr(mask,  'mask:'))

    print('Time to generate all arrays, sec %.3f' % (time()-tgen0))

    tloop_sec = time()

    for i in range(nloops):

        t0 = time()
        times = None
        raw = rawa[i,:]

        #if i == 0:
        #v = np.bitwise_or(v,0x00010000) # set required bits to 1
        #v = np.bitwise_and(v,0x0001FFFF) # set required bits to 0
        #arr1 = np.ones_like(arrf) #, dtype=DTYPE_REST)
        t1 = time()

        if CALIBMET == SIM0:
            t2 = time()
            #arrf = (raw & M14) - peds
            #print(ue.info_ndarr(arrf,  'arrf:'))
            t3 = time()
            #arrf *= gain
            t4 = time()
            #arrf *= mask
            t5 = time()
            #arrf = np.array(raw & M14, dtype=DTYPE_REST)
            t6 = time()
            #arrf = ((raw & M14) - peds) * gain
            t7 = time()
            #arrf = np.array(raw & M14, dtype=DTYPE_REST)
            t8 = time()
            arrf = ((raw & M14) - peds) * gain * mask
            t9 = time()
            times = t0, t1, t2, t3, t4, t5, t6, t7, t8, t9

        elif CALIBMET == SIM1:
            t2 = time()
            arrf = ((raw & M14) - peds)*gain
            t3 = time()
            arrf = np.select((mask>0,), (arrf,), default=0) #.astype(DTYPE_REST))
            t4 = time()
            times = t0, t1, t2, t3, t4

        elif CALIBMET == SIM2:
            t2 = time()
            np.subtract(raw & M14, peds, out=arrf)
            t3 = time()
            np.multiply(arrf, gain, out=arrf)
            t4 = time()
            np.multiply(arrf, mask, out=arrf)
            t5 = time()
            times = t0, t1, t2, t3, t4, t5

        elif CALIBMET == SIM3:
            t2 = time()
            arrf = vf(raw.ravel(), peds.ravel(), gain.ravel(), mask.ravel())
            t3 = time()
            times = t0, t1, t2, t3

        elif CALIBMET == SIM4:
            t2 = time()
            arrf = uf(raw.ravel(), peds.ravel(), gain.ravel(), mask.ravel())
            t3 = time()
            times = t0, t1, t2, t3

        elif CALIBMET == SIM5:
            t2 = time()
            dt_us_cpp, dt_us_cyt = ud.calib_std(raw, peds, gain, mask, M14, arrf)
            t3 = time()
            times = t0, t1, t2, t3
            if i%100 ==0 and rank == rank_test:
                dt_us_py = (t3 - t2)*1e6
                dt_us_loop = (t3 - tloop_sec)*1e6
                print('dt_us_cpp: %.1f, dt_us_cy: %.1f, dt_us_py: %.1f, dt_loop_us: %.1f' % (dt_us_cpp, dt_us_cyt, dt_us_py, dt_us_loop))

        else:
            print('TEST %s IS NOT IMPLEMENTED!' % str(CALIBMET))

        if i%100 ==0 and rank == rank_test:
            print('%04d: dt, msec %.4f' % (i, (times[-1]-t0)*1000.))

        if arrts is None:
            arrts = np.zeros((nloops, len(times)), dtype=float)

        arrts[i] = times
        tloop_sec = time() # reset loop time measurement, omit arrts operetions

    # at the end of the event loop
    if rank == rank_test:
        dt0 = arrts[:,-1]-arrts[:,0] # total time on loop
        arrts[:,1:] -= arrts[:,0:-1] # evaluate dt for each step
        arrts[:,0]   = dt0           # set elt 0: total time per event
        arrts[:,:]  *= 1000          # convert sec > msec

        print(title(arrts))
        print_summary(arrts, show_arrts=True, cmt='', fname='summary.txt')
        print(title(arrts))


def test_simulation_nloops(*args, **kwargs):
    """ douple loop for nloops and events to increase execution time
    """
    import psana.pyalgos.generic.NDArrGenerators as ag
    import psana.pycalgos.utilsdetector as ud
    #import psana.pscalib.calib.CalibConstants import as cc

    #CALIBMET  = kwargs.get('CALIBMET', SIM0)
    nloops    = kwargs.get('nloops', 100)
    events    = kwargs.get('events', 100)
    rank_test = kwargs.get('rank_test', 0)
    rank = 0

    if True:
        #import multiprocessing as mp
        #rank = mp.cpu_count()
        libc = CDLL("libc.so.6")
        rank = libc.sched_getcpu()

    if False:
        #if do_mpi : # do it for MPI
        from psana.detector.Utils import get_hostname
        import psutil

        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        hostname = get_hostname()
        cpu_num = psutil.Process().cpu_num()
        s_rsc = 'rank:%03d/%03d cpu:%03d' % (rank, size, cpu_num)
        print(s_rsc)

    s_rank = 'cpu:%03d' % rank

    DTYPE_RAWD = np.uint16
    DTYPE_MASK = np.uint8
    DTYPE_PEDS = np.float32
    DTYPE_GAIN = np.float32
    DTYPE_REST = np.float32
    DTYPE_BKGD = np.float32
    DTYPE_CMOD = np.float64
    DTYPE_STAT = np.uint64
    M14 = 0x3fff
    B14 = 0x4000

    arrts = None
    sh = (16, 352, 384)

    tgen0 = time()

    rawa = np.empty((events,) + sh, dtype=DTYPE_RAWD)
    for i in range(events):
        rawa[i,:] = ag.random_standard(shape=sh, mu=1000, sigma=100, dtype=DTYPE_RAWD)

    mask = ag.random_0or1(shape=sh, p1=0.90, dtype=DTYPE_MASK)
    peds = ag.random_standard(shape=sh, mu=1000, sigma=100, dtype=DTYPE_PEDS)
    gain = ag.random_standard(shape=sh, mu=5, sigma=1,      dtype=DTYPE_GAIN)
    #arrf = np.empty(sh, dtype=DTYPE_REST)

    if rank == rank_test:
        print(ue.info_ndarr(rawa,  'rawa:'))
        print(ue.info_ndarr(peds,  'peds:'))
        print(ue.info_ndarr(gain,  'gain:'))
        print(ue.info_ndarr(mask,  'mask:'))

    print('%s: time to generate all arrays %.3f sec' % (s_rank, time()-tgen0))

    arrts = np.zeros(nloops, dtype=float)

    for n in range(nloops):
      t0 = time()
      for i in range(events):
        arrf = ((rawa[i,:] & M14) - peds) * gain * mask
      arrts[n] = dt_sec = time()-t0

      #if i%10 ==0 and rank == rank_test:
      #  print('%s loop %03d: dt, msec %.4f nloops: %d events: %d' % (s_rank, n, dt_sec*1000., nloops, events))

    arrts *= 1000/events
    #print(s_rank, 'time per event in loops:', arrts)
    print('%s: median over loops per event time: %.3f ms nloops: %d events: %d' % (s_rank, np.median(arrts), nloops, events))


def test_mpi_for_data(*args, **kwargs):
    print('test_mpi_for_data args:', args)
    print('test_mpi_for_data kwargs:', kwargs)

def argument_parser():
    from argparse import ArgumentParser
    d_tname = '0'
    d_dskwargs = 'exp=rixc00121,run=140,dir=/sdf/data/lcls/drpsrcf/ffb/rix/rixc00121/xtc'  # None
    d_detname  = 'archon' # None
    d_loglevel = 'INFO' # 'DEBUG'
    h_tname    = 'test name, usually numeric number, default = %s' % d_tname
    h_dskwargs = '(str) dataset kwargs for DataSource(**kwargs), default = %s' % d_dskwargs
    h_detname  = 'detector name, default = %s' % d_detname
    h_loglevel = 'logging level, one of %s, default = %s' % (', '.join(tuple(logging._nameToLevel.keys())), d_loglevel)
    parser = ArgumentParser(description='%s is a bunch of tests for annual issues' % SCRNAME, usage=usage())
    parser.add_argument('tname',            default=d_tname,    type=str, help=h_tname)
    parser.add_argument('-k', '--dskwargs', default=d_dskwargs, type=str, help=h_dskwargs)
    parser.add_argument('-d', '--detname',  default=d_detname,  type=str, help=h_detname)
    parser.add_argument('-L', '--loglevel', default=d_loglevel, type=str, help=h_loglevel)
    return parser

def usage():
    import inspect
    return '\n  %s <tname>\n' % sys.argv[0].split('/')[-1]\
    + '\n'.join([s for s in inspect.getsource(selector).split('\n') if "tname ==" in s or "tnum in" in s])

def selector():
    if len(sys.argv) < 2:
        print(usage())
        sys.exit('EXIT due to MISSING PARAMETERS')

    parser = argument_parser()
    args = parser.parse_args()
    #args = Namespace(tname='1', dskwargs='exp=rixc00121,run=140,dir=/sdf/data/lcls/drpsrcf/ffb/rix/rixc00121/xtc', detname='archon', loglevel='INFO')
    #sys.exit('TEST EXIT')

    STRLOGLEV = args.loglevel
    INTLOGLEV = logging._nameToLevel[STRLOGLEV]
    logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d %(filename)s: %(message)s', level=INTLOGLEV)

    tname = args.tname  # sys.argv[1] if len(sys.argv)>1 else '0'
    tnum = int(tname)

    if   tname ==  '0': test_event_loop(CALIBMET=CALIB_STD)
    elif tname ==  '1': test_event_loop(CALIBMET=CALIB_LOCAL)
    elif tname ==  '2': test_event_loop(CALIBMET=CALIB_LOCAL_V2, cmpars=None) #, do_image=True, cmpars=(7,7,200,10))
    elif tname ==  '3': test_event_loop(CALIBMET=CALIB_LOCAL_V3, cmpars=None)
    elif tname == '50': test_simulation_nloops()
    elif tnum in SIMS : test_simulation(CALIBMET=tnum)
    elif tname == '98': test_ArrayIterator()
    elif tname == '99': parse_summary(fname='summary-uedcom103-r0095-ncpu-064.txt')
#    elif tname ==  'x': test_mpi_for_data(SHOW_FIGS=False,SAVE_FIGS=False,cmt='16p-v3', CALIBMET=CALIB_LOCAL_V2, loop_segs=False, cmpars=(7,7,200,10))
    else:
        print(usage())
        exit('\nTEST "%s" IS NOT IMPLEMENTED'%tname)
    sys.exit(0)


if __name__ == "__main__":
    selector()

# EOF
