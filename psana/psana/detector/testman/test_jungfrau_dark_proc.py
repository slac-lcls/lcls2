#!/usr/bin/env python

"""
:py:class:`UtilsJungfrauCalibMPI`
===================================
The same as UtilsJungfrauCalib, but uses MPI in the event loop

Usage::

mpirun -n 5 test_jungfrau_dark_proc.py -k exp=mfx100848724,run=49 -d jungfrau
mpirun -n 5 test_jungfrau_dark_proc.py -k exp=mfx100848724,run=49 -d jungfrau --fix

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2026-03-04 by Mikhail Dubrovin
"""
import logging

import psana.detector.UtilsJungfrauCalib as ujc
os, sys, time, logging, uc, ups, json = ujc.os, ujc.sys, ujc.time, ujc.logging, ujc.uc, ujc.ups, ujc.json

DIC_GAIN_MODE, DIC_IND_TO_GAIN_MODE, DarkProcJungfrau, init_repoman_and_logger, open_DataSource =\
    ujc.DIC_GAIN_MODE, ujc.DIC_IND_TO_GAIN_MODE, ujc.DarkProcJungfrau, ujc.init_repoman_and_logger, ujc.open_DataSource

logger = logging.getLogger(__name__)

os.environ['PS_EB_NODES']='1'
os.environ['PS_SRV_NODES']='1'

import psutil
cpu_num = psutil.Process().cpu_num()

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
use_mpi = size > 1
is_rank0 = rank==0
is_rank_sel = rank==(size-2) # rank==3

from psana.detector.Utils import get_hostname
hostname = get_hostname()

s_rsch = 'rank:%03d/%03d-cpu:%03d-%s' % (rank, size, cpu_num, hostname)
if is_rank0: print('%s sys.argv: %s' % (s_rsch, sys.argv))


def filter_callback(run):
    """Event filter using small data. Parameters passed via global storage"""
    nrecs = 100
    evskip = 0
    stepnum = None
    stepmax = 3
    isset = True

    logger.info('== filter_callback first call for %s' % s_rsch)
    for istep, step in enumerate(run.steps()):
        logger.info('== filter_callback begin step:%d for %s' % (istep, s_rsch))
        cond1 = True if stepmax is None else istep < stepmax
        cond2 = True if stepnum is None else istep == stepnum
        if cond1 and cond2:
            logger.info('== filter_callback stepnum:%s stepmax:%s evskip:%d nrecs:%d - begin event loop for step:%d for %s)' %\
                        (str(stepnum), str(stepmax), evskip, nrecs, istep, s_rsch))
            for ievt, evt in enumerate(step.events()):
                if  ievt > evskip-1\
                and ievt < nrecs:
                    #print('  = filter_callback yield for step:%d evt:%03d' % (istep, ievt))
                    yield evt


def jungfrau_dark_proc_mpi(parser):
    """jungfrau dark data processing for single (of 3) gain mode.
    """
    t0_sec = time()
    tdt = t0_sec

    args = parser.parse_args() # namespae of parameters
    kwargs = vars(args) # dict of parameters

    INTLOGLEV = logging._nameToLevel['INFO']
    logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d %(filename)s: %(message)s', level=INTLOGLEV)

    detname = args.detname

    kwargs['batch_size'] = 1 # this batch_size parameter may need to be tweaked
    kwargs['info_xtc_files'] = is_rank0
    kwargs['smd_callback'] = filter_callback
    #kwargs['max_events'] = kwargs.get('events', 3000)
    ds, dskwargs = open_DataSource(**kwargs)
    kwargs['ds'] = ds
    kwargs['dskwargs'] = dskwargs
    logger.debug('on %s open DataSource as: %s' % (s_rsch, str(ds)))

    smd = ds.smalldata()

    ievt = 0
    nevsel = 0
    ss = ''
    orun = next(ds.runs())

    #### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if not args.fix: kwargs['orun'] = orun #### !!!!!!!!!!!!!!!!!!!!!!!
    #### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    logger.info('%s %s begin run %s %s' % (s_rsch, 20*'=', str(orun.runnum), 20*'='))

    nevrun = 0
    nnones = 0
    odet = None
    is_rank_sum = False

    for istep, step in enumerate(orun.steps()):
        logger.info('%s ==== begin step %d' % (s_rsch, istep))
        t0_sec_step = time()

        if odet is None:
           odet = orun.Detector(detname, **kwargs)
           kwargs['odet'] = odet

        for ievt, evt in enumerate(step.events()):
            nevrun += 1

        ss = '%s runnum:%d end of step %d events run/step/selected: %4d/%4d/%4d  step time: %.3f sec'%\
             (s_rsch, orun.runnum, istep, nevrun, ievt+1, nevsel, time() - t0_sec_step)
        logger.info(ss)

        logger.info('%s smd.summary: %s' % (s_rsch, str(smd.summary)))

        #del(dpo)
        dpo=None

    logger.info('SMD.DONE in %s' % s_rsch)
    smd.done()

    if is_rank_sum:
        logger.info('SUM %s total consumed time %.3f sec' % (s_rsch, time()-t0_sec))

    logger.info('exit %s' % s_rsch)


t0_sec_tot = time()

from psana.detector.dir_root import DIR_REPO_JUNGFRAU
from psana.detector.UtilsLogging import logging, STR_LEVEL_NAMES
logger = logging.getLogger(__name__)

SCRNAME = sys.argv[0].rsplit('/')[-1]

USAGE = 'Usage:'\
      + '\n  datinfo -k exp=mfxdaq23,run=7,dir=/sdf/data/lcls/drpsrcf/ffb/mfx/mfxdaq23/xtc/ -d jungfrau # test data'\
      + '\n  mpirun -n 5 python test_jungfrau_dark_proc.py -k exp=mfx100848724,run=49 -d jungfrau'\
      + '\n\n  Try: %s -h' % SCRNAME


def argument_parser():
    from argparse import ArgumentParser

    d_dskwargs= 'exp=mfxdaq23,run=7,dir=/sdf/data/lcls/drpsrcf/ffb/MFX/mfxdaq23/xtc' # None
    d_detname = 'jungfrau' #  None
    d_fix     = False

    h_dskwargs= 'string of comma-separated (no spaces) simple parameters for DataSource(**kwargs),'\
                ' ex: exp=<expname>,run=<runs>,dir=<xtc-dir>, ...,'\
                ' or <fname.xtc> or files=<fname.xtc>'\
                ' or pythonic dict of generic kwargs, e.g.:'\
                ' \"{\'exp\':\'tmoc00318\', \'run\':[10,11,12], \'dir\':\'/a/b/c/xtc\'}\", default = %s' % d_dskwargs
    h_detname = 'detector name, default = %s' % d_detname
    h_fix  = 'fix hanging at exit issue, default = %s' % d_fix

    parser = ArgumentParser(usage=USAGE, description='Proceses dark run xtc data for epix10ka')
    parser.add_argument('-k', '--dskwargs',default=d_dskwargs,   type=str,   help=h_dskwargs)
    parser.add_argument('-d', '--detname', default=d_detname,    type=str,   help=h_detname)
    parser.add_argument('--fix', action='store_true', help=h_fix)
    return parser


def do_main():
    from time import time

    parser = argument_parser()
    args = parser.parse_args()
    kwa = vars(args)

    if len(sys.argv)<3: sys.exit('\n%s\n\nEXIT DUE TO MISSING ARGUMENTS\n' % USAGE)
    assert args.dskwargs is not None, 'WARNING: option "-k <DataSource-kwargs>" MUST be specified.'
    assert args.detname  is not None, 'WARNING: option "-d <detector-name>" MUST be specified.'

    t0_sec = time()

    print('begin: %s' % s_rsch)
    jungfrau_dark_proc_mpi(parser)
    print('exit: %s SCRIPT %s time %.3f sec and TOTAL TIME (with imports and parser) %.3f sec' % (s_rsch, SCRNAME, time() - t0_sec, time() - t0_sec_tot))


if __name__ == "__main__":
    do_main()
    sys.exit(0)

# EOF

