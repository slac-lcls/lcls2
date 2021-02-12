"""
:py:class:`UtilsAreaCalib` dark processing algorithms for generic area detector 
===============================================================================

Usage::

    from psana.detector.UtilsAreaCalib import *
    #OR
    import psana.detector.UtilsAreaCalib as uac

    uac.

This software was developed for the LCLS project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2021-02-10 by Mikhail Dubrovin
"""

import logging
logger = logging.getLogger(__name__)
import sys
import numpy as np
from psana.detector.Utils import info_dict, info_namespace, info_command_line
from psana.detector.UtilsEpix10ka import seconds, timestamp_run
from psana.pyalgos.generic.Utils import str_tstamp
from psana.pyalgos.generic.NDArrUtils import info_ndarr
from psana import DataSource


def datasource_arguments(args):
    """
    Parameters
    ----------
    args.fname: str - xtc2 file name
    args.expname: str - experiment name
    args.runs: int run number or str with comma-separated run numbers
    args.detname: str - detector name
    args.evtmax: int - maximal number of events to process

    Returns
    -------
    dict of DataSource keyword arguments
    """
    assert args.fname is not None\
           or None not in (args.expname, args.runs), 'experiment name and run or xtc2 file name need to be specified for DataSource'
    kwa = {'files':args.fname,} if args.fname is not None else\
          {'exp':args.expname,'run':[int(v) for v in args.runs.split(',')]}
    if args.evtmax: kwa['max_events'] = args.evtmax
    return kwa


def info_run_dsparms_det_classes_v1(run, cmt='run.dsparms.det_classes:', sep='\n '):
    return cmt + sep.join(['%8s : %s' % (str(k),str(v)) for k,v in run.dsparms.det_classes.items()])


def info_run_dsparms_det_classes(run, cmt='run.dsparms.det_classes:\n ', sep='\n '):
    return cmt + info_dict(run.dsparms.det_classes, fmt='%10s : %s', sep=sep)


def info_run(run, cmt='run info:', sep='\n    ', verb=0o377):
    t_sec = seconds(run.timestamp)
    ts_run = timestamp_run(run, fmt='%Y-%m-%dT%H:%M:%S')
    #ts_run = str_tstamp(fmt='%Y-%m-%dT%H:%M:%S', time_sec=t_sec)
    return cmt\
      +   'run.runnum    : %d' % (run.runnum)\
      + '%srun.expt      : %s' % (sep, run.expt)\
      + '%srun.detnames  : %s' % (sep, ' '.join(run.detnames))\
      + '%srun.timestamp : %s -> %s' % (sep, run.timestamp, ts_run)\
      +('%srun.id        : %s' % (sep, run.id) if verb & 1 else '')\
      +('%s%s' % (sep, info_run_dsparms_det_classes(run, cmt='run.dsparms.det_classes:', sep=sep+'   ')) if verb & 2 else '')


def info_uniqueid(det, cmt='det.raw._uniqueid.split("_"):', sep='\n '):
    return cmt + sep.join(det.raw._uniqueid.split('_'))


def info_detector(det, cmt='detector info:', sep='\n    '):
    return cmt\
        +  'det.raw._det_name   : %s' % (det.raw._det_name)\
        +'%sdet.raw._dettype    : %s' % (sep, det.raw._dettype)\
        +'%s_sorted_segment_ids : %s' % (sep, str(det.raw._sorted_segment_ids))\
        +'%s%s' % (sep, info_uniqueid(det, cmt='det.raw._uniqueid.split("_"):%s     '%sep, sep=sep+'     '))\
        +'%sdet methods vbisible: %s' % (sep, ' '.join([v for v in dir(det) if v[0]!='_']))\
        +'%s            hidden  : %s' % (sep, ' '.join([v for v in dir(det) if (v[0]=='_' and v[1]!='_')]))\
        +'%sdet.raw._calibconst.keys(): %s' % (sep, ', '.join(det.raw._calibconst.keys()))


def run_step_event_loop(pars): #*args, **kwa):

  print('command line: %s' % info_command_line())
  print('input parameters:\n%s' % info_namespace(pars)) #, fmt='%s: %s', sep=', '))
  #print('input parameters: %s' % info_dict(opts, fmt='%s: %s', sep=', '))

  ds = DataSource(**datasource_arguments(pars))
  #ds = DataSource(exp='tmoc00118', run=123, max_events=100)
  #ds = DataSource(exp=pars.expname, run=pars.runs, max_events=pars.evtmax)

  block = None
  rows, cols = None, None
  nrecs = pars.nrecs
  nrecs2 = nrecs-2
  iblk = -1

  for irun,run in enumerate(ds.runs()):
    print('\n==== %02d run: %d exp: %s' % (irun, run.runnum, run.expt))
    print(info_run(run, cmt='run info:\n    ', sep='\n    ', verb=3))

    det = run.Detector(pars.detname)
    print('\n  created %s detector object' % pars.detname)
    print(info_detector(det, cmt='  detector info:\n      ', sep='\n      '))

    for istep,step in enumerate(run.steps()):
      print('\nStep %1d' % istep)

      for ievt,evt in enumerate(step.events()):

        print('Event %04d' % (ievt), end='')
        raw  = det.raw.raw(evt)

        if raw is None:
            logger.info('raw is None')
            continue
        rows, cols = raw.shape
        if block is None:
           block=np.zeros((nrecs, rows, cols),dtype=raw.dtype)
           logger.info(info_ndarr(block,'Array for accumulation of raw data '))
           segs = det.raw.segments(evt)
           print(info_ndarr(segs, 'segsments '))

        iblk += 1
        block[iblk,:] = raw

        if iblk > nrecs2:
            print('\nBlock limit is reached, iblk=%d' % iblk)

            exit('TEST EXIT')

        print(info_ndarr(raw,  ' raw '), end='\r')


if __name__ == "__main__":

    SCRNAME = sys.argv[0].rsplit('/')[-1]
  
    class TestParameters:
        fname      = None
        expname    = 'tmoc00118'
        runs       = '123' # '12,14'
        evtmax     = 2000
        detname    = 'tmoopal'
        nrecs      = 1000
  
    print(80*'_')  
    logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d: %(message)s', level=logging.INFO)
  
    run_step_event_loop(TestParameters())
  
    sys.exit('End of %s' % sys.argv[0])

# EOF
