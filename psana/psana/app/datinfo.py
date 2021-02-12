#!/usr/bin/env python
#----
import os
import sys
from time import time    
import json

from psana.detector.Utils import info_dict, info_command_line, info_namespace
from psana.pyalgos.generic.NDArrUtils import info_ndarr
import psana.detector.UtilsEpix10ka as ue
from psana.detector.UtilsAreaCalib import datasource_arguments, info_detector, info_run
from psana import DataSource

import logging
logger = logging.getLogger(__name__)
DICT_NAME_TO_LEVEL = logging._nameToLevel
STR_LEVEL_NAMES = ', '.join(DICT_NAME_TO_LEVEL.keys())

#SCRNAME = sys.argv[0].rsplit('/')[-1]
SCRNAME = os.path.basename(sys.argv[0])

USAGE = '\n %s <detector> -e <experiment> -r <run-number(s)> [kwargs]' % SCRNAME\
      + '\nCOMMAND EXAMPLES:'\
      + '\n  %s epixquad -e ueddaq02 -r 27 <--- WORKS' % SCRNAME\
      + '\n  %s epixquad -e ueddaq02 -r 30-82 <--- DO NOT WORK' % SCRNAME\
      + '\n  %s epixquad -e ueddaq02 -r 83 <--- WORKS dark' % SCRNAME\
      + '\n  %s epixquad -e ueddaq02 -r 84 <--- PARTLY WORKS charge injection' % SCRNAME\
      + '\n  %s epixquad -e ueddaq02 -r 27 -L DEBUG' % SCRNAME\
      + '\n  %s epixquad -f /cds/data/psdm/ued/ueddaq02/xtc/ueddaq02-r0065-s001-c000.xtc2' % SCRNAME\
      + '\n  %s epixquad -f /cds/data/psdm/ued/ueddaq02/xtc/ueddaq02-r0086-s001-c000.xtc2' % SCRNAME\
      + '\n  %s tmoopal -e tmoc00118 -r 123' % SCRNAME\
      + '\nHELP: %s -h' % SCRNAME
#----

def ds_run_det(args):

    if not ('d' in args.typeinfo.lower()):
        print('detector information is not requested by -td option - skip it')
        return

    ds_kwa = datasource_arguments(args)
    print('DataSource kwargs: %s' % info_dict(ds_kwa, fmt='%s: %s', separator=', '))
    ds = DataSource(**ds_kwa)
    run = next(ds.runs())
    det = run.Detector(args.detname)

    expname = run.expt if run.expt is not None else args.expname # 'mfxc00318'

    print('fname:', args.fname)
    #print('expname:', expname)
    #print('runnum :', run.runnum)
    #print('run.timestamp :', run.timestamp)

    print(info_run(run, cmt='run info\n    ', sep='\n    '))
    print(info_detector(det, cmt='detector info\n    ', sep='\n    '))


def selected_record(nrec):
    return nrec<5\
       or (nrec<50 and not nrec%10)\
       or (not nrec%100)
       #or (nrec<500 and not nrec%100)\
       #or (not nrec%1000)


def info_det_evt(det, evt, ievt):
    return '  Event %05d %s '% (ievt, info_ndarr(det.raw.raw(evt), 'raw '))
 

def loop_run_step_evt(args):
  """Data access example for confluence
     run, step, event loops
  """
  typeinfo = args.typeinfo.lower()
  do_loopruns  = 'r' in typeinfo
  do_loopevts  = 'e' in typeinfo
  do_loopsteps = 's' in typeinfo

  from psana.pyalgos.generic.NDArrUtils import info_ndarr
  #from psana import DataSource
  #ds = DataSource(exp=args.expt, run=args.run, dir=f'/cds/data/psdm/{args.expt[:3]}/{args.expt}/xtc', max_events=1000)

  ds = DataSource(**datasource_arguments(args))

  if do_loopruns:
    for irun,run in enumerate(ds.runs()):
      print('\n==== %02d run: %d exp: %s detnames: %s' % (irun, run.runnum, run.expt, ','.join(run.detnames)))
    
      if not do_loopsteps: continue
      print('%s detector object' % args.detname)
      det = run.Detector(args.detname)
      is_epix10ka = 'epix' in det.raw._uniqueid
      try:    step_docstring = run.Detector('step_docstring')
      except: step_docstring = None
      print('step_docstring detector object is %s' % ('missing' if step_docstring is None else 'created'))

      dcfg = ue.config_object_epix10ka(det) if is_epix10ka else None

      for istep,step in enumerate(run.steps()):
        print('\nStep %02d' % istep, end='')

        metadic = None if step_docstring is None else json.loads(step_docstring(step))
        print('  metadata: %s' % str(metadic))

        if not do_loopevts: continue
        ievt,evt,segs = None,None,None
        for ievt,evt in enumerate(step.events()):
          #if ievt>args.evtmax: exit('exit by number of events limit %d' % args.evtmax)
          if not selected_record(ievt): continue
          if segs is None:
             segs = det.raw.segments(evt)
             print('  Event %05d %s'% (ievt, info_ndarr(segs,'segments ')))
             #print('gain mode statistics:' + ue.info_pixel_gain_mode_statistics(gmaps))
             if dcfg is not None:
               s = '    gain mode fractions for: FH       FM       FL'\
                   '       AHL-H    AML-M    AHL-L    AML-L\n%s' % (29*' ')
               print(ue.info_pixel_gain_mode_fractions(dcfg, data=det.raw.raw(evt), msg=s))

          print(info_det_evt(det, evt, ievt), end='\r')
        print(info_det_evt(det, evt, ievt), end='\n')


def do_main():

    t0_sec = time()
    #print('len(sys.argv):', len(sys.argv))
    if len(sys.argv)<2:
        print('Usage:%s\n' % USAGE)
        exit('EXIT - MISSING ARGUMENT(S)')

    parser = argument_parser()
    args = parser.parse_args()
    opts = vars(args)
    #?????defs = vars(parser.parse_args([])) # dict of defaults only

    #logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s', datefmt='%H:%M:%S', level=logging.DEBUG)
    fmt = '[%(levelname).1s] %(name)s %(message)s' if args.logmode=='DEBUG' else '[%(levelname).1s] %(message)s'
    #logging.basicConfig(filename='log.txt', filemode='w', format=fmt, level=DICT_NAME_TO_LEVEL[args.logmode])
    logging.basicConfig(format=fmt, level=DICT_NAME_TO_LEVEL[args.logmode])

    print('command line: %s' % info_command_line())
    print('input parameters: %s' % info_dict(opts, fmt='%s: %s', separator=', '))
    #pedestals_calibration(*args, **opts)
    #pedestals_calibration(**opts)
    ds_run_det(args)

    loop_run_step_evt(args)

    logger.info('DONE, consumed time %.3f sec' % (time() - t0_sec))


def argument_parser():
    from argparse import ArgumentParser

    d_detname = 'epixquad'
    d_expname = None # 'ueddaq02'
    d_runs    = None # '66' # 1021 or 1021,1022-1025
    d_fname   = None # '/cds/data/psdm/ued/ueddaq02/xtc/ueddaq02-r0027-s000-c000.xtc2'
    d_evtmax  = 0 # maximal number of events
    d_dirxtc  = None # '/cds/data/psdm/ued/ueddaq02/xtc'
    d_logmode = 'INFO'
    d_typeinfo= 'DRSE'

    h_detname = 'detector name, e.g. %s' % d_detname
    h_fname   = 'input xtc file name, default = %s' % d_fname
    h_expname = 'experiment name, e.g. %s' % d_expname
    h_runs    = 'run number or list of runs e.g. 12,14,18 or 12, default = %s' % str(d_runs)
    h_evtmax  = 'number of events to print, default = %s' % str(d_evtmax)
    h_dirxtc  = 'non-default xtc directory, default = %s' % d_dirxtc
    h_logmode = 'logging mode, one of %s, default = %s' % (STR_LEVEL_NAMES, d_logmode)
    h_typeinfo= 'type of information for output D-detector, R-run-loop, S-step-loop, E-event-loop, default = %s' % d_typeinfo

    parser = ArgumentParser(description='Print info about experiment detector and run')
    parser.add_argument('detname', type=str, help=h_detname)
    parser.add_argument('-e', '--expname', type=str, help=h_expname)
    parser.add_argument('-r', '--runs',    type=str, help=h_runs)
    parser.add_argument('-f', '--fname', default=d_fname, type=str, help=h_fname)
    parser.add_argument('-n', '--evtmax', default=d_evtmax, type=int, help=h_evtmax)
    parser.add_argument('-x', '--dirxtc', default=d_dirxtc, type=str, help=h_dirxtc)
    parser.add_argument('-L', '--logmode', default=d_logmode, type=str, help=h_logmode)
    parser.add_argument('-t', '--typeinfo', default=d_typeinfo, type=str, help=h_typeinfo)

    return parser

#----

if __name__ == "__main__":
    do_main()
    exit('End of %s'%SCRNAME)

#---- EOF
