#!/usr/bin/env python

#import os
import sys
from time import time
import json

from psana2.detector.Utils import info_dict, info_command_line, info_namespace, info_parser_arguments, str_tstamp
from psana2.pyalgos.generic.NDArrUtils import info_ndarr
import psana2.detector.UtilsEpix10ka as ue
from psana2.detector.utils_psana import datasource_kwargs_from_string, info_run, info_detnames_for_dskwargs, info_detector, seconds, timestamp_run
from psana2 import DataSource

import logging
logger = logging.getLogger(__name__)
DICT_NAME_TO_LEVEL = logging._nameToLevel
STR_LEVEL_NAMES = ', '.join(DICT_NAME_TO_LEVEL.keys())

SCRNAME = sys.argv[0].rsplit('/')[-1]
#SCRNAME = os.path.basename(sys.argv[0])

USAGE = '\n  %s -d <detector> -k <datasource-kwargs> [kwargs]' % SCRNAME\
      + '\nCOMMAND EXAMPLES:'\
      + '\n  %s -d epixquad -k exp=ueddaq02,run=27 -td -L DEBUG' % SCRNAME\
      + '\n  %s -d epixquad -k exp=ueddaq02,run=30 <--- DOES NOT WORK - missconfigured' % SCRNAME\
      + '\n  %s -d epixquad -k exp=ueddaq02,run=83 <--- dark' % SCRNAME\
      + '\n  %s -d epixquad -k exp=ueddaq02,run=84 <--- PARTLY WORKS charge injection' % SCRNAME\
      + '\n  %s -d epixquad -k /cds/data/psdm/ued/ueddaq02/xtc/ueddaq02-r0065-s001-c000.xtc2' % SCRNAME\
      + '\n  %s -d epixquad -k /cds/data/psdm/ued/ueddaq02/xtc/ueddaq02-r0086-s001-c000.xtc2' % SCRNAME\
      + '\n  %s -d tmoopal  -k exp=tmoc00118,run=123 -td' % SCRNAME\
      + '\n  %s -k exp=tmoc00318,run=8 -d epix100hw' % SCRNAME\
      + '\n  %s -k /cds/data/psdm/prj/public01/xtc/rixx45619-r0121-s001-c000.xtc2 -d epixhr' % SCRNAME\
      + '\n  %s -k /cds/data/psdm/prj/public01/xtc/tmoc00318-r0010-s000-c000.xtc2 -d epix100' % SCRNAME\
      + '\n  %s -k /cds/data/psdm/prj/public01/xtc/tmoc00118-r0222-s006-c000.xtc2 -d tmo_atmopal' % SCRNAME\
      + '\n  %s -k /cds/data/psdm/prj/public01/xtc/uedcom103-r0007-s002-c000.xtc2 -d epixquad' % SCRNAME\
      + '\n  %s -k /cds/data/psdm/prj/public01/xtc/ueddaq02-r0569-s001-c000.xtc2  -d epixquad' % SCRNAME\
      + '\n  %s -k exp=tstx00417,run=317,dir=/reg/neh/operator/tstopr/data/drp/tst/tstx00417/xtc/ -d tst_epixm' % SCRNAME\
      + '\nHELP: %s -h' % SCRNAME

def ds_run_det(args):

    if not ('d' in args.typeinfo.lower()):
        print('detector information is not requested by -td option - skip it')
        return

    dskwargs = datasource_kwargs_from_string(args.dskwargs)  # datasource_arguments(args)
    print('DataSource kwargs:%s' % info_dict(dskwargs, fmt='%s: %s', sep=' '))
    try:
      ds = DataSource(**dskwargs)
    except:
      print('Can not open DataSource\nCheck if xtc2 file is available')
      sys.exit()
    run = next(ds.runs())
    det = None if args.detname is None else run.Detector(args.detname)

    print('args.detname:%s' % str(args.detname))
    print('DataSource members and methods\ndir(ds):', dir(ds))

    xtc_path = getattr(ds, 'xtc_path', None)
    print('ds.xtc_path:', str(xtc_path))
    if xtc_path is not None:
      print('ds.n_files:', str(ds.n_files))
      print('ds.xtc_files:\n ', '\n  '.join(ds.xtc_files))
      print('ds.xtc_ext:', str(ds.xtc_ext) if hasattr(ds,'xtc_ext') else 'N/A')
      print('ds.smd_files:\n ', '\n  '.join(ds.smd_files))
    print('ds.shmem:', str(ds.shmem))
    print('ds.smalldata_kwargs:', str(ds.smalldata_kwargs))
    print('ds.timestamps:', str(ds.timestamps))
    print('ds.unique_user_rank:', str(ds.unique_user_rank()))
    print('ds.is_mpi:', str(ds.is_mpi()))
    print('ds.live:', str(ds.live))
    print('ds.destination:', str(ds.destination))

    #sys.exit('TEST EXIT')

    expname = run.expt if run.expt is not None else args.expname # 'mfxc00318'

    print('dskwargs:', args.dskwargs)
    #print('run.timestamp :', run.timestamp)

    print(info_run(run, cmt='run info\n    ', sep='\n    '))
    #print(info_detnames(run, cmt='\ncommand: '))
    print(info_detnames_for_dskwargs(args.dskwargs, cmt='\ncommand: '))

    if det is None:
        print('detector object is None for detname %s' % args.detname)
        sys.exit('EXIT')

    det_raw_attrs = dir(det.raw)
    print('\ndir(det.raw):', det_raw_attrs)

    print('det.raw._uniqueid       :', det.raw._uniqueid if '_uniqueid' in det_raw_attrs else 'MISSING')
    print('det.raw._fullname       :', det.raw._fullname() if '_fullname' in det_raw_attrs else 'MISSING')
    print('det.raw._segment_ids    :', det.raw._segment_ids() if '_segment_ids' in det_raw_attrs else 'MISSING')
    print('det.raw._segment_indices:', det.raw._segment_indices() if '_segment_indices' in det_raw_attrs else 'MISSING')

    if '_config_object' in det_raw_attrs:
      dcfg = det.raw._config_object()
      if dcfg is not None:
       for k,v in dcfg.items():
        trbit = getattr(v.config, 'trbit', None)  # v.config.trbit
        asicPixelConfig = getattr(v.config, 'asicPixelConfig', None)  # v.config.asicPixelConfig
        print('  seg:%s %s' % (str(k), info_ndarr(trbit, ' v.config.trbit for ASICs')))
        print('  seg:%s %s' % (str(k), info_ndarr(asicPixelConfig, ' v.config.asicPixelConfig')))
    else: print('det.raw._config_object  : MISSING')

    print(info_detector(det, cmt='detector info\n    ', sep='\n    '))
    print('det.raw._seg_geo.shape():', det.raw._seg_geo.shape() if det.raw._seg_geo is not None else '_seg_geo is None')


def selected_record(nrec):
    return nrec<5\
       or (nrec<50 and not nrec%10)\
       or (not nrec%100)


def info_det_evt(det, evt, ievt):
    return '  Event %05d    %s' % (ievt, ('detector is None'+80*' ' if det is None else info_ndarr(det.raw.raw(evt), 'raw ')))


def loop_run_step_evt(args):
  """Data access example for confluence
     run, step, event loops
  """
  typeinfo = args.typeinfo.lower()
  do_loopruns  = 'r' in typeinfo
  do_loopevts  = 'e' in typeinfo
  do_loopsteps = 's' in typeinfo

  from psana2.pyalgos.generic.NDArrUtils import info_ndarr
  #from psana2 import DataSource
  #ds = DataSource(exp=args.expt, run=args.run, dir=f'/cds/data/psdm/{args.expt[:3]}/{args.expt}/xtc', max_events=1000)

  dskwargs = datasource_kwargs_from_string(args.dskwargs)
  print('dskwargs', dskwargs)
  ds = DataSource(**dskwargs)

  if do_loopruns:
    for irun, run in enumerate(ds.runs()):
      print('\n==== %02d run: %d exp: %s detnames: %s' % (irun, run.runnum, run.expt, ','.join(run.detnames)))
      print('run.timestamp LCLS2 int: %d > epoch unix sec: %.6f > %s' % (run.timestamp, seconds(run.timestamp), timestamp_run(run)))
      if not do_loopsteps: continue
      print('%s detector object' % args.detname)
      det = None if args.detname is None else run.Detector(args.detname)

      is_epix10ka  = False if det is None else det.raw._dettype == 'epix10ka'
      is_epixhr2x2 = False if det is None else det.raw._dettype == 'epixhr2x2'
      is_epixm320  = False if det is None else det.raw._dettype == 'epixm320'

      try:    step_docstring = run.Detector('step_docstring')
      except: step_docstring = None
      print('step_docstring detector object is %s' % ('missing' if step_docstring is None else 'created'))
      print('det.raw._seg_geo.shape():', det.raw._seg_geo.shape() if det.raw._seg_geo is not None else '_seg_geo is None')

      timing = run.Detector('timing') if 'timing' in run.detnames else None
      if timing is not None: timing.raw._add_fields()
      tsec_old = 0
      pulseid_old = 0

      dcfg = det.raw._config_object() if '_config_object' in dir(det.raw) else None
      if dcfg is None: print('det.raw._config_object is MISSING')

      for istep, step in enumerate(run.steps()):
        print('\nStep %02d' % istep, end='')
        if is_epixm320:
          from psana2.detector.UtilsEpixm320Calib import gain_mode_name
          print(' gain mode name from config: %s' % gain_mode_name(det), end='')


        if step_docstring is not None:
          sds = step_docstring(step)
          #print('XXX step_docstring(step):', sds)
          try: sdsdict = json.loads(sds)
          except Exception as err:
            print('\nERROR FOR step_docstring: ', sds)
            logger.error('json.loads(step_docstring(step)) err: %s' % str(err))
            sdsdict = None

        metadic = None if step_docstring is None else sdsdict
        print('  metadata: %s' % str(metadic))

        if not do_loopevts: continue
        ievt, evt, segs = None, None, None
        for ievt, evt in enumerate(step.events()):
          #if ievt>args.evtmax: exit('exit by number of events limit %d' % args.evtmax)
          if not selected_record(ievt): continue
          if segs is None:
             segs = det.raw._segment_numbers if det is not None else None
             #tstamp = evt.timestamp   # like 4193682596073796843 relative to 1990-01-01
             tsec = seconds(evt.timestamp)
             tsec_diff = tsec-tsec_old
             tsec_old = tsec
             print('  Event %05d t=%.6fsec dt=%.6fsec/record %s '%\
                   (ievt, tsec, tsec_diff, info_ndarr(segs,'segments')), end='')
             if timing is not None:
                 pulseid = timing.raw.pulseId(evt) # evt.get(EventId).fiducials()
                 pulseid_diff = pulseid-pulseid_old
                 pulseid_old = pulseid
                 print('  pulseId=%d diff=%d/record' % (pulseid, pulseid_diff))
             else:
                 print(' timing is None, pulseId is N/A')
             raw = det.raw.raw(evt)
             print(info_ndarr(raw,'    det.raw.raw(evt)'))
             #print('gain mode statistics:' + ue.info_pixel_gain_mode_statistics(gmaps))

             if dcfg is not None and (is_epix10ka or is_epixhr2x2):
               s = '    gain mode fractions for: FH       FM       FL'\
                   '       AHL-H    AML-M    AHL-L    AML-L\n%s' % (29*' ')
               print(ue.info_pixel_gain_mode_fractions(det.raw, evt, msg=s))

          print(info_det_evt(det, evt, ievt), end='   \r')
        print(info_det_evt(det, evt, ievt), end='   \n')


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
    #logging.basicConfig(filename='log.txt', filemode='w', format=fmt, level=DICT_NAME_TO_LEVEL[args.logmode])
    fmt = '[%(levelname).1s] %(name)s %(message)s' if args.logmode=='DEBUG' else '[%(levelname).1s] %(message)s'
    logging.basicConfig(format=fmt, level=DICT_NAME_TO_LEVEL[args.logmode])

    print('command line: %s' % info_command_line())
    logger.info(info_parser_arguments(parser))

    #print('input parameters: %s' % info_dict(opts, fmt='%s: %s', sep=', '))
    #pedestals_calibration(*args, **opts)
    #pedestals_calibration(**opts)
    ds_run_det(args)

    loop_run_step_evt(args)

    logger.info('DONE, consumed time %.3f sec' % (time() - t0_sec))


def argument_parser():
    from argparse import ArgumentParser

    d_dskwargs = None
    d_detname = None # 'epixquad'
    d_evtmax  = 0 # maximal number of events
    d_logmode = 'INFO'
    d_typeinfo= 'DRSE'

    h_dskwargs= 'string of comma-separated (no spaces) simple parameters for DataSource(**kwargs),'\
                ' ex: exp=<expname>,run=<runs>,dir=<xtc-dir>, ...,'\
                ' or <fname.xtc> or files=<fname.xtc>'\
                ' or pythonic dict of generic kwargs, e.g.:'\
                ' \"{\'exp\':\'tmoc00318\', \'run\':[10,11,12], \'dir\':\'/a/b/c/xtc\'}\", default = %s' % d_dskwargs
    h_detname = 'detector name, e.g. %s' % d_detname
    h_evtmax  = 'number of events to print, default = %s' % str(d_evtmax)
    h_logmode = 'logging mode, one of %s, default = %s' % (STR_LEVEL_NAMES, d_logmode)
    h_typeinfo= 'type of information for output D-detector, R-run-loop, S-step-loop, E-event-loop, default = %s' % d_typeinfo

    parser = ArgumentParser(usage=USAGE, description='Print info about experiment detector and run')
    parser.add_argument('-k', '--dskwargs', type=str, help=h_dskwargs)
    parser.add_argument('-d', '--detname', default=d_detname, type=str, help=h_detname)
    parser.add_argument('-n', '--evtmax', default=d_evtmax, type=int, help=h_evtmax)
    parser.add_argument('-L', '--logmode', default=d_logmode, type=str, help=h_logmode)
    parser.add_argument('-t', '--typeinfo', default=d_typeinfo, type=str, help=h_typeinfo)

    return parser

if __name__ == "__main__":
    do_main()
    sys.exit('End of %s'%SCRNAME)

# EOF
