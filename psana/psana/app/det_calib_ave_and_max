#!/usr/bin/env python

import sys
from psana.detector.UtilsLogging import logging, STR_LEVEL_NAMES, DICT_NAME_TO_LEVEL
logger = logging.getLogger(__name__)

SCRNAME = sys.argv[0].rsplit('/')[-1]

USAGE = '%s -k <dataset-kwargs> -d <detector-name> -n <number-of-events> -m <number-events-to-skip> -M <mode-r/c/i/p> -L <log-level-str>'%SCRNAME\
      + '\n  Examples:'\
      + '\n  %s -k exp=uedc00106,run=25 -d epixquad1kfps -n 10000 -M c'%SCRNAME\

def argument_parser():
    import argparse

    d_dskwargs = 'exp=uedc00106,run=25'
    d_detname  = 'epixquad1kfps'
    d_events   = 100
    d_evskip   = 0
    d_logmode  = 'INFO'
    d_prefix   = 'img'
    d_aslice   = ':'
    d_mode     = 'p'

    h_dskwargs = 'dataset name, default = %s' % d_dskwargs
    h_detname  = 'detector name, default = %s' % d_detname
    h_events   = 'number of events to collect, default = %s' % d_events
    h_evskip   = 'number of events to skip, default = %s' % d_evskip
    h_logmode  = 'logging mode, one of %s, default = %s' % (STR_LEVEL_NAMES, d_logmode)
    h_prefix   = 'output file name prefix, default = %s' % str(d_prefix)
    h_aslice   = 'array slice, e.g. 0:180,620:, default = %s' % str(d_aslice)
    h_mode     = 'mode of data: r/c/i/p : raw/calib/image/raw-peds default = %s' % str(d_aslice)

    parser = argparse.ArgumentParser(usage=USAGE, description='Accumulates det.calib array average and max')
    parser.add_argument('-k', '--dskwargs',   default=d_dskwargs,   type=str,   help=h_dskwargs)
    parser.add_argument('-d', '--detname',  default=d_detname,  type=str,   help=h_detname)
    parser.add_argument('-n', '--events',   default=d_events,   type=int,   help=h_events)
    parser.add_argument('-m', '--evskip',   default=d_evskip,   type=int,   help=h_evskip)
    parser.add_argument('-L', '--logmode',  default=d_logmode,  type=str,   help=h_logmode)
    parser.add_argument('-f', '--prefix',   default=d_prefix,   type=str,   help=h_prefix)
    parser.add_argument('-S', '--aslice',   default=d_aslice,   type=str,   help=h_aslice)
    parser.add_argument('-M', '--mode',     default=d_mode,     type=str,   help=h_mode)

    return parser


def det_calib_ave_and_max():

    parser = argument_parser()
    args = parser.parse_args()

    kwa = vars(args)
    print('parser.parse_args: %s' % str(args))

    #if len(sys.argv)<3: sys.exit('%s\n EXIT - MISSING ARGUMENTS\n' % USAGE)

    str_dskwargs = args.dskwargs
    detname  = args.detname
    events   = args.events
    evskip   = args.evskip
    logmode  = args.logmode
    prefix   = args.prefix
    aslice   = args.aslice
    mode     = args.mode

    logging.basicConfig(format='[%(levelname).1s] %(name)s L%(lineno)04d: %(message)s', level=DICT_NAME_TO_LEVEL[logmode])

    import numpy as np
    from psana import DataSource
    import psana.detector.utils_psana as up
    from psana.pyalgos.generic.NDArrUtils import info_ndarr #, print_ndarr, divide_protected, shape_as_3d, shape_as_3d
    from psana.detector.Utils import selected_record # str_tstamp, time, get_login, info_dict, selected_record, info_command_line

    _slice = eval('np.s_[%s]' % aslice)

    nev_sum = 0
    nda_sum = None
    nda_max = None
    dettype = None
    peds = None
    runnum = None
    do_break = False

    dskwargs = up.datasource_kwargs_from_string(str_dskwargs, detname=detname)
    dskwargs['max_events'] = events
    logger.info('DataSource kwargs: %s' % str(dskwargs))

    try:
      ds = DataSource(**dskwargs)
    except Exception as err:
      logger.error('DataSource(**dskwargs) does not work:\n    %s' % err)
      sys.exit('Exit processing')

    expname = dskwargs.get('exp', None)
    runnum  = dskwargs.get('run', None)

    for irun,orun in enumerate(ds.runs()):

      if expname is None: expname = orun.expt
      if runnum is None: runnum = orun.runnum

      nevrun = 0
      logger.info('\n==== %02d run: %d exp: %s' % (irun, runnum, expname))
      logger.info(up.info_run(orun, cmt='run info:\n    ', sep='\n    ', verb=3))

      odet = orun.Detector(detname)
      if dettype is None:
          dettype = odet.raw._dettype
          #repoman.set_dettype(dettype)

      logger.info('created %s detector object' % detname)
      logger.info(up.info_detector(odet, cmt='  detector info:\n      ', sep='\n      '))

      runtstamp = orun.timestamp    # 4193682596073796843 relative to 1990-01-01
      trun_sec = up.seconds(runtstamp) # 1607569818.532117 sec
      ts_run, ts_now = up.tstamps_run_and_now(int(trun_sec))


      for i, evt in enumerate(orun.events()):
        if i<evskip: continue
        if i>events:
           do_break = True
           break
        if i<5:
            peds = odet.raw._pedestals()
            logging.debug(info_ndarr(peds, 'pedestals'))
            gains = odet.raw._gain()
            logging.debug(info_ndarr(gains, 'gains'))
            assert peds is not None

        raw = odet.raw.raw(evt)
        logging.debug(info_ndarr(raw, 'raw'))

        if raw is None:
            logging.warning('Event %4d raw is None' % i)
            continue

        arr = raw if mode in ('r','p') else\
              odet.raw.calib(evt) if mode == 'c' else\
              odet.raw.image(evt) if mode == 'i' else\
              raw

        arr = arr.astype(peds.dtype) if aslice in (':', None) else\
              arr[_slice].astype(peds.dtype)

        logging.debug(info_ndarr(arr, 'arr'))

        if selected_record(i):
            logging.info(info_ndarr(arr, 'Event %4d arr[%s]:' % (i, aslice)))

        if nda_max is None:
            nev_sum = 1
            nda_sum = np.array(arr).astype(peds.dtype)
            nda_max = np.array(arr).astype(peds.dtype)
        else:
            np.maximum(arr, nda_max, out=nda_max)
            nda_sum += arr
            nev_sum += 1
      if do_break: break # runs

    if nev_sum: nda_sum /= nev_sum

    if mode == 'p':
        nda_max -= peds
        nda_sum -= peds
    if nda_sum.ndim>2: nda_sum.shape = nda_max.shape

    logging.info('\nStatistics of events nevt:%d nev_sum:%d'%(i,nev_sum))

    _detname = detname #.replace('.','-').replace(':','-')
    _prefix = '%s-%s-%s-r%04d-e%06d-mode-%s' % (prefix, _detname, expname, runnum, nev_sum, mode)

    fname = _prefix + '-max.npy'
    np.save(fname, nda_max)
    logging.info('saved file %s' % fname)

    fname = _prefix + '-ave.npy'
    np.save(fname, nda_sum)
    #np.savetxt(fname, nda_sum, fmt='%.3f')
    logging.info('saved file %s'%fname)

    sys.exit('END OF %s' % SCRNAME)

det_calib_ave_and_max()

# EOF
