"""
:py:class:`UtilsCalib` dark processing algorithms for generic area detector
===============================================================================

Usage::

    from psana.detector.UtilsCalib import *
    #OR
    import psana.detector.UtilsCalib as uac

This software was developed for the LCLS project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2024-04-09 by Mikhail Dubrovin
"""
from psana.detector.UtilsCalib import * # logging
import json

logger = logging.getLogger(__name__)

SCRNAME = sys.argv[0].rsplit('/')[-1]

def pedestals_calibration(parser):

  args = parser.parse_args()
  kwa = vars(args)

  repoman = init_repoman_and_logger(parser=parser, **kwa)

  str_dskwargs = kwa.get('dskwargs', None)
  detname = kwa.get('det', None)
  nrecs   = kwa.get('nrecs', 100)
  stepnum = kwa.get('stepnum', None)
  stepmax = kwa.get('stepmax', 1)
  evskip  = kwa.get('evskip', 0)
  events  = kwa.get('events', 1000)

  dskwargs = up.datasource_kwargs_from_string(str_dskwargs)
  logger.info('DataSource kwargs: %s' % str(dskwargs))
  ds = DataSource(**dskwargs)

  t0_sec = time()
  tdt = t0_sec
  dpo = None
  nevtot = 0
  nevsel = 0
  nsteptot = 0
  break_loop   = False
  break_runs   = False
  dettype = None

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
        repoman.set_dettype(dettype)

    logger.info('created %s detector object' % detname)
    logger.info(up.info_detector(odet, cmt='  detector info:\n      ', sep='\n      '))

    step_docstring = orun.Detector('step_docstring')

    runtstamp = orun.timestamp       # 4193682596073796843 relative to 1990-01-01
    trun_sec = up.seconds(runtstamp) # 1607569818.532117 sec
    ts_run, ts_now = tstamps_run_and_now(int(trun_sec))

    break_steps  = False

    for istep,step in enumerate(orun.steps()):
      nsteptot += 1

      metadic = json.loads(step_docstring(step))

      logger.info('Step %1d docstring: %s' % (istep, str(metadic)))
      ss = ''

      if istep>=stepmax:
          logger.info('==== Step:%02d loop is terminated --stepmax=%d' % (istep, stepmax))
          break_steps = True
          break
      elif stepnum is not None:
          if istep < stepnum:
              logger.info('==== Step:%02d is skipped --stepnum=%d' % (istep, stepnum))
              continue
          elif istep > stepnum:
              logger.info('==== Step:%02d loop is terminated --stepnum=%d' % (istep, stepnum))
              break_steps = True
              break

      if dpo is None:
         kwa['rms_hi'] = odet.raw._data_bit_mask - 10
         kwa['int_hi'] = odet.raw._data_bit_mask - 10
         dpo = DarkProc(**kwa)
         dpo.runnum = orun.runnum
         dpo.exp = expname
         dpo.ts_run, dpo.ts_now = ts_run, ts_now #uc.tstamps_run_and_now(env, fmt=uc.TSTAMP_FORMAT)

      break_events = False

      for ievt,evt in enumerate(step.events()):
        #print('Event %04d' % ievt, end='\r')
        sys.stdout.write('Event %04d\r' % ievt)
        nevrun += 1
        nevtot += 1

        if ievt < evskip:
            logger.debug('==== Ev:%04d is skipped --evskip=%d' % (ievt,evskip))
            continue
        elif evskip>0 and (ievt == evskip):
            logger.info('Events < --evskip=%d are skipped' % evskip)

        if ievt > events-1:
            logger.info(ss)
            logger.info('\n==== Ev:%04d event loop is terminated --events=%d' % (ievt,events))
            break_events = True
            break

        raw = odet.raw.raw(evt)

        if raw is None:
            logger.info('==== Ev:%04d raw is None' % (ievt))
            continue

        nevsel += 1

        tsec = time()
        dt   = tsec - tdt
        tdt  = tsec
        if selected_record(ievt+1, events):
            ss = 'run[%d] %d  step %d  events total/run/step/selected: %4d/%4d/%4d/%4d  time=%7.3f sec dt=%5.3f sec'%\
                 (irun, orun.runnum, istep, nevtot, nevrun, ievt+1, nevsel, time()-t0_sec, dt)
            logger.info(ss)

        status = dpo.event(raw, ievt)
        if status == 2:
            logger.info('requested statistics --nrecs=%d is collected - terminate loops' % nrecs)
            break_events = True
            break
        # End of event-loop

      if ievt < events: logger.info('==== Ev:%04d end of events in run %d step %d'%\
                                     (ievt, orun.runnum, istep))
      if True:
          dpo.summary()
          ctypes = ('pedestals', 'pixel_rms', 'pixel_status') # 'status_extra'
          consts = arr_av1, arr_rms, arr_sta = dpo.constants_av1_rms_sta()
          logger.info('evaluated constants: \n  %s\n  %s\n  %s' % (
                      info_ndarr(arr_av1, 'arr_av1', first=0, last=5),\
                      info_ndarr(arr_rms, 'arr_rms', first=0, last=5),\
                      info_ndarr(arr_sta, 'arr_sta', first=0, last=5)))
          dic_consts = dict(zip(ctypes, consts))
          kwa_depl = add_metadata_kwargs(orun, odet, **kwa)
          kwa_depl['repoman'] = repoman

          print('XXX kwa_depl:', kwa_depl)
          #deploy_constants(dic_consts, **kwa_depl)
          del(dpo)
          dpo=None

      if break_steps:
        logger.info('terminate_steps')
        break # break step loop

    if break_runs:
      logger.info('terminate_runs')
      break # break run loop



  logger.debug('run/step/event loop is completed')
  repoman.logfile_save()

  sys.exit('TEST EXIT see commented deploy_constants')




#      if odet.raw._dettype  == 'epixm320':
#          epix320cfg = dcfg[0].config
#          print('XXX  epix320cfg.CompTH_ePixM', epix320cfg.CompTH_ePixM)
#          print('XXX  epix320cfg.Precharge_DAC_ePixM', epix320cfg.Precharge_DAC_ePixM)


        #logger.info('TB-debugg metadic: %s' % str(metadic))



if __name__ == "__main__":
    """
    """

    sys.stdout.write(80*'_', '\n')
    logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d %(filename)s: %(message)s', level=logging.INFO)

    kwa = {\
        'dskwargs': 'exp=tstx00417,run=317,dir=/reg/neh/operator/tstopr/data/drp/tst/tstx00417/xtc/',\
        'det'     : 'tst_epixm',\
        'dirrepo' : 'work',\
        'nrecs1'  : 100,\
        'nrecs'   : 200,\
    }

    pedestals_calibration(**kwa)
    sys.exit('End of %s' % sys.argv[0])


# EOF
