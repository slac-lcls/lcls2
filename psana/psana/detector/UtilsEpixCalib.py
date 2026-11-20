"""
:py:class:`UtilsEpixm320Calib` dark processing algorithms for generic area detector
===============================================================================

Usage::

    from psana.detector.UtilsEpixCalib import *
    #OR
    import psana.detector.UtilsEpixCalib as uac

This software was developed for the LCLS project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2024-09-08 by Mikhail Dubrovin from UtilsEpixm320Calib
"""

import sys
import psana
from psana.detector.RepoManager import set_repoman_and_logger, logging
from psana.detector.Utils import info_dict, is_true, is_none
import psana.pscalib.calib.CalibConstants as cc
import psana.detector.UtilsCalib as uc
import psana.detector.UtilsCalibRepo as ucr
import psana.detector.utils_psana as up
from psana.detector.RepoManager import init_repoman_and_logger, fname_prefix, fname_prefix_merge, calib_file_name
import json
logger = logging.getLogger(__name__)

#from psana.detector.NDArrUtils import info_ndarr, save_ndarray_in_textfile
np, time, info_ndarr, save_ndarray_in_textfile = uc.np, uc.time, uc.info_ndarr, uc.save_ndarray_in_textfile
tstamps_run_and_now, add_metadata_kwargs, detector_name_short =\
  uc.tstamps_run_and_now, uc.add_metadata_kwargs, uc.detector_name_short


deploy_constants = uc.deploy_constants_ctypes_gainmodes
calib_file_name = uc.calib_file_name
save_constants_in_repository = ucr.save_constants_in_repository_v02

SCRNAME = sys.argv[0].rsplit('/')[-1]


def selected_record(i, events=1000000):
    return i<5\
       or (i<50 and not i%10)\
       or not i%100\
       or i>events-5


def pedestals_calibration(parser):

  args = parser.parse_args()
  kwa = vars(args)

  repoman = init_repoman_and_logger(parser=parser, **kwa)

  str_dskwargs = kwa.get('dskwargs', None)
  detname = kwa.get('det', None)
  nrecs   = kwa.get('nrecs', 100)
  stepnum = kwa.get('stepnum', None)
  stepmax = kwa.get('stepmax', None)
  evskip  = kwa.get('evskip', 0)
  events  = kwa.get('events', 1000)
  ctdepl  = kwa.get('ctdepl', None) # 'prs'
  dbsuffix= kwa.get('dbsuffix', None)

  dskwargs = up.datasource_kwargs_from_string(str_dskwargs)
  shortname = None

  logger.info('DataSource kwargs:%s' % info_dict(dskwargs, fmt='  %12s: %s', sep='\n'))
  ds = psana.DataSource(**dskwargs)

  t0_sec = time()
  tdt = t0_sec
  dpo = None
  nevtot = 0
  nevsel = 0
  nsteptot = 0
  break_loop = False
  break_runs = False
  dettype = None
  dic_consts_tot = {} # {<gain_mode>:{<ctype>:nda3d_shape=(4, 192, 384)}}
  kwa_depl = None
  odet = None

  expname = dskwargs.get('exp', None)
  runnum  = dskwargs.get('run', None)

  for irun,orun in enumerate(ds.runs()):

    if expname is None: expname = orun.expt
    if runnum is None: runnum = orun.runnum

    nevrun = 0
    logger.info('\n==== %02d run: %d exp: %s' % (irun, runnum, expname))
    logger.info(up.info_run(orun, cmt='run info:    ', sep='\n    ', verb=3))

    odet = orun.Detector(detname)
    if dettype is None:
        dettype = odet.raw._dettype
        repoman.set_dettype(dettype)

    gain_modes = odet.raw._gain_modes # i.e. for epixuhr: ('FHG', 'FMG', 'FLG1', 'FLG2', 'AHLG1', 'AHLG2', 'AMLG1', 'AMLG2')
    logger.info('created %s detector object' % detname)
    logger.info(up.info_detector(odet, cmt='  detector info:\n      ', sep='\n      '))
    logger.info('odet.raw._gain_modes: %s' % str(odet.raw._gain_modes))
    logger.info('odet.raw._data_bit_mask: %s' % oct(odet.raw._data_bit_mask))

    try:
      step_docstring = orun.Detector('step_docstring')
    except:
      step_docstring = None

    runtstamp = orun.timestamp       # 4193682596073796843 relative to 1990-01-01
    trun_sec = up.seconds(runtstamp) # 1607569818.532117 sec
    ts_run, ts_now = tstamps_run_and_now(int(trun_sec))

    break_steps = False

    for istep,step in enumerate(orun.steps()):
      nsteptot += 1

      metadic = json.loads(step_docstring(step)) if step_docstring is not None else {}

      logger.info('\n==== Begin step %1d ====' % istep)
      logger.info('Step %1d docstring: %s' % (istep, str(metadic)))
      ss = ''

      if stepmax is not None and istep >= stepmax:
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

         dpo = uc.DarkProc(**kwa)
         dpo.runnum = orun.runnum
         dpo.exp = expname
         dpo.ts_run, dpo.ts_now = ts_run, ts_now #uc.tstamps_run_and_now(env, fmt=uc.TSTAMP_FORMAT)

      count_none = 0
      break_events = False

      for ievt,evt in enumerate(step.events()):
        #print('Event %04d' % ievt, end='\r')
        sys.stdout.write('Event %04d\r' % ievt)
        nevrun += 1
        nevtot += 1

        if is_true(ievt < evskip, '==== Ev:%04d is skipped --evskip=%d' % (ievt,evskip), logger_method=logger.debug):
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
            count_none += 1
            if selected_record(count_none, events):
                logger.info('==== Ev:%04d raw is None, counter: %d' % (ievt, count_none))
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

      if ievt < events: logger.info('======== Ev:%04d end of events in run %d   step %d   counter of raw==None %d'%\
                                     (ievt, orun.runnum, istep, count_none))
      if True:
          dpo.summary()
          ctypes = ('pedestals', 'pixel_rms', 'pixel_status', 'pixel_max', 'pixel_min') # 'status_extra'
          arr_av1, arr_rms, arr_sta = dpo.constants_av1_rms_sta()
          arr_max, arr_min = dpo.constants_max_min()
          consts = arr_av1, arr_rms, arr_sta, arr_max, arr_min
          logger.info('evaluated constants: \n  %s\n  %s\n  %s\n  %s\n  %s' % (
                      info_ndarr(arr_av1, 'arr_av1', first=0, last=5),\
                      info_ndarr(arr_rms, 'arr_rms', first=0, last=5),\
                      info_ndarr(arr_sta, 'arr_sta', first=0, last=5),\
                      info_ndarr(arr_max, 'arr_max', first=0, last=5),\
                      info_ndarr(arr_min, 'arr_min', first=0, last=5)))
          dic_consts = dict(zip(ctypes, consts))
          gainmode = gain_mode(odet, metadic, istep) # nsteptot)
          kwa_depl = add_metadata_kwargs(orun, odet, **kwa)
          kwa_depl['gainmode'] = gainmode
          kwa_depl['repoman'] = repoman
          longname = kwa_depl['longname'] # odet.raw._uniqueid
          if shortname is None:
            shortname = detector_name_short(longname)
          logger.info('detector long  name: %s' % longname\
                     +'detector short name: %s' % shortname)
          kwa_depl['shortname'] = shortname

          logger.info('kwa_depl:\n%s' % info_dict(kwa_depl, fmt='  %12s: %s', sep='\n'))

          save_constants_in_repository(dic_consts, **kwa_depl)
          dic_consts_tot[gainmode] = dic_consts
          del(dpo)
          dpo=None

          logger.info('==== End of step %1d ====\n' % istep)

      if break_steps:
        logger.info('terminate_steps')
        break # break step loop

    if break_runs:
      logger.info('terminate_runs')
      break # break run loop

  gainmodes = [k for k in dic_consts_tot.keys()]
  logger.info('constants'\
             +'\n  created  for gain modes: %s' % str(gainmodes)\
             +'\n  expected for gain modes: %s' % str(odet.raw._gain_modes))

  gmodes = odet.raw._gain_modes #  or gainmodes
  kwa_depl['shape_as_daq'] = odet.raw._shape_as_daq()
  kwa_depl['exp'] = expname
  kwa_depl['det'] = detname
  kwa_depl['dbsuffix'] = dbsuffix
  kwa_depl['run_orig'] = runnum

  logger.debug('XXXXX kwa_depl: %s' % str(kwa_depl))

  #ctypes = ('pedestals', 'pixel_rms', 'pixel_status', 'pixel_max', 'pixel_min')
  ctypes = [cc.dic_calib_char_to_name[c] for c in ctdepl]
  logger.debug('ctdepl: %s ctypes: %s' % (ctdepl, str(ctypes)))

  deploy_constants(ctypes, gmodes, **kwa_depl)

  logger.debug('run/step/event loop is completed')
  repoman.logfile_save()


def gain_mode_name(odet, asic=0):
    """for epixm320 returns 'AHL'=AUTO, 'SH'=HIGH, 'SL'=LOW, or None from detector configuration object
       See: https://confluence.slac.stanford.edu/display/PSDM/EPIXM
    """
    if odet.raw._dettype != 'epixm320': return None
    cfg = odet.raw._config_object()[0].config
    comp = cfg.CompTH_ePixM[asic]
    preq = cfg.Precharge_DAC_ePixM[asic]
    #print('YYY:', comp, preq)
    return {0:'SH', 12:'AHL'}.get(comp, None) if preq == 45 else\
           'SL' if preq == 50 and comp == 63 else\
           None


def gain_mode(odet, metadic, nstep):
    """gain mode potential check using step, metadata from docstring, and config.
       curreently just print available info and return the gain mod from metadic/docstring
    """
    s  = 'gain mode consistency check\n  nstep: %d' % nstep
    s += '\n  metadic: %s' % str(metadic)
    cfggainmode = None
    if odet.raw._dettype  == 'epixm320':
        dcfg = odet.raw._config_object()
        epix320cfg = dcfg[0].config
        s += '\n  epix320cfg.CompTH_ePixM: %s' % str(epix320cfg.CompTH_ePixM)
        s += '\n  epix320cfg.Precharge_DAC_ePixM: %s' % str(epix320cfg.Precharge_DAC_ePixM)
        cfggainmode = gain_mode_name(odet)

    scantype = metadic.get('scantype', None)
    gainmode = metadic.get('gain_mode', None)
    s += '\n  scantype: %s\n  gain_mode: %s' % (scantype, gainmode)
    logging.info(s)
    return gainmode if scantype=='pedestal' else cfggainmode # else None


def epix_deploy_constants_v01(odet, **kwa_depl):
    ctdepl  = kwa_depl.get('ctdepl', 'prs')
    ctypes = [cc.dic_calib_char_to_name[c] for c in ctdepl]
    gainmodes = odet.raw._gain_states if 'g' in ctypes else\
                odet.raw._gain_modes # + ('AHLG1_L', 'AHLG2_L', ...
    logger.info('odet.raw._gain_modes: %s' % str(odet.raw._gain_modes))
    deploy_constants(ctypes, gainmodes, **kwa_depl)


def epixuhr3x2_deploy_constants(odet, **kwa_depl):
    ctdepl  = kwa_depl.get('ctdepl', 'prs')
    for c in ctdepl:
        ctype = cc.dic_calib_char_to_name[c]
        gainmodes = odet.raw._gain_states     if c == 'g' else\
                    odet.raw._gain_states_low if c == 'o' else\
                    odet.raw._gain_modes # 8-modes for pedestals and orther from dark dark
        logger.info(f'begin deployment for ctype: {ctype} gainmodes: {str(gainmodes)}')
        deploy_constants((ctype,), gainmodes, **kwa_depl)


def epixm320_deploy_constants(odet, **kwa_depl):
    import psana.detector.UtilsEpixm320Calib as ue320 # import epixm320_deploy_constants
    if True: # NEW VERSION - manage gain state order for different ctype
        ue320.epixm320_deploy_constants_v02(odet, **kwa_depl)
    else: # OLD VERSION
        ue320.epixm320_deploy_constants_v01(odet, **kwa_depl)


def epix_deploy_constants(parser):
    import psana.detector.utils_psana as up
    args = parser.parse_args()
    kwa = vars(args)
    repoman = set_repoman_and_logger(kwa)
    ds, orun, odet = up.ds_run_det(**kwa)
    kwa_depl = uc.add_metadata_kwargs(orun, odet, **kwa)

    kwa_depl['fmt_peds']   = '%' + kwa.get('fmt_peds',   '.3f')
    kwa_depl['fmt_offs']   = '%' + kwa.get('fmt_offs',   '.3f')
    kwa_depl['fmt_rms']    = '%' + kwa.get('fmt_rms',    '.3f')
    kwa_depl['fmt_status'] = '%' + kwa.get('fmt_status', '4i')
    kwa_depl['fmt_max']    = '%' + kwa.get('fmt_max',    'i')
    kwa_depl['fmt_min']    = '%' + kwa.get('fmt_min',    'i')
    kwa_depl['fmt_gain']   = '%' + kwa.get('fmt_gain',   '.6f')

    dettype = kwa_depl.get('dettype', None)
    ctdepl  = kwa.get('ctdepl', 'prs')
    ctypes = [cc.dic_calib_char_to_name[c] for c in ctdepl]
    logger.info(f'epix_deploy_constants for dettype: {dettype} ctdepl: {ctdepl} ctypes: {str(ctypes)}')

    if dettype == 'epixuhr3x2':
       epixuhr3x2_deploy_constants(odet, **kwa_depl)
    elif dettype == 'epixm320':
       epixm320_deploy_constants(odet, **kwa_depl)
    else:
       epix_deploy_constants_v01(odet, **kwa_depl)


if __name__ == "__main__":
    sys.stdout.write(80*'_', '\n')
    logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d %(filename)s: %(message)s', level=logging.INFO)

    kwa = {\
        'dskwargs': 'exp=tstx00417,run=317,dir=/reg/neh/operator/tstopr/data/drp/tst/tstx00417/xtc/',\
        'det'     : 'tst_epixm',\
        'dirrepo' : 'work1',\
        'nrecs1'  : 100,\
        'nrecs'   : 200,\
    }

    pedestals_calibration(**kwa)
    sys.exit('End of %s' % sys.argv[0])

# EOF
