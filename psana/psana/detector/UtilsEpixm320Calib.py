"""
:py:class:`UtilsEpixm320Calib` dark processing algorithms for generic area detector
===============================================================================

Usage::

    from psana.detector.UtilsEpixm320Calib import *
    #OR
    import psana.detector.UtilsEpixm320Calib as uac

This software was developed for the LCLS project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2024-04-09 by Mikhail Dubrovin
"""
from psana.detector.Utils import info_dict

from psana.detector.UtilsCalib import * # logging
import json

logger = logging.getLogger(__name__)

SCRNAME = sys.argv[0].rsplit('/')[-1]


def selected_record(i, events=1000000):
    return i<5\
       or (i<50 and not i%10)\
       or not i%100\
       or i>events-5


def detector_name_short(detlong):
  """ converts long name like epixm320_0016908288-0000000000-0000000000-4005754881-2080374808-0177177345-2852126742
      to short: epixm320_000004
  """
  from psana.pscalib.calib.MDBWebUtils import pro_detector_name
  return pro_detector_name(detlong, add_shortname=True)


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
  shortname = None

  logger.info('DataSource kwargs:%s' % info_dict(dskwargs, fmt='  %12s: %s', sep='\n'))
  ds = DataSource(**dskwargs)

  t0_sec = time()
  tdt = t0_sec
  dpo = None
  nevtot = 0
  nevsel = 0
  nsteptot = 0
  break_loop = False
  break_runs = False
  dettype = None
  dic_consts_tot = {} # {<gain_mode>:{<ctype>:nda3d_shape:(4, 192, 384)}}
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

    #sys.exit('TEST EXIT')

    odet = orun.Detector(detname)
    if dettype is None:
        dettype = odet.raw._dettype
        repoman.set_dettype(dettype)

    logger.info('created %s detector object' % detname)
    logger.info(up.info_detector(odet, cmt='  detector info:\n      ', sep='\n      '))

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

      print('\n==== Begin step %1d ====' % istep)
      logger.info('Step %1d docstring: %s' % (istep, str(metadic)))
      ss = ''

      if istep >= stepmax:
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

      count_none = 0
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
          print('detector long  name: %s' % longname)
          print('detector short name: %s' % shortname)
          kwa_depl['shortname'] = shortname

          #kwa_depl['segment_ids'] = odet.raw._segment_ids()

          logger.info('kwa_depl:\n%s' % info_dict(kwa_depl, fmt='  %12s: %s', sep='\n'))
          #sys.exit('TEST EXIT')

          save_constants_in_repository(dic_consts, **kwa_depl)
          dic_consts_tot[gainmode] = dic_consts
          del(dpo)
          dpo=None

          print('==== End of step %1d ====\n' % istep)

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

  ctypes = ('pedestals', 'pixel_rms', 'pixel_status', 'pixel_max', 'pixel_min')
  gmodes = odet.raw._gain_modes #  or gainmodes
  kwa_depl['shape_as_daq'] = odet.raw._shape_as_daq()
  kwa_depl['exp'] = expname
  kwa_depl['det'] = detname
  kwa_depl['run_orig'] = runnum

  #print('XXXXX kwa_depl', kwa_depl)
  #sys.exit('TEST EXIT')

  deploy_constants(ctypes, gmodes, **kwa_depl)

  logger.debug('run/step/event loop is completed')
  repoman.logfile_save()

  #sys.exit('TEST EXIT see commented deploy_constants')


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
       curreently just print available inffo and return the gain mod from metadic/docstring
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



#def save_constants_in_repository_per_asic(dic_consts, **kwa):
#    """ DEPRECATED - ASIC can not be replacable unit - no reason to keep constants peer ASIC"""
#
#    logger.debug('save_constants_in_repository kwa:', kwa)
#
#    CTYPE_DTYPE = cc.dic_calib_name_to_dtype # {'pedestals': np.float32,...}
#    repoman  = kwa.get('repoman', None)
#    expname  = kwa.get('exp', None)
#    detname  = kwa.get('det', None)
#    dettype  = kwa.get('dettype', None)
#    deploy   = kwa.get('deploy', False)
#    dirrepo  = kwa.get('dirrepo', './work')
#    dirmode  = kwa.get('dirmode',  0o2775)
#    filemode = kwa.get('filemode', 0o664)
#    group    = kwa.get('group', 'ps-users')
#    tstamp   = kwa.get('tstamp', '2010-01-01T00:00:00')
#    tsshort  = kwa.get('tsshort', '20100101000000')
#    runnum   = kwa.get('run_orig', None)
#    uniqueid = kwa.get('uniqueid', 'not-def-id')
#    segids   = kwa.get('segment_ids', [])
#    gainmode = kwa.get('gainmode', None)
#
#    fmt_peds   = kwa.get('fmt_peds', '%.3f')
#    fmt_rms    = kwa.get('fmt_rms',  '%.3f')
#    fmt_status = kwa.get('fmt_status', '%4i')
#
#    CTYPE_FMT = {'pedestals'   : fmt_peds,
#                 'pixel_rms'   : fmt_rms,
#                 'pixel_status': fmt_status,
#                 'status_extra': fmt_status}
#
#    if repoman is None:
#       repoman = RepoManager(dirrepo=dirrepo, dirmode=dirmode, filemode=filemode, group=group, dettype=dettype)
#
#    logger.info('segment_ids:\n%s' % '\n'.join([id for id in segids]))
#
#    for i,sid in enumerate(segids):
#      logger.info('\nsave segment constants for gain mode:%s in repo for id:%s' % (gainmode, sid))
#      for ctype, nda in dic_consts.items():
#
#        dir_ct = repoman.makedir_ctype(sid, ctype)
#        fprefix = fname_prefix(detname, tsshort, expname, runnum, dir_ct)
#
#        fname = '%s-%s-%s.data' % (fprefix, ctype, gainmode)
#        fmt = CTYPE_FMT.get(ctype,'%.5f')
#        nda2d = nda[i,:,:] # select for shape:(4, 192, 384)
#        #print(info_ndarr(nda2d, '   %s' % ctype))
#
#        save_ndarray_in_textfile(nda2d, fname, filemode, fmt)
#        ###save_2darray_in_textfile(nda, fname, filemode, fmt)
#        logger.info('saved: %s' % fname)


def calib_file_name(fprefix, ctype, gainmode, fmt='%s-%s-%s.data'):
    return fmt % (fprefix, ctype, gainmode)


def save_constants_in_repository(dic_consts, **kwa):

    logger.debug('save_constants_in_repository kwa:', kwa)

    CTYPE_DTYPE = cc.dic_calib_name_to_dtype # {'pedestals': np.float32,...}
    repoman  = kwa.get('repoman', None)
    expname  = kwa.get('exp', None)
    detname  = kwa.get('det', None)
    dettype  = kwa.get('dettype', None)
    deploy   = kwa.get('deploy', False)
    dirrepo  = kwa.get('dirrepo', './work')
    dirmode  = kwa.get('dirmode',  0o2775)
    filemode = kwa.get('filemode', 0o664)
    group    = kwa.get('group', 'ps-users')
    tstamp   = kwa.get('tstamp', '2010-01-01T00:00:00')
    tsshort  = kwa.get('tsshort', '20100101000000')
    runnum   = kwa.get('run_orig', None)
    uniqueid = kwa.get('uniqueid', 'not-def-id')
    shortname= kwa.get('shortname', 'not-def-shortname')
    segids   = kwa.get('segment_ids', [])
    gainmode = kwa.get('gainmode', None)

    fmt_peds   = kwa.get('fmt_peds', '%.3f')
    fmt_rms    = kwa.get('fmt_rms',  '%.3f')
    fmt_status = kwa.get('fmt_status', '%4i')
    fmt_max    = kwa.get('fmt_max', '%i')
    fmt_min    = kwa.get('fmt_min', '%i')

    CTYPE_FMT = {'pedestals'   : fmt_peds,
                 'pixel_rms'   : fmt_rms,
                 'pixel_status': fmt_status,
                 'pixel_max'   : fmt_max,
                 'pixel_min'   : fmt_min,
                 'status_extra': fmt_status}

    if repoman is None:
       repoman = RepoManager(dirrepo=dirrepo, dirmode=dirmode, filemode=filemode, group=group, dettype=dettype)
    #dircons = repoman.makedir_constants(dname='constants')
    #fprefix = fname_prefix(detname, tsshort, expname, runnum, dircons)

    #logger.info('segment_ids:\n%s' % '\n'.join([id for id in segids]))

    segid = uniqueid.split('_')[1]

    logger.info('\nsave segment constants for gain mode:%s in repo for segment id: %s' % (gainmode, segid))

    for ctype, nda in dic_consts.items():

        dir_ct = repoman.makedir_ctype(segid, ctype)
        #fprefix = fname_prefix(detname, tsshort, expname, runnum, dir_ct)
        #fprefix = fname_prefix(dettype, tsshort, expname, runnum, dir_ct)
        fprefix = fname_prefix(shortname, tsshort, expname, runnum, dir_ct)

        fname = calib_file_name(fprefix, ctype, gainmode)
        fmt = CTYPE_FMT.get(ctype,'%.5f')
        print(info_ndarr(nda, '   %s' % ctype))  # shape:(4, 192, 384)

        save_ndarray_in_textfile(nda, fname, filemode, fmt)
        ###save_2darray_in_textfile(nda, fname, filemode, fmt)

        logger.info('saved: %s' % fname)


def deploy_constants(ctypes, gainmodes, **kwa):

    from psana.pscalib.calib.MDBUtils import data_from_file
    from psana.pscalib.calib.MDBWebUtils import add_data_and_two_docs

    CTYPE_DTYPE = cc.dic_calib_name_to_dtype # {'pedestals': np.float32,...}
    repoman  = kwa.get('repoman', None)
    expname  = kwa.get('exp', None)
    detname  = kwa.get('det', None)
    dettype  = kwa.get('dettype', None)
    deploy   = kwa.get('deploy', False)
    dirrepo  = kwa.get('dirrepo', './work')
    dirmode  = kwa.get('dirmode',  0o2775)
    filemode = kwa.get('filemode', 0o664)
    group    = kwa.get('group', 'ps-users')
    tstamp   = kwa.get('tstamp', '2010-01-01T00:00:00')
    tsshort  = kwa.get('tsshort', '20100101000000')
    runnum   = kwa.get('run_orig',None)
    uniqueid = kwa.get('uniqueid', 'not-def-id')
    shortname= kwa.get('shortname', 'not-def-shortname')
    shape_as_daq = kwa.get('shape_as_daq', (4, 192, 384))

    fmt_peds   = kwa.get('fmt_peds', '%.3f')
    fmt_rms    = kwa.get('fmt_rms',  '%.3f')
    fmt_status = kwa.get('fmt_status', '%4i')
    fmt_max    = kwa.get('fmt_max', '%i')
    fmt_min    = kwa.get('fmt_min', '%i')

    CTYPE_FMT = {'pedestals'   : fmt_peds,
                 'pixel_rms'   : fmt_rms,
                 'pixel_status': fmt_status,
                 'pixel_max'   : fmt_max,
                 'pixel_min'   : fmt_min,
                 'status_extra': fmt_status}

    if repoman is None:
       repoman = RepoManager(dirrepo=dirrepo, dirmode=dirmode, filemode=filemode, group=group, dettype=dettype)
    #dircons = repoman.makedir_constants(dname='constants')
    #fprefix = fname_prefix(detname, tsshort, expname, runnum, dircons)

    #segid = uniqueid.split('_',1)[-1]
    segid = uniqueid.split('_')[1]
    logger.info('\n\n\ndeploy_constants'\
               +'\n  segment id: %s' % segid\
               +'\n  gainmodes : %s' % str(gainmodes))

    #for ctype, nda in dic_consts.items():
    for ctype in ctypes:

      dir_ct = repoman.makedir_ctype(segid, ctype)
      #fprefix = fname_prefix(detname, tsshort, expname, runnum, dir_ct)
      #fprefix = fname_prefix(dettype, tsshort, expname, runnum, dir_ct)
      fprefix = fname_prefix(shortname, tsshort, expname, runnum, dir_ct)
      logger.info('=========================== combine calib array for %s' % ctype
                 +'\n  shortname:%s\n  tsshort:%s\n  expname:%s\n  runnum:%d\n  dir_ct:%s\n  fprefix:%s\n\n'%\
                  (shortname, tsshort, expname, runnum, dir_ct, fprefix))
      dic_nda = {}

      for gm in gainmodes:

        fname = calib_file_name(fprefix, ctype, gm)
        fmt = CTYPE_FMT.get(ctype,'%.5f')
        #save_ndarray_in_textfile(nda, fname, filemode, fmt)
        #save_2darray_in_textfile(nda, fname, filemode, fmt)

        logger.info('extract constants from repo file: %s' % fname)

        dtype = 'ndarray'
        kwa['iofname'] = fname
        kwa['ctype'] = ctype
        kwa['dtype'] = dtype
        kwa['extpars'] = {'content':'extended parameters dict->json->str',}
        _ = kwa.pop('exp',None) # remove parameters from kwargs - they passed as positional arguments
        _ = kwa.pop('det',None)

        try:
          data = data_from_file(fname, ctype, dtype, True)
          logger.info(info_ndarr(data, 'constants loaded from file', last=10))
        except AssertionError as err:
          logger.warning(err)
          data = np.zeros(shape_as_daq, np.uint16)
          logger.info(info_ndarr(data, 'substitute array with', last=10))
        dic_nda[gm] = data

      nda = np.stack([dic_nda[gm] for gm in gainmodes])
      fname = calib_file_name(fprefix, ctype, 'comb')
      fmt = CTYPE_FMT.get(ctype,'%.5f')
      save_ndarray_in_textfile(nda, fname, filemode, fmt)
      logger.info(info_ndarr(nda, 'constants combined for %s' % ctype))  # shape:(3, 4, 192, 384)
      logger.info('saved in file: %s' % fname)

      if deploy:
            logger.info('DEPLOY metadata: %s' % info_dict(kwa, fmt='%s: %s', sep='  ')) #fmt='%12s: %s'

            detname = kwa['longname']
            resp = add_data_and_two_docs(nda, expname, detname, **kwa) # url=cc.URL_KRB, krbheaders=cc.KRBHEADERS
            if resp:
                #id_data_exp, id_data_det, id_doc_exp, id_doc_det = resp
                logger.debug('deployment id_data_exp:%s id_data_det:%s id_doc_exp:%s id_doc_det:%s' % resp)
            else:
                logger.info('constants are not deployed')
                exit()
      else:
            logger.warning('TO DEPLOY CONSTANTS IN DB ADD OPTION -D')


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
