#!/usr/bin/env python
""" import psana.detector.UtilsEpixm320ChargeInjection as ueci
"""
import os
import sys
from time import time, sleep
import json

import psana.detector.UtilsGraphics as ug
from psana.detector.UtilsLogging import logging  # DICT_NAME_TO_LEVEL, init_stream_handler
import psana.detector.UtilsCalib as uc
import psana.detector.UtilsEpix10kaCalib as uec
from psana.detector.UtilsEpixm320Calib import save_constants_in_repository
from psana.detector.utils_psana import seconds, str_tstamp, info_run, info_detector, seconds,\
    dict_run, dict_detector, dict_datasource
from psana.detector.NDArrUtils import info_ndarr, divide_protected, save_2darray_in_textfile, save_ndarray_in_textfile
#from psana.detector.Utils import info_dict
from psana.detector.RepoManager import init_repoman_and_logger
logger = logging.getLogger(__name__)

SCRNAME = sys.argv[0].rsplit('/')[-1]

create_directory = uec.create_directory
gr = ug.gr



def charge_injection(parser):

    args = parser.parse_args()
    kwa = vars(args)
    repoman = init_repoman_and_logger(parser=parser, **kwa)

    str_dskwargs = kwa.get('dskwargs', None)
    detname    = kwa.get('det', None)
    idx        = kwa.get('idx', 0)
    dirrepo    = kwa.get('dirrepo', uec.DIR_REPO_EPIX10KA)
    plotim     = kwa.get('plotim', 1)
    fmt_offset = kwa.get('fmt_offset', '%.6f')
    fmt_peds   = kwa.get('fmt_peds',   '%.3f')
    fmt_rms    = kwa.get('fmt_rms',    '%.3f')
    fmt_status = kwa.get('fmt_status', '%d')
    fmt_gain   = kwa.get('fmt_gain',   '%.6f')
    fmt_chi2   = kwa.get('fmt_chi2',   '%.3f')
    dirmode    = kwa.get('dirmode', 0o2775)
    filemode   = kwa.get('filemode', 0o664)
    group      = kwa.get('group', 'ps-users')
    stepnum    = kwa.get('stepnum', None)
    stepmax    = kwa.get('stepmax', 230)
    stepskip   = kwa.get('stepskip', 0)
    events     = kwa.get('events', 100000)
    evstep     = kwa.get('evstep', 1)
    evskip     = kwa.get('evskip', 0)
    logmode    = kwa.get('logmode', 'DEBUG')
    pixrc      = kwa.get('pixrc', None) # ex.: '23,123'
    nsigm      = kwa.get('nsigm', 8)
    deploy     = kwa.get('deploy', False)
    sslice     = kwa.get('slice', '0:,0:')
    irun       = None
    exp        = None
    npmin      = 5
    nstep_peds = 0
    tsec_show  = 2
    tsec_show_end = 60
    step_docstring = None
    dfid_med = 7761 # THIS VALUE DEPENDS ON EVENT RATE -> SHOULD BE AUTOMATED

    dskwargs = uec.datasource_kwargs_from_string(str_dskwargs)
    logger.info('dskwargs:%s\n' % uec.info_dict(dskwargs))

    expname = dskwargs.get('exp', None)
    runnum  = dskwargs.get('run', None)


    try: ds = uec.DataSource(**dskwargs)
    except Exception as err:
        logger.error('DataSource(**dskwargs) does not work for **dskwargs: %s\n    %s' % (dskwargs, err))
        sys.exit('EXIT - requested DataSource does not exist or is not accessible.')

    dict_ds = dict_datasource(ds)
    logger.info('dict_datasource:%s\n' % uec.info_dict(dict_ds))

#    xtc_files = getattr(ds, 'xtc_files', None)
#    logger.info('ds.xtc_files:\n  %s' % ('None' if xtc_files is None else '\n  '.join(xtc_files)))

#    sys.exit('TEST EXIT')


#    cpdic = get_config_info_for_dataset_detname(**kwa)

#    logger.info('config_info:%s' % uec.info_dict(cpdic))  # fmt=fmt, sep=sep+sepnext)

#    tstamp    = cpdic.get('tstamp', None)
#    panel_ids = cpdic.get('panel_ids', None)
#    exp       = cpdic.get('expname', None)
#    shape     = cpdic.get('shape', None)
#    irun      = cpdic.get('runnum', None)
#    dettype   = cpdic.get('dettype', None)
#    dsname    = dskwargs
#    nr,nc     = shape

#    repoman.set_dettype(dettype)

#    gainbitw = ue.gain_bitword(dettype)  # 0o100000
#    databitw = ue.data_bitword(dettype)  # 0o077777
#    logger.info('gainbitw %s databitw %s' % (oct(gainbitw), oct(databitw)))
#    assert gainbitw is not None, 'gainbitw has to be defined for dettype %s' % str(dettype)
#    assert databitw is not None, 'databitw has to be defined for dettype %s' % str(dettype)

#    if display:
#        fig2, axim2, axcb2 = gr.fig_img_cbar_axes()
#        gr.move_fig(fig2, 500, 10)
#        gr.plt.ion() # do not hold control on plt.show()
#    else:
#        fig2, axim2, axcb2 = None, None, None

#    panel_id = get_panel_id(panel_ids, idx)
#    logger.info('panel_id: %s' % panel_id)


    t0_sec = time()
    tdt = t0_sec
    nevtot = 0
    nevsel = 0
    nsteptot = 0
    break_runs = False
    dettype = None
    dic_consts_tot = {} # {<gain_mode>:{<ctype>:nda3d_shape:(4, 192, 384)}}
    kwa_depl = None
    dic_run = None
    dbo = None
    flimg = None


    for irun,orun in enumerate(ds.runs()):

      if dic_run is None:
          dic_run = dict_run(orun)
          logger.info('dict_run:%s' % uec.info_dict(dic_run))  # fmt=fmt, sep=sep+sepnext)

      if expname is None: expname = orun.expt
      runnum = orun.runnum

      logger.info('\n==== %02d run: %d exp: %s' % (irun, runnum, expname))
      logger.info(info_run(orun, cmt='run info:    ', sep='\n    ', verb=3))

      odet = orun.Detector(detname)
      if dettype is None:
          dettype = odet.raw._dettype
          repoman.set_dettype(dettype)

      logger.info('created %s detector object' % detname)

      dict_det = dict_detector(odet)
      logger.info('dict_detector:%s' % uec.info_dict(dict_det))

      try:
        step_docstring = orun.Detector('step_docstring')
      except:
        step_docstring = None

      runtstamp = orun.timestamp    # 4193682596073796843 code of sec and msec relative to 1990-01-01
      trun_sec = seconds(runtstamp) # 1607569818.532117 sec
      ts_run, ts_now = uec.tstamps_run_and_now(int(trun_sec))

      nevrun = 0
      break_steps  = False

      for istep,step in enumerate(orun.steps()):
        nsteptot += 1

        metadic = json.loads(step_docstring(step)) if step_docstring is not None else {}

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

        if dbo is None:
           kwa['rms_hi'] = odet.raw._data_bit_mask - 10
           kwa['int_hi'] = odet.raw._data_bit_mask - 10

           dbo = uc.DataBlock(**kwa)
           dbo.runnum = runnum
           dbo.exp = expname
           dbo.ts_run, dbo.ts_now = ts_run, ts_now #uc.tstamps_run_and_now(env, fmt=uc.TSTAMP_FORMAT)

        break_events = False

        for ievt,evt in enumerate(step.events()):
          #print('Event %04d' % ievt, end='\r')
          sys.stdout.write('Event %04d\r' % ievt)
          nevrun += 1
          nevtot += 1

          if ievt < evskip:
              logger.debug('==== Ev:%04d is skipped --evskip=%d' % (ievt,evskip))
              continue
          elif ievt>0 and ievt == evskip:
              logger.info('Events < --evskip=%d are skipped' % evskip)

          raw = odet.raw.raw(evt)

          if raw is None:
              logger.info('==== Ev:%04d raw is None' % (ievt))
              continue

          if plotim==1:
             #nda = odet.raw.calib(evt)
             nda = odet.raw.raw(evt)
             img = odet.raw.image(evt, nda=nda)
             if flimg is None:
                flimg = ug.fleximage(img, arr=nda, h_in=8, w_in=16, nneg=1, npos=3)

             gr.set_win_title(flimg.fig, titwin='Event %d' % nevtot)
             flimg.update(img, arr=nda)
             gr.show(mode='DO NOT HOLD')

          nevsel += 1

          tsec = time()
          dt   = tsec - tdt
          tdt  = tsec
          if uc.selected_record(ievt+1, events):
              ss = 'run[%d] %d  step %d  events total/run/step/selected: %4d/%4d/%4d/%4d  time=%7.3f sec dt=%5.3f sec'%\
                   (irun, runnum, istep, nevtot, nevrun, ievt+1, nevsel, time()-t0_sec, dt)
              logger.info(ss)

          status = dbo.event(raw, nevtot)

          #print('XXX status:', status)

          if status:
              logger.info('requested statistics --nrecs=%d is collected - terminate loops' % dbo.nrecs)
              break_events = True
              break_steps = True
              break_runs = True
              break
          # End of event-loop

          if ievt > evstep-2:
              logger.info('Events in step > --evstep=%d next step' % evstep)
              break_events = True
              break

          if ievt > events-2:
              logger.info(ss)
              logger.info('\n==== Ev:%04d event loop is terminated --events=%d' % (ievt,events))
              break_events = True
              break

        if ievt < events: logger.info('==== Ev:%04d end of events in run %d step %d'%\
                                       (ievt, runnum, istep))

        if break_steps:
          logger.info('terminate_steps')
          break # break step loop

      if break_runs:
        logger.info('terminate_runs')
        break # break run loop

    kwa_summary = {\
                   'repoman': repoman,
                   'exp': expname,
                   'det': detname,
                   'dettype': dict_det['dettype'],
                   'deploy': deploy,
                   'dirrepo': dirrepo,
                   'dirmode': dirmode,
                   'filemode': filemode,
                   'tstamp': dic_run['tstamp_run'],
                   'tsshort': dic_run['tstamp_run'],
                   'run_orig': runnum,
                   'uniqueid': dict_det['uniqueid'],
                   'segment_ids': dict_det['segment_ids'],
                   'gainmode': metadic['gain_mode'],
                   }

    logger.info('kwa_summary:%s' % uec.info_dict(kwa_summary))

    summary(dbo, **kwa_summary)
#    dic_consts_tot[gainmode] = dic_consts
    del(dbo)

    if plotim==1: gr.show()

#    gainmodes = [k for k in dic_consts_tot.keys()]
#    logger.info('constants'\
#               +'\n  created  for gain modes: %s' % str(gainmodes)\
#               +'\n  expected for gain modes: %s' % str(odet.raw._gain_modes))

#    ctypes = ('pedestals', 'pixel_rms', 'pixel_status', 'pixel_max', 'pixel_min')
#    gmodes = odet.raw._gain_modes #  or gainmodes
#    kwa_depl['shape_as_daq'] = odet.raw._shape_as_daq()
#    kwa_depl['exp'] = expname
#    kwa_depl['det'] = detname
#    kwa_depl['run_orig'] = runnum

#    deploy_constants(ctypes, gmodes, **kwa_depl)

#    logger.debug('run/step/event loop is completed')
#    repoman.logfile_save()

    #sys.exit('TEST EXIT see commented deploy_constants')


def summary(dbo, **kwa):

    block  = dbo.block
    evnums = dbo.evnums
    logger.info('block summary: \n  %s\n  %s\n' % (
                info_ndarr(block, 'block', first=0, last=5),\
                info_ndarr(evnums, 'evnums', first=0, last=5),\
    ))

    ctypes = ('pixel_max', 'pixel_min')
    consts = arr_max, arr_min = dbo.max_min()

    logger.info('evaluated constants: \n  %s\n  %s' % (
        info_ndarr(arr_max, 'arr_max', first=0, last=5),\
        info_ndarr(arr_min, 'arr_min', first=0, last=5)))

    dic_consts = dict(zip(ctypes, consts))

    #sys.exit('TEST EXIT')
    save_constants_in_repository(dic_consts, **kwa)

#    gainmode = gain_mode(odet, metadic, istep) # nsteptot)
#    kwa_depl = add_metadata_kwargs(orun, odet, **kwa)
#    kwa_depl['gainmode'] = gainmode
#    kwa_depl['repoman'] = repoman
#    #kwa_depl['segment_ids'] = odet.raw._segment_ids()

#    logger.info('kwa_depl:\n%s' % info_dict(kwa_depl, fmt='  %12s: %s', sep='\n'))
#    #sys.exit('TEST EXIT')

#EOF
