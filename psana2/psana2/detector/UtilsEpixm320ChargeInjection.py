#!/usr/bin/env python
""" import psana.detector.UtilsEpixm320ChargeInjection as ueci
"""
import os
import sys
from time import time, sleep
import json

import psana2.detector.UtilsGraphics as ug
from psana2.detector.UtilsLogging import logging  # DICT_NAME_TO_LEVEL, init_stream_handler
import psana2.detector.UtilsCalib as uc
from psana2.detector.dir_root import DIR_REPO_EPIXM320
import psana2.detector.UtilsEpix10kaCalib as uec
from psana2.detector.UtilsEpixm320Calib import save_constants_in_repository
from psana2.detector.utils_psana import seconds, str_tstamp, info_run, info_detector, seconds,\
    dict_run, dict_detector, dict_datasource
from psana2.detector.NDArrUtils import info_ndarr, divide_protected, save_2darray_in_textfile, save_ndarray_in_textfile
#from psana2.detector.Utils import info_dict
from psana2.detector.RepoManager import init_repoman_and_logger
logger = logging.getLogger(__name__)

SCRNAME = sys.argv[0].rsplit('/')[-1]

create_directory = uec.create_directory
#        import psana.pyalgos.generic.Graphics as gr
gr = ug.gr
np = ug.np

B14 = 0o40000 # 16384
M14 = 0x7fff  # raw & M14
M15 = 0x7fff  # raw & M15

class DataBlock():
    """primitive data block accumulation w/o processing with slices"""
    def __init__(self, **kwa):
        self.kwa    = kwa
        self.nrecs  = kwa.get('nrecs',1000)
        #self.datbits= kwa.get('datbits', 0xffff) # data bits 0xffff - 16-bit mask for detector without gain bit/s
        self.aslice = None
        self.block  = None
        self.irec   = -1

    def set_asic_slice(self, aslice):
        """aslice = (asic, np.s_[:], np.s_[colbeg:colend])"""
        self.aslice = aslice
        self.irec   = -1
        logger.info('set asic slice for raw: %s' % str(aslice))

    def event(self, raw, evnum):
        """Switch between gain mode processing objects using igm index of the gain mode (0,1,2).
           - evnum (int) - event number
           DEP: - igm (int) - index of the gain mode in DIC_GAIN_MODE
        """
        logger.debug('event %d' % evnum)

        if raw is None: return False # block is not full

        if self.block is None :
           self.block=np.zeros((self.nrecs,)+tuple(raw.shape), dtype=raw.dtype)
           self.evnums=np.zeros((self.nrecs,), dtype=np.uint16)
           logger.info(info_ndarr(self.block,'created empty data block'))
           logger.info(info_ndarr(self.evnums,'and array for event numbers'))
           logger.info('raw.shape: %s' % str(raw.shape))   # raw.shape: (4, 192, 384)
           logger.info('aslice: %s' % str(self.aslice))
           logger.info(info_ndarr(raw[self.aslice], 'raw[aslice]'))

        if self.not_full():
            self.irec +=1
            irec, aslice = self.irec, self.aslice
            self.block[(irec,) + aslice] = raw[aslice]
            self.evnums[irec] = evnum
        return self.is_full()

    def is_full(self):
        return not self.not_full()

    def not_full(self):
        return self.irec < self.nrecs-1

    def max_min(self):
        return np.max(self.block, axis=0),\
               np.min(self.block, axis=0)


class DataBlockProc(DataBlock):
    """extendiing DataBlock with specific accumulation and processing methods"""

    def __init__(self, **kwa):
        DataBlock.__init__(self, **kwa)
        self.gmode   = kwa.get('gmode', 'N/A')
        self.plotim  = kwa.get('plotim', 0)
        self.figpref = kwa.get('figpref', 'figs/fig')

    def init_accumulation(self, istep):
        print('-- step %02d: init for gmode %s' % (istep, self.gmode))

    def summary(self, istep, gmode, cmt=''):
        print('-- %s step %02d: summary for gmode %s' % (cmt, istep, self.gmode))
        assert gmode==self.gmode, 'gain mode in summary %s difers rom init/collect %s' % (gmode, self.gmode)
        block = self.block & M15
        if self.plotim & 2: plot_block(block, figpref=self.figpref, gmode=self.gmode)
        if self.plotim & 4: graph_block(block, figpref=self.figpref, gmode=self.gmode)
        self._evaluate_constants()

    def _evaluate_constants(self):
        print('-- TBD evaluate_constants for gmode %s' % self.gmode)
        print(info_ndarr(self.block, 'data block'))


def plot_block(block, figpref=None, gmode='N/A'):
        flimg1 = None
        #logger.info(info_ndarr(self.evnums, 'evnums', last=128))
        logger.info(info_ndarr(block, 'data block'))
        nrecs, nasics, rows, cols = block.shape
        nasics = block.shape[1]
        for asic in range(nasics):
          #for irec in range(nrecs):
            irec = int(nrecs/4)
            nda = block[irec,asic,:,:]
            title = '%s rec: %03d asic: %d' % (gmode, irec, asic)
            logger.info(info_ndarr(nda, title))
            img = nda
            if flimg1 is None:
               flimg1 = ug.fleximage(img, arr=nda, h_in=8, w_in=16, amin=0, amax=40000)
            gr.set_win_title(flimg1.fig, titwin=title)
            gr.add_title_labels_to_axes(flimg1.axim, title=title, xlabel='columns', ylabel='rows', fslab=14, fstit=20, color='k')
            flimg1.update(img, arr=nda)
            gr.show(mode='DO NOT HOLD')
            #if irec == int(nrecs/4):
            fnm = '%s-img-asic%d-rec%03d-%s.png' % (figpref, asic, irec, gmode)
            gr.save_fig(flimg1.fig, fname=fnm, verb=True)
        gr.show()


def graph_block(block, figpref=None, gmode='N/A', ncbanks=6, nrbanks=4):
        flimg1 = None
        logger.info(info_ndarr(block, 'data block'))
        nrecs, nasics, rows, cols = shape0 = block.shape
        #block.shape = (nrecs, nasics, rows, ncbanks, int(cols/ncbanks))
        x = np.arange(0, nrecs, dtype=np.int16)
        logger.info(info_ndarr(x, 'x:'))
        nrows1 = int(rows/nrbanks)
        ncols1 = int(cols/ncbanks)

        for asic in range(nasics):
          fig = None
          nregs = 0
          for bc in range(ncbanks):
           c0 = nrows1*bc
           cslice = np.s_[c0:c0+ncols1]
           for br in range(nrbanks):
            nregs +=1
            r0 = nrows1*br
            rslice = np.s_[r0:r0+nrows1]
            y = np.median(block[:,asic,rslice,cslice], axis=(-2,-1))
            logger.info(info_ndarr(y, '%s median for asic: %d bc: %d br: %d' % (gmode, asic, bc, br)))
            if fig is None:
              fig, ax = gr.plotGraph(x,y, figsize=(10,10), window=(0.15, 0.10, 0.78, 0.86), pfmt='b-', lw=2)
              title = '%s asic:%d' % (gmode, asic)
              gr.set_win_title(fig, titwin=title)
              gr.add_title_labels_to_axes(ax, title=title, xlabel='record', ylabel='median intensity', fslab=14, fstit=20, color='k')
            else:
              ax.plot(x, y, linewidth=2)
          gr.show()
          gr.save_fig(fig, fname='%s-graph-asic%d-%s.png' % (figpref, asic, gmode), verb=True)
        #block.shape = shape0













def charge_injection(parser):

    args = parser.parse_args()
    kwa = vars(args)
    repoman = init_repoman_and_logger(parser=parser, **kwa)

    str_dskwargs = kwa.get('dskwargs', None)
    detname    = kwa.get('det', None)
    idx        = kwa.get('idx', 0)
    dirrepo    = kwa.get('dirrepo', DIR_REPO_EPIXM320)
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
    evstep     = kwa.get('evstep', 10000)
    evskip     = kwa.get('evskip', 0)
    logmode    = kwa.get('logmode', 'DEBUG')
    pixrc      = kwa.get('pixrc', None) # ex.: '23,123'
    nsigm      = kwa.get('nsigm', 8)
    deploy     = kwa.get('deploy', False)
    sslice     = kwa.get('slice', '0:,0:')
    figpref    = kwa.get('figpref', 'figs/fig')
    irun       = None
    exp        = None
    npmin      = 5
    nstep_peds = 0
    tsec_show  = 2
    tsec_show_end = 60
    step_docstring = None
    dfid_med = 7761 # THIS VALUE DEPENDS ON EVENT RATE -> SHOULD BE AUTOMATED

    dskwargs = uec.datasource_kwargs_from_string(str_dskwargs)
    #dskwargs.setdefault('detectors', [detname,'step_docstring'])
    logger.info('dskwargs:%s\n' % uec.info_dict(dskwargs))

    if plotim:
      figdir = figpref.rsplit('/')[0]
      assert os.path.exists(figdir), 'directory for figures %s does not exist' % figdir

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

      if expname is None: expname = orun.expt # would work in case of inpuut from filename
      if runnum is None: runnum = orun.runnum

      logger.info('\n\n\n==== run: %d exp: %s' % (runnum, expname))
      logger.info(info_run(orun, cmt='run info:\n    ', sep='\n    ', verb=3))

      odet = orun.Detector(detname)
      if dettype is None:
          dettype = odet.raw._dettype
          repoman.set_dettype(dettype)

      logger.info('created Detector object for %s' % detname)
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
      break_steps = False
      gmode_cur = None
      status = False

      for istep,step in enumerate(orun.steps()):
        nsteptot += 1

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

        metadic = json.loads(step_docstring(step)) if step_docstring is not None else {}
        logger.info('\n\n== step %1d docstring: %s' % (istep, str(metadic)))
        #== step 31 docstring: {'detname': 'epixm_0', 'scantype': 'chargeinj', 'events': 128, 'pulserStep': 8, 'bandStep': 48, 'nBandSteps': 8, 'gain_mode': 'User', 'asic': 3, 'startCol': 336, 'lastCol': 383, 'step': 31}
        asic   = metadic['asic']
        colbeg = metadic['startCol']
        colend = metadic['lastCol'] + 1
        aslice = (asic, np.s_[:], np.s_[colbeg:colend])

        evmax = metadic['events']
        gmode = metadic['gain_mode']
        if gmode != gmode_cur:
            if gmode_cur is not None:
                dbo.summary(istep, gmode_cur, cmt='Loop')
            del(dbo)
            dbo = None
            gmode_cur = gmode

        if dbo is None:
            kwa['rms_hi'] = odet.raw._data_bit_mask - 10
            kwa['int_hi'] = odet.raw._data_bit_mask - 10
            kwa['nrecs']  = metadic['events']
            kwa['gmode']  = gmode
            kwa['plotim'] = plotim
            kwa['figpref'] = figpref

            dbo = DataBlockProc(**kwa)
            dbo.runnum = runnum
            dbo.exp = expname
            dbo.ts_run, dbo.ts_now = ts_run, ts_now #uc.tstamps_run_and_now(env, fmt=uc.TSTAMP_FORMAT)
            dbo.init_accumulation(istep)

        dbo.set_asic_slice(aslice)

        ss = ''

        nevsel = 0
        nevnone = 0
        break_events = False

        for ievt,evt in enumerate(step.events()):
          #print('Event %04d' % ievt, end='\r')
          #sys.stdout.write('Event %04d' % ievt)
          nevrun += 1
          nevtot += 1

          if ievt < evskip:
              logger.debug('==== ievt:%04d is skipped --evskip=%d' % (ievt,evskip))
              continue
          elif ievt>0 and ievt == evskip:
              logger.info('Events < --evskip=%d are skipped' % evskip)

          raw = odet.raw.raw(evt)

          if raw is None:
              nevnone += 1
              #print('-- ievt:%04d raw is None' % ievt, end='\r')
              continue

          nevsel += 1

          if plotim & 1 and dbo.irec==int(dbo.nrecs/4):

             #nda = odet.raw.calib(evt)
             nda = odet.raw.raw(evt)
             img = odet.raw.image(evt, nda=nda)
             if flimg is None:
                flimg = ug.fleximage(img, arr=nda, h_in=8, w_in=16, amin=0, amax=50000)
             #title = 'istep %02d ievt+1 %04d nevtot %04d' % (istep, ievt+1, nevtot)
             title = 'istep %02d irec %03d gmode %s' % (istep, dbo.irec, gmode)
             gr.set_win_title(flimg.fig, titwin=title)
             flimg.update(img, arr=nda)
             gr.add_title_labels_to_axes(flimg.axim, title=title, xlabel='columns', ylabel='rows', fslab=14, fstit=20, color='k')
             gr.show(mode='DO NOT HOLD')
             flimg.save(fname='%s-img-istep%02d-rec%03d-%s.png' % (figpref, istep, dbo.irec, gmode))

          tsec = time()
          dt   = tsec - tdt
          tdt  = tsec
          #if uc.selected_record(ievt+1, events):
          if nevsel%10 == 0:
              ss = 'run[%d] %d  step %d  events total/run/step/selected/none: %4d/%4d/%4d/%4d/%4d  time=%7.3f sec dt=%5.3f sec'%\
                   (irun, runnum, istep, nevtot, nevrun, ievt+1, nevsel, nevnone, time()-t0_sec, dt)
              print(ss, end='\r')

          status = dbo.event(raw, nevtot)
          #print(info_ndarr(raw[aslice], 'raw[asic, rslice, cslice]'), end='\r')

          if status:
              logger.info('requested statistics --nrecs=%d is collected - terminate loops' % dbo.nrecs)
              break_events = True
              #break_steps = True
              #break_runs = True
              break
          # End of event-loop

          if ievt > evstep-2:
              logger.info('Events in step > --evstep=%d next step' % evstep)
              break_events = True
              break

          if ievt > events-2:
              logger.info(ss)
              logger.info('\n==== ievt:%04d event loop is terminated --events=%d' % (ievt,events))
              break_events = True
              break

        ss = 'run[%d]: %d  step %02d  events total/run/step/selected/none: %4d/%4d/%4d/%4d/%4d time=%7.3f sec dt=%5.3f sec'%\
                   (irun, runnum, istep, nevtot, nevrun, ievt+1, nevsel, nevnone, time()-t0_sec, dt)
        logger.info(ss)
        #logger.info('==== end of step %d events ievt+1=nevsel+none %4d=%d+%d'%\
        #            (istep, ievt+1, nevsel, nevnone))

        if break_steps:
          logger.info('terminate_steps')
          break # break step loop

      dbo.summary(istep, gmode_cur, cmt='Last') # for the last step

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

    #summary(dbo, **kwa_summary)

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
                info_ndarr(evnums, 'evnums', first=0, last=5)))

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
