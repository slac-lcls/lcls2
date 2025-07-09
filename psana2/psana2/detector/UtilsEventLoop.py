"""
:py:class:`UtilsEventLoop` psana2 event loop
============================================

Usage::

    from psana2.detector.UtilsEventLoop import *
    #OR
import psana2.detector.UtilsEventLoop as uel

This software was developed for the LCLS project.
If you use all or part of it, please give an appropriate acknowledgment.

@date 2025-02-11
@author Mikhail Dubrovin
"""

import sys
import logging
from time import time
import json

import psana2
from psana2.detector.Utils import info_dict
import psana2.detector.utils_psana as up # import seconds, data_source_kwargs, info_detector #, datasource_kwargs_from_string
import psana2.detector.UtilsCalib as uc
import psana2.detector.RepoManager as rm
import psana2.detector.NDArrUtils as au #info_ndarr, save_2darray_in_textfile, save_ndarray_in_textfile # import divide_protected
logger = logging.getLogger(__name__)
from psana2.detector.UtilsEpixm320Calib import gain_mode, save_constants_in_repository

seconds, data_source_kwargs, info_detector = up.seconds, up.data_source_kwargs, up.info_detector

SCRNAME = sys.argv[0].rsplit('/')[-1]


def info_datasource(ds):
    xtc_path = getattr(ds, 'xtc_path', None)
    s = '\n  ds.xtc_path: %s' % str(xtc_path)
    if xtc_path is not None:
      s += '\n  ds.n_files: %s ' % str(ds.n_files)\
         + '\n  ds.xtc_files:\n  %s' % ('\n  '.join(ds.xtc_files))\
         + '\n  ds.xtc_ext: %s' % (str(ds.xtc_ext) if hasattr(ds,'xtc_ext') else 'N/A')\
         + '\n  ds.smd_files:\n  %s' % ('\n  '.join(ds.smd_files))
    s += '\n  ds.shmem: %s' % str(ds.shmem)\
       + '\n  ds.smalldata_kwargs: %s' % str(ds.smalldata_kwargs)\
       + '\n  ds.timestamps: %s' % str(ds.timestamps)\
       + '\n  ds.unique_user_rank: %s' % str(ds.unique_user_rank())\
       + '\n  ds.is_mpi: %s' % str(ds.is_mpi())\
       + '\n  ds.live: %s' % str(ds.live)\
       + '\n  ds.destination: %s' % str(ds.destination)
    return s


def message(msg='MUST BE REIMPLEMENTED, IF NEEDED', metname='', logmethod=logger.debug, fmt='METHOD %s %s'):
    """classname, methodname: self.__class__.__name__, sys._getframe().f_code.co_name"""
    logmethod(fmt % (metname, msg))


class EventLoop:

    def __init__(self, parser):
        self.parser = parser
        self.args = parser.parse_args()
        self.kwa = vars(self.args)
        logger.debug('dict: %s' % str(self.kwa))
        logger.info('input parameters:%s' % info_dict(self.kwa))

    def open_data_sourse(self):
        self.dskwargs = dskwargs = data_source_kwargs(**self.kwa)
        try: ds = psana.DataSource(**dskwargs)
        except Exception as err:
            logger.error('DataSource(**dskwargs) does not work for **dskwargs: %s\n    %s' % (dskwargs, err))
            sys.exit('EXIT - requested DataSource does not exist or is not accessible.')
        print(info_datasource(ds))
        self.ds = ds

    def detector(self, orun):
        detname, kwa = self.args.detname, self.kwa
        try: det = orun.Detector(detname, **kwa)
        except Exception as err:
            logger.error('Detector(%s, **kwa) does not work for **kwargs: %s\n    %s' % (detname, str(kwa), err))
            sys.exit('EXIT')
        print(info_detector(det, cmt='detector info\n    ', sep='\n    '))
        self.det = det

    def init_event_loop(self):
        message(metname=sys._getframe().f_code.co_name, logmethod=logger.warning)
        print('init_event_loop - dskwargs: %s detname: %s' % (str(self.dskwargs), self.detname))

    def begin_run(self):
        message(metname=sys._getframe().f_code.co_name, logmethod=logger.warning)
        print('begin_run expname: %s runnum: %s' % (self.expname, str(self.runnum)))

    def end_run(self):
        message(metname=sys._getframe().f_code.co_name, logmethod=logger.warning)

    def begin_step(self):
        message(metname=sys._getframe().f_code.co_name, logmethod=logger.warning)
        print('begin_step istep/nevtot: %d/%s' % (self.istep, str(self.metadic)))

    def end_step(self):
        message(metname=sys._getframe().f_code.co_name, logmethod=logger.warning)

    def proc_event(self, msgmaxnum=5):
        if   self.ievt  > msgmaxnum: return
        elif self.ievt == msgmaxnum:
            message(msg='STOP WARNINGS', metname='', logmethod=logger.warning)
        else:
            message(metname=sys._getframe().f_code.co_name, logmethod=logger.warning)
        #print('proc_event ievt/nevtot: %d/%d' % (self.ievt, self.nevtot))

    def summary(self):
        message(metname=sys._getframe().f_code.co_name, logmethod=logger.warning)

    def event_loop(self):

        args = self.parser.parse_args()
        defs = self.parser.parse_args([])
        self.kwa = kwa = vars(args)

        self.repoman = rm.init_repoman_and_logger(parser=self.parser, **kwa)

        s_dskwargs = kwa.get('dskwargs', None)
        detname = kwa.get('detname', None)
        stepnum = kwa.get('stepnum', None)
        stepmax = kwa.get('stepmax', 1000)
        s_steps = kwa.get('steps', None)
        steps   = None if s_steps is None else\
                  eval('(%s)' % s_steps) # conv. str like '5,6,7' > (5,6,7)
        evskip  = kwa.get('evskip', 0)
        events  = kwa.get('events', 20)
        s_aslice= kwa.get('aslice', None)
        events  = kwa.get('events', 20)
        self.aslice = None if s_aslice is None else\
                      eval('np.s_[%s]' % s_aslice)
        self.segind = kwa.get('segind', None)
        self.nrecs = kwa.get('nrecs', 10)
        shortname = None
        self.detname = detname

        self.dskwargs = dskwargs = up.datasource_kwargs_from_string(s_dskwargs)
        logger.info('DataSource kwargs:%s' % info_dict(dskwargs, fmt='  %12s: %s', sep='\n'))

        self.ds = ds = psana.DataSource(**dskwargs)

        t0_sec = time()
        tdt = t0_sec
        nevtot = 0
        nevsel = 0
        nsteptot = 0
        break_bw = 0 # 1/2/4 - runs/steps/events
        dettype = None
        odet = None
        self.status = None

        self.init_event_loop()

        expname = dskwargs.get('exp', None)
        runnum  = dskwargs.get('run', None)

        for irun,orun in enumerate(ds.runs()):

          if expname is None: expname = orun.expt
          if runnum is None: runnum = orun.runnum

          nevrun = 0
          logger.info('\n==== %02d run: %d exp: %s' % (irun, runnum, expname))
          logger.info(up.info_run(orun, cmt='run info:    ', sep='\n    ', verb=3))

          runtstamp = orun.timestamp       # 4193682596073796843 relative to 1990-01-01
          trun_sec = up.seconds(runtstamp) # 1607569818.532117 sec
          self.ts_run, self.ts_now = up.tstamps_run_and_now(int(trun_sec))

          logger.debug('  run.timestamp: %d' % runtstamp)
          logger.debug('  run unix epoch time %06f sec' % trun_sec)
          logger.debug('  run tstamp: %s' % self.ts_run)
          logger.debug('  now tstamp: %s' % self.ts_now)

          self.odet = odet = orun.Detector(detname)

          if dettype is None:
              dettype = odet.raw._dettype
              self.repoman.set_dettype(dettype)

          logger.info('created %s detector object' % detname)
          logger.info(up.info_detector(odet, cmt='  detector info:\n      ', sep='\n      '))

          try: step_docstring = orun.Detector('step_docstring')
          except Exception as err:
            logger.warning('run.Detector("step_docstring"):\n    %s' % err)
            #sys.exit('Exit processing due to missing info about dark data step.')
            step_docstring = None
          self.step_docstring = step_docstring

          segment_ids = odet.raw._segment_ids() #ue.segment_ids_det(odet)
          segment_inds = odet.raw._segment_numbers  #_segment_indices() #ue.segment_indices_det(odet)

          logger.debug('segment inds and ids in the detector\n  '+\
                       '\n  '.join(['seg:%02d id:%s' % (i,id) for i,id in zip(segment_inds,segment_ids)]))

          databits = kwa.get('databits', None) #0xffff)
          if databits is None: databits = odet.raw._data_bit_mask
          logger.info('databits: %d or %s or %s' % (databits, oct(databits), hex(databits)))

          self.irun    = irun
          self.orun    = orun
          self.expname = expname
          self.runnum  = runnum
          self.begin_run()

          for istep,step in enumerate(orun.steps()):
            nsteptot += 1

            if steps is not None:
              if not(nsteptot in steps):
                logger.info('==== Step:%02d is skipped, --steps=%s' % (nsteptot, str(steps)))
                continue

            self.istep = istep
            self.step = step
            self.metadic = metadic = json.loads(step_docstring(step)) if step_docstring is not None else {}

            self.begin_step()

            print('\n==== Begin step %1d ====' % istep)
            logger.info('Step %1d docstring: %s' % (istep, str(metadic)))
            ss = ''

            if istep >= stepmax:
                logger.info('==== Step:%02d loop is terminated --stepmax=%d' % (istep, stepmax))
                break_bw |= 2
                break
            elif stepnum is not None:
                if istep < stepnum:
                    logger.info('==== Step:%02d is skipped --stepnum=%d' % (istep, stepnum))
                    continue
                elif istep > stepnum:
                    logger.info('==== Step:%02d loop is terminated --stepnum=%d' % (istep, stepnum))
                    break_bw |= 2
                    break

            count_none = 0
            #break_bw &= ~4 # clear bit 4

            for ievt,evt in enumerate(step.events()):
              sys.stdout.write('Event %04d\r' % ievt)
              nevrun += 1
              nevtot += 1

              self.ievt   = ievt
              self.evt    = evt
              self.nevrun = nevrun
              self.nevtot = nevtot

              if ievt < evskip:
                  logger.debug('==== Ev:%04d is skipped --evskip=%d' % (ievt,evskip))
                  continue
              elif evskip>0 and (ievt == evskip):
                  logger.info('Events < --evskip=%d are skipped' % evskip)

              if ievt > events-1:
                  logger.info(ss)
                  logger.info('\n==== Ev:%04d event loop is terminated --events=%d' % (ievt,events))
                  break_bw |= 4
                  break

              raw = odet.raw.raw(evt)

              if raw is None:
                  count_none += 1
                  if uc.selected_record(count_none, events):
                      logger.info('==== Ev:%04d raw is None, counter: %d' % (ievt, count_none))
                  continue

              nevsel += 1

              tsec = time()
              dt   = tsec - tdt
              tdt  = tsec
              if uc.selected_record(ievt+1, events):
                  ss = 'run[%d] %d  step %d  events total/run/step/selected: %4d/%4d/%4d/%4d  time=%7.3f sec dt=%5.3f sec'%\
                       (irun, runnum, istep, nevtot, nevrun, ievt+1, nevsel, time()-t0_sec, dt)
                  logger.info(ss)

              self.proc_event()

              if self.status == 2:
                  #logger.info('requested statistics --nrecs=%d is collected - terminate loops' % nrecs)
                  logger.info('self.status == 2 - terminate event loop')
                  break_bw |= 0o7
                  break
              # End of event-loop

              #print('Ev:%04d break_bw: %s' % (ievt, bin(break_bw)))
              

            if ievt < events: logger.info('======== Ev:%04d end of events in run %d   step %d   counter of raw==None %d'%\
                                           (ievt, orun.runnum, istep, count_none))


            if break_bw & 2:
              logger.info('terminate_steps')
              self.end_step()
              break # break step loop

          if break_bw & 1:
            logger.info('terminate_runs')
            self.end_run()
            break # break run loop

        self.summary()

        logger.debug('run/step/event loop is completed')
        self.repoman.logfile_save()


class EventLoopTest(EventLoop):
    msgelt='ELT:'

    def __init__(self, parser):
        EventLoop.__init__(self, parser)

    def init_event_loop(self):
        message(msg=self.msgelt, metname=sys._getframe().f_code.co_name, logmethod=logger.info)
        print('init_event_loop - dskwargs: %s detname: %s' % (str(self.dskwargs), self.detname))
        #kwa['init_event_loop'] = 'OK'
        self.dpo = None
        self.status = None
        self.dic_consts_tot = {} # {<gain_mode>:{<ctype>:nda3d_shape:(4, 192, 384)}}
        self.kwa_depl = {}


    def begin_run(self):
        message(msg=self.msgelt, metname=sys._getframe().f_code.co_name, logmethod=logger.info)
        print('begin_run expname: %s runnum: %s' % (self.expname, str(self.runnum)))

    def end_run(self):
        message(msg=self.msgelt, metname=sys._getframe().f_code.co_name, logmethod=logger.info)

    def begin_step(self):
        message(msg=self.msgelt, metname=sys._getframe().f_code.co_name, logmethod=logger.info)
        print('begin_step istep/nevtot: %d/%s' % (self.istep, str(self.metadic)))

        dpo = self.dpo
        if dpo is None:
            odet = self.odet
            kwa = self.kwa
            kwa['rms_hi'] = odet.raw._data_bit_mask - 10
            kwa['int_hi'] = odet.raw._data_bit_mask - 10

            dpo = self.dpo = uc.DarkProc(**kwa)
            dpo.runnum = self.runnum
            dpo.exp = self.expname
            dpo.ts_run, dpo.ts_now = self.ts_run, self.ts_now #uc.tstamps_run_and_now(env, fmt=uc.TSTAMP_FOR

    def end_step(self):
        message(msg=self.msgelt, metname=sys._getframe().f_code.co_name, logmethod=logger.info)
        if True:
                odet = self.odet
                dpo = self.dpo
                dpo.summary()
                ctypes = ('pedestals', 'pixel_rms', 'pixel_status', 'pixel_max', 'pixel_min') # 'status_extra'
                arr_av1, arr_rms, arr_sta = dpo.constants_av1_rms_sta()
                arr_max, arr_min = dpo.constants_max_min()
                consts = arr_av1, arr_rms, arr_sta, arr_max, arr_min
                logger.info('evaluated constants: \n  %s\n  %s\n  %s\n  %s\n  %s' % (
                            au.info_ndarr(arr_av1, 'arr_av1', first=0, last=5),\
                            au.info_ndarr(arr_rms, 'arr_rms', first=0, last=5),\
                            au.info_ndarr(arr_sta, 'arr_sta', first=0, last=5),\
                            au.info_ndarr(arr_max, 'arr_max', first=0, last=5),\
                            au.info_ndarr(arr_min, 'arr_min', first=0, last=5)))
                dic_consts = dict(zip(ctypes, consts))
                gainmode = gain_mode(odet, self.metadic, self.istep) # nsteptot)\
                kwa_depl = self.kwa_depl
                kwa_depl = uc.add_metadata_kwargs(self.orun, odet, **kwa)
                kwa_depl['gainmode'] = gainmode
                kwa_depl['repoman'] = self.repoman
                longname = kwa_depl['longname'] # odet.raw._uniqueid
                if shortname is None:
                  shortname = uc.detector_name_short(longname)
                print('detector long  name: %s' % longname)
                print('detector short name: %s' % shortname)
                kwa_depl['shortname'] = shortname

                #kwa_depl['segment_ids'] = odet.raw._segment_ids()

                logger.info('kwa_depl:\n%s' % info_dict(kwa_depl, fmt='  %12s: %s', sep='\n'))
                #sys.exit('TEST EXIT')

                save_constants_in_repository(dic_consts, **kwa_depl)
                self.dic_consts_tot[gainmode] = dic_consts
                del(dpo)
                dpo=None

                print('==== End of step %1d ====\n' % istep)

    def proc_event(self, msgmaxnum=5):
        if   self.ievt  > msgmaxnum: return
        elif self.ievt == msgmaxnum:
            message(msg='STOP WARNINGS', metname='', logmethod=logger.warning)
        else:
            message(msg=self.msgelt, metname=sys._getframe().f_code.co_name, logmethod=logger.warning)

        #print('proc_event ievt/nevtot: %d/%d' % (self.ievt, self.nevtot))
        self.status = self.dpo.event(self.odet.raw.raw(self.evt), self.ievt)

    def summary(self):
        message(msg=self.msgelt, metname=sys._getframe().f_code.co_name, logmethod=logger.info)
        gainmodes = [k for k in self.dic_consts_tot.keys()]
        logger.info('constants'\
                   +'\n  created  for gain modes: %s' % str(gainmodes)\
                   +'\n  expected for gain modes: %s' % str(self.odet.raw._gain_modes))

        ctypes = ('pedestals', 'pixel_rms', 'pixel_status', 'pixel_max', 'pixel_min')
        gmodes = self.odet.raw._gain_modes #  or gainmodes
        kwa_depl = self.kwa_depl
        kwa_depl['shape_as_daq'] = None if self.odet.raw is None else self.odet.raw._shape_as_daq()
        kwa_depl['exp']          = self.expname
        kwa_depl['det']          = self.detname
        kwa_depl['run_orig']     = self.runnum

        #deploy_constants(ctypes, gmodes, **kwa_depl)


if __name__ == "__main__":

  def USAGE():
    import inspect
    return '\n  %s <TNAME>\n' % sys.argv[0].split('/')[-1]\
          +'\n  test dataset: datinfo -k exp=mfxdaq23,run=7,dir=/sdf/data/lcls/drpsrcf/ffb/MFX/mfxdaq23/xtc -d jungfrau'
    #+ '\n'.join([s for s in inspect.getsource(selector).split('\n') if "TNAME in" in s])

  def argument_parser():
    from argparse import ArgumentParser
    d_tname    = '0'
    d_dirrepo  = './work1' # DIR_REPO_JUNGFRAU
    d_dskwargs = 'exp=mfxdaq23,run=7' # dir=/sdf/data/lcls/drpsrcf/ffb/MFX/mfxdaq23/xtc
    d_detname  = 'jungfrau'
    d_loglevel = 'INFO' # 'DEBUG'
    d_subtest  = None
    d_segind   = None
    h_dirrepo  = 'non-default repository of calibration results, default = %s' % d_dirrepo
    h_tname    = 'test name, usually numeric number, default = %s' % d_tname
    h_dskwargs = '(str) dataset kwargs for DataSource(**kwargs), default = %s' % d_dskwargs
    h_detname  = 'detector name, default = %s' % d_detname
    h_subtest  = '(str) subtest name, default = %s' % d_subtest
    h_loglevel = 'logging level, one of %s, default = %s' % (', '.join(tuple(logging._nameToLevel.keys())), d_loglevel)
    h_segind   = 'segment index in det.raw.raw array to process, default = %s' % str(d_segind)
    parser = ArgumentParser(description='%s is a bunch of tests for annual issues' % SCRNAME, usage=USAGE())
    #parser.add_argument('tname',            default=d_tname,    type=str, help=h_tname)
    parser.add_argument('-o', '--dirrepo',  default=d_dirrepo,  type=str, help=h_dirrepo)
    parser.add_argument('-k', '--dskwargs', default=d_dskwargs, type=str, help=h_dskwargs)
    parser.add_argument('-d', '--detname',  default=d_detname,  type=str, help=h_detname)
    parser.add_argument('-L', '--loglevel', default=d_loglevel, type=str, help=h_loglevel)
    parser.add_argument('-s', '--subtest',  default=d_subtest,  type=str, help=h_subtest)
    parser.add_argument('-I', '--segind',   default=d_segind,   type=int, help=h_segind)
    return parser


if __name__ == "__main__":
    parser = argument_parser()
    args = parser.parse_args()
    #tname = args.tname
    print(80*'_')

#    kwa = vars(args)
#    STRLOGLEV = args.loglevel
#    INTLOGLEV = logging._nameToLevel[STRLOGLEV]
#    #logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d %(filename)s: %(message)s', level=INTLOGLEV)
    #print('XXX tname:', tname)

    #evl = EventLoop(parser)
    evl = EventLoopTest(parser)
    evl.open_data_sourse()
    evl.event_loop()

    sys.exit('End of %s' % SCRNAME)

# EOF
