"""
:py:class:`UtilsEventLoop` psana2 event loop
============================================

Usage::

    import psana.detector.UtilsEventLoop as uel
    o = uel.o
    # e.g.:
    message = uel.message

This software was developed for the LCLS project.
If you use all or part of it, please give an appropriate acknowledgment.

@date 2025-02-11
@author Mikhail Dubrovin
"""

import sys
import logging
from time import time
import json
import psana
import psana.detector.Utils as ut # info_dict
import psana.detector.utils_psana as up # import seconds, data_source_kwargs, info_detector #, datasource_kwargs_from_string
import psana.detector.RepoManager as rm
import psana.detector.NDArrUtils as au #info_ndarr, save_2darray_in_textfile, save_ndarray_in_textfile # import divide_protected
import psana.detector.UtilsEpixm320Calib as ue320 # gain_mode, save_constants_in_repository

seconds, data_source_kwargs, info_detector, info_datasource =\
    up.seconds, up.data_source_kwargs, up.info_detector, up.info_datasource
gain_mode, save_constants_in_repository =\
    ue320.gain_mode, ue320.save_constants_in_repository
info_dict = ut.info_dict

logger = logging.getLogger(__name__)
SCRNAME = sys.argv[0].rsplit('/')[-1]


def message(msg='MUST BE REIMPLEMENTED, IF NEEDED', metname='', logmethod=logger.debug, fmt='METHOD %s %s'):
    """classname, methodname: self.__class__.__name__, sys._getframe().f_code.co_name"""
    logmethod(fmt % (metname, msg))


class EventLoop:

    def __init__(self, parser):
        self.parser = parser
        self.args = parser.parse_args()
        self.kwa = vars(self.args)
        logger.info('input parameters:%s' % info_dict(self.kwa))
        self.repoman = rm.init_repoman_and_logger(parser=self.parser, **self.kwa)

    def open_data_sourse(self, logmeth=logger.debug):
        self.dskwargs = dskwargs = data_source_kwargs(**self.kwa)
        try: ds = psana.DataSource(**dskwargs)
        except Exception as err:
            logger.error('DataSource(**dskwargs) does not work for **dskwargs: %s\n    %s' % (dskwargs, err))
            sys.exit('EXIT - requested DataSource does not exist or is not accessible.')
        logmeth(info_datasource(ds))
        self.ds = ds

    def detector(self, orun, logmeth=logger.debug):
        detname, kwa = self.args.detname, self.kwa
        try: det = orun.Detector(detname, **kwa)
        except Exception as err:
            logger.error('Detector(%s, **kwa) does not work for **kwargs: %s\n    %s' % (detname, str(kwa), err))
            sys.exit('EXIT')
        logmeth(info_detector(det, cmt='detector info\n    ', sep='\n    '))
        self.det = det

    def init_event_loop(self):
        message(metname=sys._getframe().f_code.co_name, logmethod=logger.warning)
        logger.info('init_event_loop - dskwargs: %s detname: %s' % (str(self.dskwargs), self.detname))

    def begin_run(self):
        message(metname=sys._getframe().f_code.co_name, logmethod=logger.warning)
        logger.info('begin_run expname: %s runnum: %s' % (self.expname, str(self.runnum)))

    def end_run(self):
        message(metname=sys._getframe().f_code.co_name, logmethod=logger.warning)
        logger.info('end_run expname: %s runnum: %s' % (self.expname, str(self.runnum)))

    def begin_step(self):
        message(metname=sys._getframe().f_code.co_name, logmethod=logger.warning)
        logger.info('begin_step istep:%d metadic: %s' % (self.istep, str(self.metadic)))

    def end_step(self):
        message(metname=sys._getframe().f_code.co_name, logmethod=logger.warning)
        logger.info('end_step istep/nevtot: %d/%d' % (self.istep, self.nevtot))

    def proc_event(self, msgmaxnum=5):
        if   self.ievt  > msgmaxnum: return
        elif self.ievt == msgmaxnum:
            message(msg='STOP WARNINGS', metname='', logmethod=logger.warning)
        else:
            message(metname=sys._getframe().f_code.co_name, logmethod=logger.warning)
        #logger.info('proc_event ievt/nevtot: %d/%d' % (self.ievt, self.nevtot))

    def summary(self):
        message(metname=sys._getframe().f_code.co_name, logmethod=logger.warning)

    def event_loop(self):

        #args = self.parser.parse_args()
        #defs = self.parser.parse_args([])
        kwa = self.kwa

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
        #logmeth = kwa.get('logmeth', logger.info)
        self.aslice = None if s_aslice is None else\
                      eval('np.s_[%s]' % s_aslice)
        self.segind = kwa.get('segind', None)
        self.nrecs = kwa.get('nrecs', 10)
        shortname = None
        self.detname = detname
        

        self.dskwargs = dskwargs = up.datasource_kwargs_from_string(s_dskwargs)
        logger.info('DataSource kwargs:%s' % info_dict(dskwargs, fmt='  %12s: %s', sep='\n'))

        #self.ds = ds = psana.DataSource(**dskwargs)
        self.open_data_sourse()
        ds = self.ds

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

            logger.info('\n==== Begin step %1d ====' % istep\
                       +'\nStep %1d docstring: %s' % (istep, str(metadic)))
            ss = ''

            if stepmax is not None and istep >= stepmax:
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
                  if ut.selected_record(count_none, events):
                      logger.info('==== Ev:%04d raw is None, counter: %d' % (ievt, count_none))
                  continue

              nevsel += 1

              tsec = time()
              dt   = tsec - tdt
              tdt  = tsec
              if ut.selected_record(ievt+1, events):
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
            if ievt < events: logger.info('======== Ev:%04d end of events in run %d   step %d   counter of raw==None %d'%\
                                           (ievt, orun.runnum, istep, count_none))

            self.end_step()
            if break_bw & 2:
              logger.info('terminate_steps')
              break # break step loop

          self.end_run()
          if break_bw & 1:
            logger.info('terminate_runs')
            break # break run loop

        self.summary()

        logger.debug('run/step/event loop is completed')
        self.repoman.logfile_save()


if __name__ == "__main__":
    sys.exit('to test of this module try: detector/testman/test_%s' % SCRNAME)

# EOF
