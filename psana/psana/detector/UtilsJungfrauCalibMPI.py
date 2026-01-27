"""
:py:class:`UtilsJungfrauCalibMPI`
===================================
The same as UtilsJungfrauCalib, but uses MPI in the event loop

Usage::
    from psana.detector.UtilsJungfrauCalibMPI import *

phase 0: check data
datinfo -k exp=mfx100848724,run=49 -d jungfrau

phase 1 ONLY: USE --nrecs == --nrecs1, e.g.:
jungfrau_dark_proc -k exp=mfx100848724,run=49 -d jungfrau -o ./work1 --stepnum 0 --nrecs 50 --nrecs1 50

phase 2 with MPI:

mpirun -n 4 jungfrau_dark_proc -k exp=mfx100848724,run=49 -d jungfrau -o ./work1 --stepnum 0 --nrecs 100 --nrecs1 0

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2026-01-08 by Mikhail Dubrovin
"""

import sys
from psana.detector.UtilsJungfrauCalib import *

from psana.detector.Utils import get_hostname
import psutil
hostname = get_hostname()
cpu_num = psutil.Process().cpu_num()

print('sys.argv %s' % sys.argv)

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
s_rsch = 'rank:%03d/%03d cpu:%03d host:%s' % (rank, size, cpu_num, hostname)

if rank==0: print('start DarkProcJungfrauMPI for %s' % s_rsch)

#sys.exit('TEST EXIT')

class DarkProcJungfrauMPI(DarkProcJungfrau):
    """Extends DarkProcJungfrau for MPI"""
    def __init__(self, **kwa):
        logger.info('__init__ for %s' % s_rsch)
        DarkProcJungfrau.__init__(self, **kwa)

    def add_event(self, raw, irec):
        logger.info('add_event for %s' % s_rsch)
        DarkProcJungfrau.add_event(self, raw, irec)

    def summary(self):
        logger.info('summary for %s' % s_rsch)
        ompi = self.ompi
        arr_sta    = ompi.bor(self.arr_sta)
        arr_sum0   = ompi.sum(self.arr_sum0)
        arr_sum1   = ompi.sum(self.arr_sum1)
        arr_sum2   = ompi.sum(self.arr_sum2)
        arr_max    = ompi.max(self.arr_max)
        arr_min    = ompi.min(self.arr_min)
        sta_int_lo = ompi.sum(self.sta_int_lo)
        sta_int_hi = ompi.sum(self.sta_int_hi)

        if rank==0:
            self.arr_sta    = arr_sta
            self.arr_sum0   = arr_sum0
            self.arr_sum1   = arr_sum1
            self.arr_sum2   = arr_sum2
            self.arr_max    = arr_max
            self.arr_min    = arr_min
            self.sta_int_lo = sta_int_lo
            self.sta_int_hi = sta_int_hi

            DarkProcJungfrau.summary(self)



def jungfrau_dark_proc(parser):
    """jungfrau dark data processing for single (of 3) gain mode.
    """
    t0_sec = time()
    tdt = t0_sec

    args = parser.parse_args() # namespae of parameters
    kwargs = vars(args) # dict of parameters

    repoman = init_repoman_and_logger(parser=parser, **kwargs)
    kwargs['repoman'] = repoman

    detname = args.detname
    evskip  = args.evskip
    events  = args.events
    stepnum = args.stepnum
    stepmax = args.stepmax
    evcode  = args.evcode
    segind  = args.segind
    igmode  = args.igmode
    dirrepo = args.dirrepo

    dirmode  = kwargs.get('dirmode',  0o2775)
    filemode = kwargs.get('filemode', 0o664)
    group    = kwargs.get('group', 'ps-users')

    s = 'DIC_GAIN_MODE {<name> : <number>}'
    for k,v in DIC_GAIN_MODE.items(): s += '\n%16s: %d' % (k,v)
    logger.info(s)

    kwargs[batch_size] = 2 # this batch_size parameter may need to be tweaked
    ds, dskwargs = open_DataSource(**kwargs)
    ompi = ds.smalldata()

    dpo = None
    igm0 = None
    ievt = -1
    istep = -1
    nevtot = 0
    nevsel = 0
    nsteptot = 0
    ss = ''
    uniqueid = None
    dettype = None
    step_docstring = None
    terminate_runs = False

    #for irun, orun in enumerate(ds.runs()):
    if True: # USE SINGLE DARK RUN WITH 3 STEPS
        irun = 0
        orun = next(ds.runs())
        logger.info('\n%s Run %d %s' % (20*'=', orun.runnum, 20*'='))

        trun_sec = ups.seconds(orun.timestamp) # 1607569818.532117 sec
        ts_run, ts_now = ups.tstamps_run_and_now(trun_sec) #, fmt=uc.TSTAMP_FORMAT)

        odet = orun.Detector(detname, **kwargs)
        if dettype is None:
            dettype = odet.raw._dettype
            repoman.set_dettype(dettype)
            uniqueid = odet.raw._uniqueid
            logger.info('det.raw._uniqueid.split: %s' % str('\n'.join(uniqueid.split('_'))))

        logger.info('created %s detector object of type %s' % (detname, dettype))
        logger.info(ups.info_detector(odet, cmt='detector info:\n      ', sep='\n      '))

        try:
          step_docstring = orun.Detector('step_docstring')
        except:
          step_docstring = None

        terminate_steps = False
        nevrun = 0
        nnones = 0
        for istep, step in enumerate(orun.steps()):
            nsteptot += 1

            metadic = json.loads(step_docstring(step)) if step_docstring is not None else {}
            logger.info((100*'=') + '\n step_docstring ' +
                  'is None' if step_docstring is None else\
                  str(metadic))

            if stepmax is not None and nsteptot>stepmax:
                logger.info('==== Step:%02d loop is terminated, --stepmax=%d' % (nsteptot, stepmax))
                terminate_runs = True
                break

            if stepnum is not None:
                # process calibcycle stepnum ONLY if stepnum is specified
                if istep < stepnum:
                    logger.info('Skip step %d < --stepnum = %d' % (istep, stepnum))
                    continue
                elif istep > stepnum:
                    logger.info('Break further processing due to step %d > --stepnum = %d' % (istep, stepnum))
                    terminate_runs = True
                    break

            igm = igmode if igmode is not None else\
                  metadic['gainMode'] if step_docstring is not None\
                  else istep
            gmname = DIC_IND_TO_GAIN_MODE.get(igm, None)
            kwargs['gainmode'] = gmname
            logger.info('gain mode: %s igm: %d' % (gmname, igm))

            if dpo is None:
               kwargs['dettype'] = dettype
               dpo = DarkProcJungfrauMPI(**kwargs)
               dpo.runnum = orun.runnum
               dpo.exp = orun.expt
               dpo.ts_run, dpo.ts_now = ts_run, ts_now #uc.tstamps_run_and_now(env, fmt=uc.TSTAMP_FORMAT)
               dpo.detid = uniqueid
               dpo.gmindex = igm
               dpo.gmname = gmname
               dpo.odet = odet
               dpo.orun = orun
               dpo.ompi = ompi

               igm0 = igm

            if igm != igm0:
               logger.warning('event for wrong gain mode index %d, expected %d' % (igm, igm0))

            logger.info('%s\n== begin step %d gain mode "%s" index %d' % (120*'-', istep, gmname, igm))

            for ievt, evt in enumerate(step.events()):

                nevrun += 1
                nevtot += 1

                if ievt<evskip:
                    s = 'skip event %d < --evskip=%d' % (ievt, evskip)
                    #print(s, end='\r')
                    if (selected_record(ievt+1, events) and ievt<evskip-1)\
                    or ievt==evskip-1: logger.info(s)
                    continue

                if nevtot>=events:
                    print()
                    logger.info('break at nevtot %d == --events=%d' % (nevtot, events))
                    terminate_steps = True
                    terminate_runs = True
                    break

                raw = odet.raw.raw(evt)
                if raw is None:
                    logger.debug('det.raw.raw(evt) is None in event %d' % ievt)
                    nnones =+ 1
                    continue

                raw = (raw if segind is None else raw[segind,:]) # NO & M14 here

                nevsel += 1

                tsec = time()
                dt   = tsec - tdt
                tdt  = tsec
                if selected_record(ievt+1, events):
                    ss = 'run[%d] %d  step %d  events total/run/step/selected/none: %4d/%4d/%4d/%4d/%4d  time=%7.3f sec dt=%5.3f sec'%\
                         (irun, orun.runnum, istep, nevtot, nevrun, ievt+1, nevsel, nnones, time()-t0_sec, dt)
                    logger.info(ss)

                if dpo is not None:
                    status = dpo.event(raw,ievt)
                    if status == 1:
                        logger.error('This option nrecs == nrecs1 should not be used with mpi')
                        terminate_runs = True
                        terminate_steps = True
                        break # evt loop
                    elif status == 2:
                        logger.info('requested statistics --nrecs=%d is collected - terminate loops' % args.nrecs)
                        #if ecm:
                        #    terminate_runs = True
                        #    terminate_steps = True
                        break # evt loop
                # End of event-loop

            print()
            ss = 'run[%d] %d  end of step %d  events total/run/step/selected: %4d/%4d/%4d/%4d'%\
                 (irun, orun.runnum, istep, nevtot, nevrun, ievt+1, nevsel)
            logger.info(ss)

            print('XXX begin ompi.done in %s' % s_rsch)
            ompi.done()
            print('XXX end ompi.done in %s' % s_rsch)

# code for NON-MPI
#            if args.nrecs == args.nrecs1 and status==1:
#                uc.save_block_results(dpo, orun, odet, **kwargs)
#            else:
#                uc.save_results_in_repository(dpo, orun, odet, **kwargs)

            uc.save_results_in_repository(dpo, orun, odet, **kwargs)
            dpo=None

            if terminate_steps:
                logger.info('terminate_steps')
                break
            # End of step-loop

        logger.info(ss)
        logger.info('run %d, number of steps processed %d' % (orun.runnum, istep+1))
        # End of run-loop

    logger.info('%s\ntotal consumed time = %.3f sec.' % (40*'_', time()-t0_sec))
    repoman.logfile_save()

# EOF

