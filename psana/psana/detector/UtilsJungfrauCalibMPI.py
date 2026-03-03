"""
:py:class:`UtilsJungfrauCalibMPI`
===================================
The same as UtilsJungfrauCalib, but uses MPI in the event loop

Usage::
    from psana.detector.UtilsJungfrauCalibMPI import *

phase 0: check data
datinfo -k exp=mfx100848724,run=49 -d jungfrau

phase 1 and 2 TOGETHER
jungfrau_dark_proc -k exp=mfx100848724,run=49 -d jungfrau -o ./work1 --nrecs 100 --nrecs1 50

phase 1 ONLY: USE --nrecs == --nrecs1, e.g.:
jungfrau_dark_proc -k exp=mfx100848724,run=49 -d jungfrau -o ./work1 --nrecs 50 --nrecs1 50

phase 2 with MPI:
mpirun -n 5 jungfrau_dark_proc -k exp=mfx100848724,run=49 -d jungfrau -o ./work1 --nrecs 100 --nrecs1 0

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2026-01-08 by Mikhail Dubrovin
"""
import logging

#from psana.detector.UtilsJungfrauCalib import *
import psana.detector.UtilsJungfrauCalib as ujc
os, sys, time, logging, uc, ups, json = ujc.os, ujc.sys, ujc.time, ujc.logging, ujc.uc, ujc.ups, ujc.json

DIC_GAIN_MODE, DIC_IND_TO_GAIN_MODE, DarkProcJungfrau, init_repoman_and_logger, open_DataSource =\
    ujc.DIC_GAIN_MODE, ujc.DIC_IND_TO_GAIN_MODE, ujc.DarkProcJungfrau, ujc.init_repoman_and_logger, ujc.open_DataSource

logger = logging.getLogger(__name__)


os.environ['PS_EB_NODES']='1'
os.environ['PS_SRV_NODES']='1'

import psutil
cpu_num = psutil.Process().cpu_num()

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
use_mpi = size > 1
is_rank0 = rank==0
is_rank_sel = rank==(size-2) # rank==3

from psana.detector.Utils import get_hostname
hostname = get_hostname()

s_rsch = 'rank:%03d/%03d-cpu:%03d-%s' % (rank, size, cpu_num, hostname)
if is_rank0: print('%s sys.argv: %s' % (s_rsch, sys.argv))

class DarkProcJungfrauMPI(DarkProcJungfrau):
    """Extends DarkProcJungfrau for MPI"""
    def __init__(self, **kwa):
        logger.debug('__init__ for %s' % s_rsch)
        DarkProcJungfrau.__init__(self, **kwa)
        self.rank_sum = None

    def add_event(self, raw, irec):
        logger.debug('add_event for %s' % s_rsch)
        DarkProcJungfrau.add_event(self, raw, irec)

    def summary(self):
        #logger.info(uc.info_ndarr(self.arr_sum0, 'XXX summary begin irec: %d for %s arr_sum0:' % (self.irec, s_rsch), first=0, last=5))

        smd = self.smd
        if self.irec  == -1: return
        irec       = smd.sum(self.irec)
#        print('XXX summary irec:', irec, ' self.irec:', self.irec)

        arr_sum0   = smd.sum(self.arr_sum0)
        arr_sum1   = smd.sum(self.arr_sum1)
        arr_sum2   = smd.sum(self.arr_sum2)
        arr_max    = smd.max(self.arr_max)
        arr_min    = smd.min(self.arr_min)
        sta_int_lo = smd.sum(self.sta_int_lo)
        sta_int_hi = smd.sum(self.sta_int_hi)
        bad_switch = smd.bor(self.bad_switch)

        #logger.info(uc.info_ndarr(arr_sum0, 'XXX summary end irec: %s for %s arr_sum0:' % (str(irec), s_rsch), first=0, last=5))

        if arr_sum0 is not None:
            self.irec       = irec[0]
            self.rank_sum   = rank
            self.arr_sum0   = arr_sum0
            self.arr_sum1   = arr_sum1
            self.arr_sum2   = arr_sum2
            self.arr_max    = arr_max
            self.arr_min    = arr_min
            self.sta_int_lo = sta_int_lo
            self.sta_int_hi = sta_int_hi
            self.bad_switch = bad_switch

            logger.info('begin evaluation of results in %s on reduction rank' % s_rsch)

            DarkProcJungfrau.summary(self)


class Storage:
    def __init__(self):
        self.isset = False

    def setattr_from_kwargs(self, keys=('nrecs', 'evskip', 'stepnum', 'stepmax'), **kwargs):
        """Sets self attributes from dict kwargs for tuple of keys"""
        if self.isset: return
        self.isset = True
        self.keys = keys
        d = ups.dict_filter(kwargs, list_keys=keys)
        for k in d.keys():
            setattr(self, k, d[k])
        logger.info('%s storage parameters: %s' % (s_rsch, self.info_pars()))

    def info_pars(self, sep=' ', fmt='%s:%s'):
        lst_pars = [fmt % (k, str(getattr(self, k, None))) for k in self.keys]
        return sep + sep.join(lst_pars)

storage = Storage()


def filter_callback(run):
    """Event filter using small data. Parameters passed via global storage"""
    logger.info('== filter_callback first call for %s' % s_rsch)
    for istep, step in enumerate(run.steps()):
        logger.info('== filter_callback begin step:%d for %s' % (istep, s_rsch))
        cond1 = True if storage.stepmax is None else istep < storage.stepmax
        cond2 = True if storage.stepnum is None else istep == storage.stepnum
        if cond1 and cond2:
            logger.info('== filter_callback stepnum:%s stepmax:%s evskip:%d nrecs:%d - begin event loop for step:%d for %s)' %\
                        (str(storage.stepnum), str(storage.stepmax), storage.evskip, storage.nrecs, istep, s_rsch))
            for ievt, evt in enumerate(step.events()):
                if  ievt > storage.evskip-1\
                and ievt < storage.nrecs:
                    #print('  = filter_callback yield for step:%d evt:%03d' % (istep, ievt))
                    yield evt


def jungfrau_dark_proc(parser):
    """switching between run with/wo mpi"""
    print('start jungfrau_dark_proc on %s use_mpi: %s' % (s_rsch, use_mpi))

    if use_mpi: jungfrau_dark_proc_mpi(parser)
    else:   ujc.jungfrau_dark_proc(parser)


def jungfrau_dark_proc_mpi(parser):
    """jungfrau dark data processing for single (of 3) gain mode.
    """
    t0_sec = time()
    tdt = t0_sec

    args = parser.parse_args() # namespae of parameters
    kwargs = vars(args) # dict of parameters

    repoman = init_repoman_and_logger(parser=parser, info_parser=is_rank0, **kwargs)
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
    deploy  = args.deploy

    dirmode  = kwargs.get('dirmode',  0o2775)
    filemode = kwargs.get('filemode', 0o664)
    group    = kwargs.get('group', 'ps-users')

    storage.setattr_from_kwargs(**kwargs)

    #sys.exit(0)

    if is_rank_sel:
      logger.info('%s sys.argv: %s' % (s_rsch, str(sys.argv)))
      s = '%s DIC_GAIN_MODE {<name> : <number>}' % s_rsch
      for k,v in DIC_GAIN_MODE.items(): s += '\n%16s: %d' % (k,v)
      logger.info(s)

    kwargs['batch_size'] = 1 # this batch_size parameter may need to be tweaked
    kwargs['info_xtc_files'] = is_rank0
    kwargs['smd_callback'] = filter_callback
    kwargs['max_events'] = kwargs.get('events', 3000)
    ds, dskwargs = open_DataSource(**kwargs)
    kwargs['ds'] = ds
    kwargs['dskwargs'] = dskwargs
    logger.debug('on %s open DataSource as: %s' % (s_rsch, str(ds)))

    smd = ds.smalldata()

    dpo = None
    merger = None
    igm0 = None
    ievt = 0
    #istep = -1
    nevsel = 0
    ss = ''
    uniqueid = None
    dettype = None
    step_docstring = None
    terminate_runs = False

    #for irun, orun in enumerate(ds.runs()):
    orun = next(ds.runs())
    kwargs['orun'] = orun
    logger.info('%s %s begin run %s %s' % (s_rsch, 20*'=', str(orun.runnum), 20*'='))

    terminate_steps = False
    nevrun = 0
    nnones = 0
    odet = None
    is_rank_sum = False

    for istep, step in enumerate(orun.steps()):
        logger.info('%s ==== begin step %d' % (s_rsch, istep))
        t0_sec_step = time()

        timestamp = getattr(orun, 'timestamp', None)
        trun_sec = ups.seconds(timestamp) # 1607569818.532117 sec
        ts_run, ts_now = ups.tstamps_run_and_now(trun_sec) #, fmt=uc.TSTAMP_FORMAT)

        # Detector is in the step/event loop to hide it from all ranks
        #d = ups.dict_filter(kwargs, list_keys=('accept_missing',))
        if odet is None:
           odet = orun.Detector(detname, **kwargs)
           kwargs['odet'] = odet

        if dettype is None:
            dettype = odet.raw._dettype
            repoman.set_dettype(dettype)
            uniqueid = odet.raw._uniqueid
            #logger.info('det.raw._uniqueid.split: %s' % str('\n'.join(uniqueid.split('_'))))

        if is_rank_sel:
            logger.info('%s created %s detector object of type %s' % (s_rsch, detname, dettype))
            logger.info(ups.info_detector(odet, cmt='detector info on %s:\n      ' % s_rsch, sep='\n      '))

        try:
          step_docstring = orun.Detector('step_docstring')
        except:
          step_docstring = None

        metadic = json.loads(step_docstring(step)) if step_docstring is not None else {}
        if is_rank_sel:
           logger.info((100*'=') + '\n step_docstring ' +
              'is None' if step_docstring is None else\
              str(metadic))

#        if stepmax is not None and nsteptot>stepmax:
#            logger.info('==== Step:%02d loop is terminated, --stepmax=%d' % (nsteptot, stepmax))
#            terminate_runs = True
#            break

#        if stepnum is not None:
#            # process calibcycle stepnum ONLY if stepnum is specified
#            if istep < stepnum:
#                logger.info('Skip step %d < --stepnum = %d' % (istep, stepnum))
#                continue
#            elif istep > stepnum:
#                logger.info('Break further processing due to step %d > --stepnum = %d' % (istep, stepnum))
#                terminate_runs = True
#                break

        igm = igmode if igmode is not None else\
              metadic['gainMode'] if step_docstring is not None\
              else istep
        gmname = DIC_IND_TO_GAIN_MODE.get(igm, None)
        kwargs['gainmode'] = gmname
        if is_rank_sel:
            logger.info('gain mode: %s igm: %d' % (gmname, igm))

        if dpo is None:
           kwargs['dettype'] = dettype
           kwargs['info_block_results'] = is_rank_sel
           dpo = DarkProcJungfrauMPI(**kwargs)
           dpo.runnum = orun.runnum
           dpo.exp = orun.expt
           dpo.ts_run, dpo.ts_now = ts_run, ts_now #uc.tstamps_run_and_now(env, fmt=uc.TSTAMP_FORMAT)
           dpo.detid = uniqueid
           dpo.gmindex = igm
           dpo.gmname = gmname
           dpo.odet = odet
           dpo.orun = orun
           dpo.smd = smd

           igm0 = igm

        if igm != igm0:
           logger.warning('event for wrong gain mode index %d, expected %d' % (igm, igm0))

        if is_rank_sel:
           logger.info('%s\n== begin step %d gain mode "%s" index %d on %s' % (120*'-', istep, gmname, igm, s_rsch))

        for ievt, evt in enumerate(step.events()):

            nevrun += 1

#            if ievt<evskip:
#                s = 'skip event %d < --evskip=%d' % (ievt, evskip)
#                #print(s, end='\r')
#                if (ujc.selected_record(ievt+1, events) and ievt<evskip-1)\
#                or ievt==evskip-1: logger.info(s)
#                continue

            #if nevtot>=events:
            #    print()
            #    logger.info('break at nevtot %d == --events=%d' % (nevtot, events))
            #    terminate_steps = True
            #    terminate_runs = True
            #    break

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
            if ujc.selected_record(ievt+1, events):
                ss = '%s runnum:%d  step:%d  events run/step/selected/none: %4d/%4d/%4d/%4d  time=%7.3f sec dt=%5.3f sec'%\
                     (s_rsch, orun.runnum, istep, nevrun, ievt+1, nevsel, nnones, time()-t0_sec, dt)
                logger.info(ss)

            if dpo is not None:
                status = dpo.event(raw, ievt)
                if status == 1:
                    logger.error('This option nrecs == nrecs1 should not be used with mpi')
#                    terminate_runs = True
#                    terminate_steps = True
#                    break # evt loop
                elif status == 2:
                    logger.info('requested statistics --nrecs=%d is collected' % args.nrecs)
                    #if ecm:
                    #    terminate_runs = True
                    #    terminate_steps = True
#                    break # evt loop
            # End of event-loop

        ss = '%s runnum:%d  end of step %d events run/step/selected: %4d/%4d/%4d  step time: %.3f sec'%\
             (s_rsch, orun.runnum, istep, nevrun, ievt+1, nevsel, time() - t0_sec_step)
        logger.info(ss)
#        print(s_rsch, ' runnum:', orun.runnum, '  end of step:', istep, ' events run/step/selected:', nevrun, ievt, nevsel)

        #if False: ####### TEST

        print('XXX %s smd.summary:' % s_rsch, smd.summary)

        if smd.summary:
            #if dpo.irec != -1:
            dpo.summary()
            if rank == dpo.rank_sum:
                is_rank_sum = True
#                if deploy:
    
                 if True:
                    if merger is None:
                        merger = uc.MergerDarkArrays()
                    merger.add_arrs_for_gain_range(dpo)
                    if igm ==2:
                        print('XXXXX TBD save results in DB')
                        ujc.jungfrau_deploy_dark_direct(merger, **kwargs)
                else:
                    logger.info('begin save_results_in_repository in %s' % s_rsch)
                    uc.save_results_in_repository(dpo, orun, dpo.odet, **kwargs)

        dpo=None

        #logger.info('%s ==== end step %d' % (s_rsch, istep))

#        if terminate_steps:
#            logger.info('terminate_steps in %s' % s_rsch)
#            break
        # End of step-loop

    logger.info('SMD.DONE in %s' % s_rsch)
    smd.done()

    if is_rank_sum:
        logger.info('SUM %s total consumed time %.3f sec' % (s_rsch, time()-t0_sec))
        repoman.logfile_save()

# EOF

