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

def _env_is_true(name):
    return os.environ.get(name, '').lower() in ('1', 'true', 'yes', 'on')


def _stage2_timing_template():
    return {
        'rank': rank,
        'host': hostname,
        'run': None,
        'steps': 0,
        'events': 0,
        'raw_none': 0,
        'first_evt_s': -1.0,
        'ds_create_s': 0.0,
        'smd_create_s': 0.0,
        'next_run_s': 0.0,
        'det_create_s': 0.0,
        'event_loop_s': 0.0,
        'raw_s': 0.0,
        'dpo_event_s': 0.0,
        'core_add_event_s': 0.0,
        'add_mask_s': 0.0,
        'add_astype_s': 0.0,
        'add_gate_lo_s': 0.0,
        'add_gate_hi_s': 0.0,
        'add_gate_mask_s': 0.0,
        'add_sum0_s': 0.0,
        'add_sum1_s': 0.0,
        'add_square_s': 0.0,
        'add_sum2_s': 0.0,
        'add_sta_lo_s': 0.0,
        'add_sta_hi_s': 0.0,
        'add_max_s': 0.0,
        'add_min_s': 0.0,
        'bad_switch_s': 0.0,
        'summary_call_s': 0.0,
        'summary_total_s': 0.0,
        'summary_eval_s': 0.0,
        'neutral_init_s': 0.0,
        'red_irec_s': 0.0,
        'red_sum0_s': 0.0,
        'red_sum1_s': 0.0,
        'red_sum2_s': 0.0,
        'red_max_s': 0.0,
        'red_min_s': 0.0,
        'red_int_lo_s': 0.0,
        'red_int_hi_s': 0.0,
        'red_bad_switch_s': 0.0,
        'merger_s': 0.0,
        'deploy_s': 0.0,
        'save_s': 0.0,
        'total_s': 0.0,
    }


def _stage2_reduce_total(item):
    return sum(item[k] for k in (
        'red_irec_s', 'red_sum0_s', 'red_sum1_s', 'red_sum2_s',
        'red_max_s', 'red_min_s', 'red_int_lo_s', 'red_int_hi_s',
        'red_bad_switch_s'))


def _print_stage2_timing_summary(local):
    gathered = comm.gather(local, root=0)
    if not is_rank0:
        return

    lines = [
        '',
        'STAGE2_TIMING SUMMARY',
        'rank host run steps events raw_none first_evt ds_create smd_create next_run det_create '
        'event_loop raw dpo_event core_add_event bad_switch summary reduce_total '
        'summary_eval neutral_init merger deploy save total',
    ]
    for item in gathered:
        red_total = _stage2_reduce_total(item)
        lines.append('%04d %s %s %5d %6d %8d %.6f %.6f %.6f %.6f %.6f '
              '%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f' % (
              item['rank'], item['host'], str(item['run']),
              item['steps'], item['events'], item['raw_none'], item['first_evt_s'],
              item['ds_create_s'], item['smd_create_s'], item['next_run_s'],
              item['det_create_s'], item['event_loop_s'], item['raw_s'],
              item['dpo_event_s'], item['core_add_event_s'], item['bad_switch_s'],
              item['summary_call_s'], red_total, item['summary_eval_s'],
              item['neutral_init_s'], item['merger_s'], item['deploy_s'],
              item['save_s'], item['total_s']))

    total_events = sum(item['events'] for item in gathered)
    total_raw_none = sum(item['raw_none'] for item in gathered)
    max_run_init = max(item['next_run_s'] for item in gathered)
    max_event_loop = max(item['event_loop_s'] for item in gathered)
    max_raw = max(item['raw_s'] for item in gathered)
    max_dpo_event = max(item['dpo_event_s'] for item in gathered)
    max_core_add_event = max(item['core_add_event_s'] for item in gathered)
    max_bad_switch = max(item['bad_switch_s'] for item in gathered)
    max_summary = max(item['summary_call_s'] for item in gathered)
    max_reduce = max(_stage2_reduce_total(item) for item in gathered)
    max_total = max(item['total_s'] for item in gathered)
    loop_rate = 0.0 if max_event_loop <= 0 else total_events / max_event_loop
    raw_rate = 0.0 if max_raw <= 0 else total_events / max_raw
    dpo_event_rate = 0.0 if max_dpo_event <= 0 else total_events / max_dpo_event

    lines.extend((
        '',
        'AGGREGATE',
        'run_init_max_s=%.6f' % max_run_init,
        'loop_time_max_s=%.6f' % max_event_loop,
        'events_sum_all_ranks=%d' % total_events,
        'loop_rate_events_per_s=%.3f' % loop_rate,
        'raw_time_max_s=%.6f' % max_raw,
        'raw_rate_events_per_s=%.3f' % raw_rate,
        'dpo_event_time_max_s=%.6f' % max_dpo_event,
        'dpo_event_rate_events_per_s=%.3f' % dpo_event_rate,
        'core_add_event_time_max_s=%.6f' % max_core_add_event,
        'bad_switch_time_max_s=%.6f' % max_bad_switch,
        'summary_time_max_s=%.6f' % max_summary,
        'reduce_total_time_max_s=%.6f' % max_reduce,
        'raw_none_sum_all_ranks=%d' % total_raw_none,
        'total_time_max_s=%.6f' % max_total,
        'aggregate events_seen_by_main_loop=%d raw_none=%d max_event_loop=%.6fs '
          'max_raw=%.6fs max_dpo_event=%.6fs max_core_add_event=%.6fs '
          'max_bad_switch=%.6fs max_summary=%.6fs max_reduce=%.6fs max_total=%.6fs' % (
          total_events, total_raw_none, max_event_loop, max_raw, max_dpo_event,
          max_core_add_event, max_bad_switch, max_summary, max_reduce, max_total),
    ))

    add_event_keys = (
        'add_mask_s', 'add_astype_s', 'add_gate_lo_s', 'add_gate_hi_s',
        'add_gate_mask_s', 'add_sum0_s', 'add_sum1_s', 'add_square_s',
        'add_sum2_s', 'add_sta_lo_s', 'add_sta_hi_s', 'add_max_s', 'add_min_s')
    lines.extend((
        '',
        'ADD_EVENT_DETAIL SUMMARY',
        'rank host events mask astype gate_lo gate_hi gate_mask sum0 sum1 square sum2 sta_lo sta_hi max min',
    ))
    for item in gathered:
        lines.append('%04d %s %6d ' % (item['rank'], item['host'], item['events'])
                     + ' '.join('%.6f' % item[k] for k in add_event_keys))

    lines.extend((
        '',
        'ADD_EVENT_DETAIL AGGREGATE',
        'add_mask_time_max_s=%.6f' % max(item['add_mask_s'] for item in gathered),
        'add_astype_time_max_s=%.6f' % max(item['add_astype_s'] for item in gathered),
        'add_gate_lo_time_max_s=%.6f' % max(item['add_gate_lo_s'] for item in gathered),
        'add_gate_hi_time_max_s=%.6f' % max(item['add_gate_hi_s'] for item in gathered),
        'add_gate_mask_time_max_s=%.6f' % max(item['add_gate_mask_s'] for item in gathered),
        'add_sum0_time_max_s=%.6f' % max(item['add_sum0_s'] for item in gathered),
        'add_sum1_time_max_s=%.6f' % max(item['add_sum1_s'] for item in gathered),
        'add_square_time_max_s=%.6f' % max(item['add_square_s'] for item in gathered),
        'add_sum2_time_max_s=%.6f' % max(item['add_sum2_s'] for item in gathered),
        'add_sta_lo_time_max_s=%.6f' % max(item['add_sta_lo_s'] for item in gathered),
        'add_sta_hi_time_max_s=%.6f' % max(item['add_sta_hi_s'] for item in gathered),
        'add_max_time_max_s=%.6f' % max(item['add_max_s'] for item in gathered),
        'add_min_time_max_s=%.6f' % max(item['add_min_s'] for item in gathered),
    ))

    for line in lines:
        print(line, flush=True)
        logger.info(line)


class DarkProcJungfrauMPI(DarkProcJungfrau):
    """Extends DarkProcJungfrau for MPI"""
    def __init__(self, **kwa):
        logger.debug('__init__ for %s' % s_rsch)
        DarkProcJungfrau.__init__(self, **kwa)
        self.rank_sum = None

    def add_event(self, raw, irec):
        logger.debug('add_event for %s' % s_rsch)
        timing = getattr(self, '_stage2_timing', None)
        if timing is None:
            DarkProcJungfrau.add_event(self, raw, irec)
            return

        t0 = time()
        uc.DarkProc.add_event(self, raw, irec)
        timing['core_add_event_s'] += time() - t0

        t0 = time()
        self.add_statistics_bad_gain_switch(raw, irec)
        timing['bad_switch_s'] += time() - t0

    def _init_neutral_summary_state(self):
        """Mona - protection against hungry ranks.
           Ensure ranks with no selected events still join summary reductions."""
        if hasattr(self, 'arr_sum0'):
            return
        #uc.load_block_results(self, self.orun, self.odet, **self.kwa)
        self.init_proc()

    def summary(self):
        #logger.info(uc.info_ndarr(self.arr_sum0, 'XXX summary begin irec: %d for %s arr_sum0:' % (self.irec, s_rsch), first=0, last=5))
        timing = getattr(self, '_stage2_timing', None)
        t_summary0 = time()
        smd = self.smd
        local_irec = self.irec
        if self.irec == -1:
            logger.warning('HUNGRY RANK !!!! in summary %s' % s_rsch)
            t_neutral0 = time()
            self._init_neutral_summary_state()
            if timing is not None:
                timing['neutral_init_s'] += time() - t_neutral0
            local_irec = 0

        def timed_reduce(key, func, arg):
            if timing is None:
                return func(arg)
            t0 = time()
            value = func(arg)
            timing[key] += time() - t0
            return value

        irec       = timed_reduce('red_irec_s', smd.sum, local_irec)
#        print('XXX summary irec:', irec, ' self.irec:', self.irec)

        arr_sum0   = timed_reduce('red_sum0_s', smd.sum, self.arr_sum0)
        arr_sum1   = timed_reduce('red_sum1_s', smd.sum, self.arr_sum1)
        arr_sum2   = timed_reduce('red_sum2_s', smd.sum, self.arr_sum2)
        arr_max    = timed_reduce('red_max_s', smd.max, self.arr_max)
        arr_min    = timed_reduce('red_min_s', smd.min, self.arr_min)
        sta_int_lo = timed_reduce('red_int_lo_s', smd.sum, self.sta_int_lo)
        sta_int_hi = timed_reduce('red_int_hi_s', smd.sum, self.sta_int_hi)
        bad_switch = timed_reduce('red_bad_switch_s', smd.bor, self.bad_switch)

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

            t_eval0 = time()
            DarkProcJungfrau.summary(self)
            if timing is not None:
                timing['summary_eval_s'] += time() - t_eval0

        if timing is not None:
            timing['summary_total_s'] += time() - t_summary0


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
    """jungfrau dark data processing in mpi"""
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
    segind  = args.segind
    dirrepo = args.dirrepo
    save    = args.save
    deploy  = args.deploy

    dirmode  = kwargs.get('dirmode',  0o2775)
    filemode = kwargs.get('filemode', 0o664)
    group    = kwargs.get('group', 'ps-users')
    timing_enabled = _env_is_true('JUNGFRAU_DARK_PROC_STAGE2_TIMING')
    stage2_timing = _stage2_timing_template() if timing_enabled else None
    if timing_enabled and is_rank0:
        msg = 'STAGE2_TIMING enabled by JUNGFRAU_DARK_PROC_STAGE2_TIMING'
        print(msg, flush=True)
        logger.info(msg)

    storage.setattr_from_kwargs(**kwargs)

    if is_rank_sel:
      logger.info('%s sys.argv: %s' % (s_rsch, str(sys.argv)))
      s = '%s DIC_GAIN_MODE {<name> : <number>}' % s_rsch
      for k,v in DIC_GAIN_MODE.items(): s += '\n%16s: %d' % (k,v)
      logger.info(s)

    kwargs['batch_size'] = 1 # this batch_size parameter may need to be tweaked
    kwargs['info_xtc_files'] = is_rank0
    kwargs['smd_callback'] = filter_callback
    kwargs.setdefault('skip_calib_load', 'all')
    #kwargs['max_events'] = kwargs.get('events', 3000)
    t_ds0 = time()
    ds, dskwargs = open_DataSource(**kwargs)
    if stage2_timing is not None:
        stage2_timing['ds_create_s'] = time() - t_ds0
    #### kwargs['ds'] = ds
    kwargs['dskwargs'] = dskwargs
    logger.debug('on %s open DataSource as: %s' % (s_rsch, str(ds)))

    t_smd0 = time()
    smd = ds.smalldata()
    if stage2_timing is not None:
        stage2_timing['smd_create_s'] = time() - t_smd0

    dpo = None
    merger = None
    igm0 = None
    ievt = 0
    nevsel = 0
    ss = ''
    uniqueid = None
    dettype = None
    step_docstring = None
    terminate_runs = False

    #for irun, orun in enumerate(ds.runs()):
    t_run0 = time()
    orun = next(ds.runs())
    if stage2_timing is not None:
        stage2_timing['next_run_s'] = time() - t_run0
        stage2_timing['run'] = orun.runnum

    #### !!!!!!!!!!!!!!!!!!!!
    #### kwargs['orun'] = orun  #### !!!!!!!!!!!!!!!!!!!!
    #### !!!!!!!!!!!!!!!!!!!!

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
           t_det0 = time()
           odet = orun.Detector(detname, **kwargs)
           if stage2_timing is not None:
               stage2_timing['det_create_s'] += time() - t_det0
           #kwargs['odet'] = odet

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

        igm = metadic.get('gainMode', istep) if step_docstring is not None\
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
           dpo._stage2_timing = stage2_timing
           igm0 = igm

        if igm != igm0:
           logger.warning('event for wrong gain mode index %d, expected %d' % (igm, igm0))

        if is_rank_sel:
           logger.info('%s\n== begin step %d gain mode "%s" index %d on %s' % (120*'-', istep, gmname, igm, s_rsch))

        if stage2_timing is not None:
            stage2_timing['steps'] += 1
        t_event_loop0 = time()
        for ievt, evt in enumerate(step.events()):
            nevrun += 1
            if stage2_timing is not None and stage2_timing['first_evt_s'] < 0:
                stage2_timing['first_evt_s'] = time() - t0_sec

            t_raw0 = time()
            raw = odet.raw.raw(evt)
            if stage2_timing is not None:
                stage2_timing['raw_s'] += time() - t_raw0
            if raw is None:
                logger.debug('det.raw.raw(evt) is None in event %d' % ievt)
                nnones =+ 1
                if stage2_timing is not None:
                    stage2_timing['raw_none'] += 1
                continue

            raw = (raw if segind is None else raw[segind,:]) # NO & M14 here

            nevsel += 1
            if stage2_timing is not None:
                stage2_timing['events'] += 1

            tsec = time()
            dt   = tsec - tdt
            tdt  = tsec
            if ujc.selected_record(ievt+1, events):
                ss = '%s runnum:%d  step:%d  events run/step/selected/none: %4d/%4d/%4d/%4d  time=%7.3f sec dt=%5.3f sec'%\
                     (s_rsch, orun.runnum, istep, nevrun, ievt+1, nevsel, nnones, time()-t0_sec, dt)
                logger.info(ss)

            if dpo is not None:
                t_dpo_event0 = time()
                status = dpo.event(raw, ievt)
                if stage2_timing is not None:
                    stage2_timing['dpo_event_s'] += time() - t_dpo_event0
                if status == 1:
                    logger.error('This option nrecs == nrecs1 should not be used with mpi')
                elif status == 2:
                    logger.info('requested statistics --nrecs=%d is collected' % args.nrecs)
            # End of event-loop
        if stage2_timing is not None:
            stage2_timing['event_loop_s'] += time() - t_event_loop0

        ss = '%s runnum:%d end of step %d events run/step/selected: %4d/%4d/%4d  step time: %.3f sec'%\
             (s_rsch, orun.runnum, istep, nevrun, ievt+1, nevsel, time() - t0_sec_step)
        logger.info(ss)

        logger.info('%s smd.summary: %s' % (s_rsch, str(smd.summary)))

        if smd.summary:
            t_summary_call0 = time()
            dpo.summary()
            if stage2_timing is not None:
                stage2_timing['summary_call_s'] += time() - t_summary_call0
            if rank == dpo.rank_sum:
                is_rank_sum = True
                if True:
                    if merger is None:
                        merger = uc.MergerDarkArrays()
                    t_merger0 = time()
                    merger.add_arrs_for_gain_range(dpo)
                    if stage2_timing is not None:
                        stage2_timing['merger_s'] += time() - t_merger0
                    if igm == 2:
                        t_deploy0 = time()
                        resp = os.system('klist')
                        logger.info('>>>> save results in DB on %s\n  command klist:\n%s' % (s_rsch, str(resp)))
                        ujc.jungfrau_deploy_dark_direct(merger, orun, odet, **kwargs)
                        if stage2_timing is not None:
                            stage2_timing['deploy_s'] += time() - t_deploy0
                if save:
                    logger.info('begin save_results_in_repository in %s' % s_rsch)
                    t_save0 = time()
                    uc.save_results_in_repository(dpo, orun, odet, **kwargs)
                    if stage2_timing is not None:
                        stage2_timing['save_s'] += time() - t_save0

        #del(dpo)
        dpo = None

    smd.done()
    logger.info('SMD.DONE %s' % s_rsch)

    if stage2_timing is not None:
        stage2_timing['total_s'] = time() - t0_sec
        _print_stage2_timing_summary(stage2_timing)

    if is_rank_sum:
        logger.info('SUM %s total consumed time %.3f sec' % (s_rsch, time()-t0_sec))
        repoman.logfile_save()

# EOF
