from psana.psexp import mode, StepHistory, repack_for_bd, repack_with_step_dg, repack_with_mstep_dg, PacketFooter
from psana.psexp import EventBuilderManager, TransitionId, Events
from psana.psexp.run import RunLegion
from psana.psexp import Step
from psana import dgram
import numpy as np
import time
import logging
logger = logging.getLogger(__name__)

evt_kinds = {
    0: "ClearReadout",
    1: "Reset",
    2: "Configure",
    3: "Unconfigure",
    4: "BeginRun",
    5: "EndRun",
    6: "BeginStep",
    7: "EndStep",
    8: "Enable",
    9: "Disable",
    10: "SlowUpdate",
    11: "Unused_11",
    12: "L1Accept",
    13: "NumberOf",
}

pygion = None
if mode == 'legion':
    import pygion
    import sys
    from pygion import task, RW, RO, WD, Partition, Ipartition, Region, Ispace, Domain, Reduce
else:
    # Nop when not using Legion
    def task(fn=None, **kwargs):
        if fn is None:
            return lambda fn: fn
        return fn
    RO=True
    WD=True
    def Reduce(r):
        pass

run_objs = []
# builds batches and launches eb_tasks
# Use futures to track task completion for next set of batches
# Batches are patched to reflect missing SU datagrams
class LSmd0(object):
    """ Sends blocks of smds to eb nodes
    Identifies limit timestamp of the slowest detector then
    sends all smds within that timestamp.
    """
    def __init__(self, configs, smdr_man, dsparms):
        self.smdr_man = smdr_man
        self.configs = configs
        # Collecting Smd0 performance using prometheus
        self.c_sent = dsparms.prom_man.get_metric('psana_smd0_sent')
        self.step_hist = StepHistory(2, len(self.configs))

    """ support separate chunks for step and smds
    """
    def get_region_step_smd_chunk(self):
        # pack_smds and step separately
        pack_smds = {}
        pack_steps = {}

        # internally smdreader i.e. smdr_man.smdr
        # keeps track of start,step,buffer position
        # for this chunk
        for i_chunk in self.smdr_man.chunks():
            # Check missing steps: assume only single eb
            # Initially returns empty views
            # Next update (via extend_buffers_state) will record new transition
            # history
            missing_step_views = self.step_hist.get_buffer(1)
            step_views = [self.smdr_man.smdr.show(i, step_buf=True)
                          for i in range(self.smdr_man.n_files)]
            # append the new step view to the end of the buffer
            extend_buffers = self.step_hist.extend_buffers_state(step_views,1)
            # pack only the buffer without steps
            # pack step views only if there are missing steps
            # add those to legion's step region
            eb_id = 1
            step_only = 1
            if extend_buffers:
                pack_steps[1] = self.smdr_man.smdr.repack_parallel(missing_step_views,
                                                                   eb_id, step_only)
            else:
                pack_steps[1] = bytearray()
            pack_smds[1] = self.smdr_man.smdr.repack_only_buf(eb_id)
            yield pack_smds[1], pack_steps[1]

        # check final missing steps
        missing_step_views = self.step_hist.get_buffer(1)
        pack_steps[1] = self.smdr_man.smdr.repack_parallel(missing_step_views, 1, 1)
        yield bytearray(), pack_steps[1]

current_dgrams = 0
def eb_debug_batches(idx, smd_batch, cnt):
    run = run_objs[idx]
    pf = PacketFooter(view=smd_batch, num_views=cnt)
    for j, chunks in enumerate(pf.split_multiple_packets()):
        offsets = [0] * pf.n_packets
        logger.debug(f'n_packets={pf.n_packets}, partition[{j}]')
        for i, chunk in enumerate(chunks):
            logger.debug(f'----File %d----' % (i))
            while offsets[i] < pf.get_size(i):
                # Creates a dgram from this chunk at the given offset.
                d = dgram.Dgram(view=chunk, config=run.configs[i], offset=offsets[i])
                logger.debug(f'timestamp: %s : size: %d %s' % (str(d.timestamp()), d._size, evt_kinds[d.service()]))
                offsets[i] += d._size

# debug task that logs all the datagrams
@task(privileges=[RO])
def eb_task_debug_multiple(r, idx, smd_batch, cnt):
    logger.debug(f'EB_Task_With_Multiple_Region_DEBUG: Subregion has volume %s extent %s bounds %s' % (
        r.ispace.volume, r.ispace.domain.extent, r.ispace.bounds))
    if smd_batch:
        logger.debug(f'--------------L1Accept Dgrams:---------------')
        eb_debug_batches(idx, smd_batch, 1)
    if cnt:
        logger.debug(f'-------------Transition Dgrams--------------')
        logger.debug(f'dgram_partitions = {cnt}')
        eb_debug_batches(idx, bytearray(r.x), cnt)

def smd_batches_with_transitions(smd_batch, run, r, num_dgrams):
    eb_man = EventBuilderManager(smd_batch, run.configs, run.dsparms, run)
    batches = {}
    # check for final transition batch
    if smd_batch is None or len(smd_batch)==0 and len(r.x) != 0:
        batches[0] = repack_with_mstep_dg(smd_batch, bytearray(r.x),
                                          run.configs, num_dgrams)
        yield batches[0]

    for smd_batch_dict in eb_man.smd_batches():
        smd_batch, _ = smd_batch_dict[0]
        batches[0] = repack_with_mstep_dg(smd_batch,
                                         bytearray(r.x),
                                         run.configs, num_dgrams)
        yield batches[0]

def smd_batches_with_unique_transitions(smd_batch, run, r, num_dgrams, cur_dgrams):
    eb_man = EventBuilderManager(smd_batch, run.configs, run.dsparms, run)
    batches = {}
    # check for final transition batch
    if smd_batch is None or len(smd_batch)==0 and len(r.x) != 0:
        batches[0] = repack_with_mstep_dg(smd_batch, bytearray(r.x),
                                          run.configs, num_dgrams-cur_dgrams)
        yield batches[0]

    else:
        for smd_batch_dict in eb_man.smd_batches():
            smd_batch, _ = smd_batch_dict[0]
            batches[0] = repack_with_mstep_dg(smd_batch,
                                              bytearray(r.x),
                                              run.configs,num_dgrams-cur_dgrams)
            yield batches[0]

def smd_batches_without_transitions(smd_batch, run):
    eb_man = EventBuilderManager(smd_batch, run.configs, run.dsparms, run)
    batches = {}
    for smd_batch_dict in eb_man.smd_batches():
        smd_batch, _ = smd_batch_dict[0]
        yield smd_batch

start_timestamp = {}
end_timestamp = {}

def perform_callback(evt, run, redv):
    global start_timestamp
    global end_timestamp
    if evt.service() == TransitionId.L1Accept:
        if run.event_fn:
            run.event_fn(evt, run.det)
        if run.reduc:
            run.reduc_fn(redv, evt, run.det)
    elif evt.service() == TransitionId.BeginStep and run.step_begin_fn:
        if evt.timestamp not in start_timestamp:
            start_timestamp[evt.timestamp] = 1
            run.step_begin_fn(evt, run.det)
    elif evt.service() == TransitionId.EndStep and run.step_end_fn:
        # make sure we don't see an end step before a begin step
        if evt.timestamp not in end_timestamp:
            assert len(end_timestamp) == len(start_timestamp)-1
            end_timestamp[evt.timestamp] = 1
            run.step_end_fn(evt, run.det)


@task(privileges=[RO])
def eb_task_with_multiple_region(r, smd_batch, idx, num_dgrams):
    ''' log the datagrams
    eb_task_debug_multiple(r, idx, smd_batch, num_dgrams)
    '''
    run = run_objs[idx]
    global current_dgrams
    # this is only processing non transition events, add unique transitions
    for batch in smd_batches_with_unique_transitions(smd_batch, run, r, num_dgrams,
                                                     current_dgrams):
        for evt in batch_all_events(batch, run):
            perform_callback(evt,run,None)
    assert current_dgrams <= num_dgrams

    if current_dgrams < num_dgrams:
        current_dgrams = num_dgrams

def eb_reduc(r, redc, smd_batch, idx, num_dgrams):
    ''' log the datagrams
    eb_task_debug_multiple(r, idx, smd_batch, num_dgrams)
    '''
    logger.debug(f'EB_Task_With_Multiple_Region_Reduc: Subregion has volume %s extent %s bounds %s' % (
        r.ispace.volume, r.ispace.domain.extent, r.ispace.bounds))
    logger.debug(f'EB_Task_With_Multiple_Region_Reduc: Subregion Reduc has volume %s extent %s bounds %s' % (
        redc.ispace.volume, redc.ispace.domain.extent, redc.ispace.bounds))
    run = run_objs[idx]
    global current_dgrams
    for batch in smd_batches_with_unique_transitions(smd_batch, run, r, num_dgrams,
                                                     current_dgrams):
        for evt in batch_all_events(batch, run):
            perform_callback(evt, run, redc.rval)

    assert current_dgrams <= num_dgrams
    if current_dgrams < num_dgrams:
        current_dgrams = num_dgrams

# EB task with a region for transition datagrams
# and a reduction region
@task(privileges=[RO,Reduce('+')])
def eb_reduc_task_sum(r, redc, smd_batch, idx, num_dgrams):
    eb_reduc(r, redc, smd_batch, idx, num_dgrams)

@task(privileges=[RO,Reduce('-')])
def eb_reduc_task_minus(r, redc, smd_batch, idx, num_dgrams):
    eb_reduc(r, redc, smd_batch, idx, num_dgrams)

@task(privileges=[RO,Reduce('min')])
def eb_reduc_task_min(r, redc, smd_batch, idx, num_dgrams):
    eb_reduc(r, redc, smd_batch, idx, num_dgrams)

@task(privileges=[RO,Reduce('max')])
def eb_reduc_task_max(r, redc, smd_batch, idx, num_dgrams):
    eb_reduc(r, redc, smd_batch, idx, num_dgrams)

@task(privileges=[RO,Reduce('/')])
def eb_reduc_task_div(r, redc, smd_batch, idx, num_dgrams):
    eb_reduc(r, redc, smd_batch, idx, num_dgrams)

@task(privileges=[RO,Reduce('*')])
def eb_reduc_task_mult(r, redc, smd_batch, idx, num_dgrams):
    eb_reduc(r, redc, smd_batch, idx, num_dgrams)


@task(privileges=[WD])
def fill_task(r):
    pygion.fill(r, 'x', 0)

@task
def make_region_task(size):
    r = Region([size], {'x': pygion.int8})
    return r

@task(privileges=[WD])
def fill_data(r, data):
    logger.debug(f'Task fill_data: Subregion has volume %s extent %s bounds %s' % (
        r.ispace.volume, r.ispace.domain.extent, r.ispace.bounds))
    np.copyto(r.x,bytearray(data))


# Partition -> [P.ispace.volume, P.ispace.volume + len(step_data)-1]
# Fill the new partition with step_data
def fill_new_subregion(r, p, step_data):
    index_space = []
    start = p.ispace.volume
    size = len(bytearray(step_data))
    ip = Ipartition.pending(r.ispace, [1])
    index_space.append(Ispace([size], [start]))
    ip.union([0], index_space)
    p = Partition(r, ip)
    logger.debug(f'fill_new_subregion: Subregion has bounds %s' % (index_space[0].bounds))
    fill_data(p[0], bytearray(step_data))
    return p

# Partition -> [0,size_old-1] U [size_old, size_new-1]
def union_partitions(r, pold, pnew):
    index_space = []
    size_old = pold.ispace.volume
    size_new = pnew.ispace.volume
    ip = Ipartition.pending(r.ispace, [1])
    index_space.append(Ispace([size_old], [0]))
    index_space.append(Ispace([size_new], [size_old]))
    logger.debug(f'union_partitions: Subregion[0] has bounds %s' % (index_space[0].bounds))
    logger.debug(f'union_partitions: Subregion[1] has bounds %s' % (index_space[1].bounds))
    ip.union([0], index_space)
    p = Partition(r, ip)
    return p

# 1) check if new transition/step data exists
# 2) if True:
#      a) create new partition and fill subregion ->fill_new_subregion
#      b) merge old partition and new partition and return new merged partition -> union_partitions
def update_partition(r, p, step_data,cnt):
    if len(step_data) != 0:
        pnew = fill_new_subregion(r, p[0], step_data)
        punion = union_partitions(r, p[0], pnew[0])
        cnt=cnt+1
        return punion, cnt
    return p,cnt

def smd_chunks_steps(run):
    return run.ds.smd0.get_region_step_smd_chunk()

def perform_eb(r,p,smd_data,step_data,global_procs,pt,num_partitions,idx):
    if global_procs==1:
        pt=-1
    else:
        pt=pt+1
        pt=pt%(global_procs-1)
    # make a new partition only if additional transitions have occured in the chunk
    p, num_partitions = update_partition(r, p, step_data,num_partitions)
    eb_task_with_multiple_region(p[0], bytearray(smd_data), idx, num_partitions, point=pt+1)
    return p, pt, num_partitions


def perform_final_eb(p, idx, num_partitions, global_procs):
    if global_procs > 1:
        for i in range(global_procs-1):
            eb_task_with_multiple_region(p[0], bytearray(), idx, num_partitions, point=i+1)

def perform_eb_reduc(r,p,smd_data,step_data,global_procs,pt,num_partitions,
                     idx,reduc_region,reduc_type):
    if global_procs==1:
        pt=-1
    else:
        pt=pt+1
        pt=pt%(global_procs-1)

    eb_reduc_task = {
        '+': eb_reduc_task_sum,
        '-': eb_reduc_task_minus,
        'min': eb_reduc_task_min,
        'max': eb_reduc_task_max,
        '/': eb_reduc_task_div,
        '*': eb_reduc_task_mult
    }
    reduc_task = eb_reduc_task.get(reduc_type, None)
    assert reduc_task !=  None
    # make a new partition only if additional transitions have occured in the chunk
    p, num_partitions = update_partition(r, p, step_data,num_partitions)
    reduc_task(p[0], reduc_region,
               bytearray(smd_data), idx, num_partitions,
               point=pt+1)
    return p, pt, num_partitions

# Use regions for transition data
# 1 partition  = [start:end]
def make_ipartition(r_ispace, start, end):
    colors = [1]
    index_spaces = []
    ip1 = Ipartition.pending(r_ispace, [1])
    index_spaces.append(Ispace([end-start],[start]))
    ip1.union([0], index_spaces)
    return ip1


def init_region_partition():
    r = make_region_task(sys.maxsize).get()
    ip = make_ipartition(r.ispace, 0, -1)
    p = Partition(r, ip)
    fill_task(p[0])
    return r, p

# This is the entry task for SMD0 with a Region for Transition Datagrams with multiple Partitions
@task(inner=True)
def run_smd0_with_region_task_multiple_psana2(idx):
    global_procs = pygion.Tunable.select(pygion.Tunable.GLOBAL_PYS).get()
    num_partitions=0
    point=-1
    r, p = init_region_partition()
    run = run_objs[idx]
    for smd_data, step_data in smd_chunks_steps(run):
        p, point, num_partitions = perform_eb(r,p,smd_data,step_data,global_procs,point,num_partitions,idx)
    # guarantees all ranks see all the transitions
    perform_final_eb(p, idx, num_partitions, global_procs)
    pygion.execution_fence(block=True)


@task(privileges=[RO])
def run_smd0_reduc_final_task(r, idx):
    run = run_objs[idx]
    run.reduc_final_fn(r.rval)

# perform the reduction operation
def perform_reduc_op(redc, idx):
    global_procs = pygion.Tunable.select(pygion.Tunable.GLOBAL_PYS).get()
    num_partitions=0
    point=-1
    r, p = init_region_partition()
    run = run_objs[idx]
    reduc_type = run.reduc_privileges

    for smd_data, step_data in smd_chunks_steps(run):
        p, point, num_partitions = perform_eb_reduc(r,p,smd_data,step_data,
                                                    global_procs,point,num_partitions,
                                                    idx,redc,reduc_type)
    pygion.execution_fence(block=True)
    # callback for final reduction
    if run.reduc_final_fn:
        run_smd0_reduc_final_task(redc,idx,point=0)

# This is the entry task for SMD0 with
# a) a Region for Transition Datagrams with multiple Partitions
# b) Reduction Operation with a callback
@task(inner=True)
def run_smd0_reduc_task(idx):
    run = run_objs[idx]
    field_dict = {"rval":getattr(pygion, run.reduc_rtype)}
    reduc_region = Region(run.reduc_shape, field_dict)
    pygion.fill(reduc_region, 'rval', run.reduc_fill_val)
    perform_reduc_op(reduc_region, idx)

@task(privileges=[RO])
def dump_reduc(r):
    print('Dumping Reduc Values')
    print(r.rval)

def batch_events(smd_batch, run):
    batch_iter = iter([smd_batch, bytearray()])
    def get_smd():
        for this_batch in batch_iter:
            return this_batch
    events = Events(run.configs, run.dm, run.dsparms, 
            filter_callback=run.dsparms.filter, get_smd=get_smd)
    for i, evt in enumerate(events):
        logger.debug(f'evt[{i}] = {evt_kinds[evt.service()]}')
        if evt.service() != TransitionId.L1Accept: continue
        yield evt

def batch_all_events(smd_batch, run):
    batch_iter = iter([smd_batch, bytearray()])
    def get_smd():
        for this_batch in batch_iter:
            return this_batch
    events = Events(run.configs, run.dm, run.dsparms,
            filter_callback=run.dsparms.filter, get_smd=get_smd)
    for i, evt in enumerate(events):
        logger.debug(f'evt[{i}] = {evt_kinds[evt.service()]}')
        yield evt

run_to_process = []
def analyze(run, event_fn=None, det=None, step_begin_fn=None, step_end_fn=None):
    run.event_fn = event_fn
    run.step_begin_fn = step_begin_fn
    run.step_end_fn = step_end_fn
    run.det = det
    if pygion.is_script:
        num_procs = pygion.Tunable.select(pygion.Tunable.GLOBAL_PYS).get()
        bar = pygion.c.legion_phase_barrier_create(pygion._my.ctx.runtime, pygion._my.ctx.context, num_procs)
        pygion.c.legion_phase_barrier_arrive(pygion._my.ctx.runtime, pygion._my.ctx.context, bar, 1)
        global_task_registration_barrier = pygion.c.legion_phase_barrier_advance(pygion._my.ctx.runtime, pygion._my.ctx.context, bar)
        pygion.c.legion_phase_barrier_wait(pygion._my.ctx.runtime, pygion._my.ctx.context, bar)
        run_objs.append(run)
        if run.reduc:
            return run_smd0_reduc_task(len(run_objs)-1,point=0)
        else:
            return run_smd0_with_region_task_multiple_psana2(len(run_objs)-1,point=0)
    else:
        run_objs.append(run)
    

if pygion is not None and not pygion.is_script:
    @task(top_level=True)
    def legion_main():
        for i, _ in enumerate(run_objs):
            if run.reduc:
                run_smd0_reduc_task(i,point=0)
            else:
                run_smd0_with_region_task_multiple_psana2(i,point=0)
