"""One fresh MPI case, with untimed setup, strict gates and GPU-resident cube checks."""
import argparse
import subprocess
from contextlib import closing
import gc
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(1, str(HERE/'vendor/sim10_compute_cube_20261115'))
from support import require, write, affinity, EVENT_BYTES, histogram, PACKAGE, verify_inputs, normalize_pci
from cpu_plan import thread_snapshot
from runtime import activate, identity
from storage import cache, condition_cache, counters, evidence
from reference import load_constants, digest
from cube_check import inspect_cube
from host_memory import snapshot, demand, validate_host_memory
import reference
reference.SNAPSHOT = PACKAGE/'calibration'


def execute(arguments):
    rank = int(os.environ['SLURM_PROCID'])
    placement = json.loads(os.environ['SIM10_GPU_PLAN'])
    nodes=arguments.bds//4
    require(nodes in (1,2,4) and arguments.bds==4*nodes and arguments.batch==200, 'Topology/batch')
    require(len(placement['rank_uuids']) == arguments.bds + 2, 'GPU rank map')
    active_gpus = len(set(placement['rank_uuids'][2:]))
    require(active_gpus == arguments.bds and len(placement['hosts'])==nodes, 'Four GPUs per node')
    granted = subprocess.check_output(['nvidia-smi', '-i', os.environ['CUDA_VISIBLE_DEVICES'], '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip().splitlines()
    local=[r for r,h in enumerate(placement['rank_hosts']) if h==platform.node()]
    require(rank in local, 'Actual rank host')
    local_uuids=[placement['rank_uuids'][r] for r in local if r>=2]
    require(set(local_uuids)==set(granted) and len(local_uuids)==4, 'Local granted GPU set')
    node_leader=min(local)==rank
    require(placement['rank_uuids'][:2] == ['', ''], 'Helper GPU masks')
    require(all(value.startswith('GPU-') for value in placement['rank_uuids'][2:]), 'Physical UUID mapping')
    os.environ['CUDA_VISIBLE_DEVICES'] = placement['rank_uuids'][rank]
    chosen = affinity()
    require(all(os.environ.get(key) == value for key, value in dict(
        PS_PARALLEL='mpi', PS_FAKESTEP_FLAG='0', PS_JF_SHARE_DERIVED='0', PS_GEO_SHARE='0',
        KVIKIO_NTHREADS='4', KVIKIO_TASK_SIZE=str(1 << 20), KVIKIO_BOUNCE_BUFFER_SIZE=str(16 << 20)).items()),
        'Frozen environment controls')
    activate()
    from mpi4py import MPI
    from psana import DataSource
    from psana.gpu import GpuTask
    from psana.psexp.run import Run
    from stages import StageTask, CopyAudit, adapter
    import numpy as np
    comm = MPI.COMM_WORLD
    require(comm.rank == rank and comm.size == arguments.bds + 2, 'MPI roles')
    require('CRAY MPICH' in MPI.Get_library_version().upper(), 'Cray MPI')
    started = time.monotonic()
    host_memory = dict(startup=snapshot(), demand=demand(4, 4, 16))
    require(host_memory['startup']['mem_available_bytes'] >= 32 << 30, 'Insufficient host setup headroom')
    info = json.loads(Path(arguments.inputs).read_text())
    stamps = list(map(int, info['timestamps'][:arguments.events]))
    require(len(stamps) == arguments.events and stamps == sorted(set(stamps)), 'Global source identity')
    require(info['media'] == 'lustre', 'Qualified run4 Lustre')
    reference.SNAPSHOT=Path(info['calibration_snapshot'])
    if node_leader:
        verify_inputs(info)
    from instrument import bindings, budget_peaks
    handles_context = bindings(info['payload_records'])
    handles = handles_context.__enter__()
    budget_context = budget_peaks()
    peaks = budget_context.__enter__()
    if arguments.kind != 'acceptance':
        require(arguments.events == 100000*nodes and arguments.kind in ('full','anchor'), 'Frozen event count')
    else:
        require(arguments.events == 1003*nodes, 'Bounded tail fixture')
    constants, constant_records = ({}, {})
    if arguments.stage in ('EC', 'ECC'):
        constants, constant_records = load_constants()

    def inject(run):
        require(rank == 0, 'Base calibration loader must only run on smd0')
        run._calib_const = {'jungfrau': constants} if constants else {}
        run.dsparms.calibconst = run._calib_const

    adapter.register_reader()
    task = StageTask(arguments.stage, arguments.batch, stamps,
                     diagnostic=arguments.kind == 'pilot')
    device = None
    settings = {}
    if rank >= 2:
        import cupy as cp
        import kvikio
        import kvikio.defaults as defaults
        from task_runtime import kvikio_identity
        require(cp.cuda.runtime.getDeviceCount() == 1, 'One visible GPU per BD')
        require(kvikio.__version__.startswith('26.10.'), 'Pinned KvikIO version')
        require(kvikio_identity() == json.loads(Path(os.environ['SIM10_NATIVE_RECEIPT']).read_text())['kvikio_identity'],
                'KvikIO provenance changed')
        settings = dict(compat_mode=kvikio.CompatMode.ON, num_threads=4, task_size=1 << 20,
                        bounce_buffer_size=16 << 20, auto_direct_io_read=False)
        defaults.set(settings)
        require(all(defaults.get(key) == value for key, value in settings.items()), 'KvikIO effective settings')
        from psana.gpu.gpudgram import parser as gpu_parser
        for factory in (gpu_parser._walk_kernel, gpu_parser._locate_kernel, gpu_parser._init_locators_kernel, gpu_parser._locate_fields_kernel):
            factory().compile()
        device = dict(mask=os.environ['CUDA_VISIBLE_DEVICES'], pci=normalize_pci(cp.cuda.runtime.deviceGetPCIBusId(0)),
                      name=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(), cupy=cp.__version__,
                      kvikio=kvikio.__version__, cuda_runtime=cp.cuda.runtime.runtimeGetVersion())
        require(device['pci']==placement['pci_by_uuid'][device['mask']], 'UUID/PCI hardware mapping')
    with patch.object(Run, '_setup_run_calibconst', inject):
        datasource = DataSource(exp=info['exp'], run=info['run'], dir=info['directory'],
            detectors=['jungfrau'], gpu_det='jungfrau', live=False, max_events=arguments.events,
            skip_calib_load='all', gpu_bulk_read=True, gpu_bulk_target_bytes=512 << 10,
            batch_size=arguments.batch, n_gpu_streams=1, gpu_memory_budget_gb=6,
            gpu_d2h_pinned_bytes=64 << 20, gpu_fn=GpuTask(task, inputs=task.inputs, calibconst=task.calibconst),
            log_level='ERROR')
        run = next(datasource.runs())
    require(run.comms.node_type() == ('smd0' if rank == 0 else 'eb' if rank == 1 else 'bd'), 'Unexpected role')
    optional_shared_resources = {name: getattr(run, name, None) is None for name in
        ('_jungfrau_shared_mem', '_geo_shared_mem', '_shared_geo_cache', '_shared_calibc_cache')}
    require(all(optional_shared_resources.values()), 'Unexpected optional derived/geometry allocation')
    distributed_checks = {}
    for key, expected in constant_records.items():
        value = run.dsparms.calibconst['jungfrau'][key][0]
        require(digest(value) == expected['sha256'], 'Distributed host constants mismatch')
        distributed_checks[key] = expected
    numerical = None
    manager = copier = None
    manager_reuse = []
    if rank >= 2:
        manager = run._make_gpu_event_manager()
        require(manager._n_bd_per_gpu == arguments.bds // active_gpus and manager.gpu_reader.task_size in (None, 1 << 20), 'Pipeline peers/task')
        copier = CopyAudit(manager._output_d2h)
        task.prepare(manager, constant_records)
        if os.environ.get('SIM10_NUMERICAL_REFERENCE') == '1':
            require(arguments.kind == 'acceptance' and arguments.stage == 'ECC', 'Numerical qualification only')
            from numerical import NumericalReference
            numerical = NumericalReference(info, stamps, constants)
            task.acceptance = numerical

        def use_prepared():
            require(not manager_reuse and not manager._closed, 'Precreated manager can be consumed only once')
            manager_reuse.append(True)
            return manager

        run._make_gpu_event_manager = use_prepared
        cp.cuda.Device().synchronize()
    setup_seconds = time.monotonic() - started
    progress=Path(str(arguments.output)+'.rank%d.progress.json'%rank)
    progress_history = []

    def mark(phase, **details):
        now = time.monotonic()
        progress_history.append(dict(phase=phase, monotonic_seconds=now,
                                     elapsed_seconds=now-started, **details))
        write(progress, dict(rank=rank, phase=phase, setup_seconds=setup_seconds,
                             history=progress_history))

    mark('setup_complete')
    host_memory['before_loop'] = snapshot()
    threads_before = thread_snapshot()
    comm.Barrier()
    observation = None
    from node_observer import network, start_memory, stop_memory
    watches=start_memory(local_uuids) if node_leader else []
    if node_leader:
        leaves = info.get('observed_leaves', [])
        idle_before = counters(leaves)
        time.sleep(2)
        idle_after = counters(leaves)
        cache_gate = condition_cache(info['files'], Path(info['directory']), 'lustre',
                                     Path(str(arguments.output) + '.cache.'+platform.node()+'.json'))
        conditioning, initial = [s['files'] for s in cache_gate['attempts'][-1]['scans']]
        observation = dict(cache_gate=cache_gate, conditioning=conditioning, initial_cache=initial, idle_before=idle_before,
                           idle_after=idle_after, before=counters(leaves),
                           conditioning_seconds=cache_gate['elapsed_seconds'])
        mark('conditioning_complete', conditioning_seconds=observation['conditioning_seconds'])
    comm.Barrier()
    net_before=network() if node_leader else None
    comm.Barrier()
    begin = time.perf_counter()
    delivery_ticks=[]
    delivered, errors_seen, cpu_reads, io = [], 0, [0], None
    trace_window = None
    from psana.psexp.event_manager import EventManager
    original_read = EventManager._read

    def counted_read(owner, *args, **kwargs):
        cpu_reads[0] += 1
        return original_read(owner, *args, **kwargs)

    if rank >= 2:
        from bulk_accounting import accounting
        tracer = int(next(line.split(':')[1] for line in Path('/proc/self/status').read_text().splitlines()
                          if line.startswith('TracerPid:')))
        require((tracer > 0) == (arguments.kind == 'pilot'), 'Only pilot cases must be syscall traced')
        trace_window = dict(tracer_pid=tracer, start=time.time())
        with patch.object(EventManager, '_read', counted_read), accounting() as io:
            with closing(run.events()) as events:
                for event in events:
                    delivered.append(int(event.timestamp))
                    if len(delivered)%200==0:delivery_ticks.append([time.perf_counter()-begin,len(delivered)])
                    if arguments.stage != 'S':
                        errors_seen += int(np.count_nonzero(event.gpu.get('sim10.errors').on_cpu))
            cp.cuda.Device().synchronize()
        trace_window['end'] = time.time()
    else:
        with closing(run.events()) as events:
            for event in events:
                delivered.append(int(event.timestamp))
    comm.Barrier()
    seconds = comm.allreduce(time.perf_counter() - begin, op=MPI.MAX)
    # Keep diagnostic file writes outside the unchanged processing timer.
    mark('processing_complete', processing_seconds=seconds, delivered=len(delivered))
    net_after=network() if node_leader else None
    delivered_by_rank=comm.allgather(len(delivered))
    node_delivered=sum(n for r,n in enumerate(delivered_by_rank) if r>=2 and placement['rank_hosts'][r]==platform.node())
    if node_leader:
        observation.update(seconds=seconds, after=counters(info.get('observed_leaves', [])))
        observation['final_cache'] = cache(info['files'], Path(info['directory']), 'lustre')
        if arguments.kind != 'acceptance':
            observation['storage_evidence'] = evidence(observation, 'lustre', EVENT_BYTES * node_delivered, seconds)
    if rank >= 2:
        count = len(delivered)
        require(delivered == sorted(set(delivered)), 'Local identity/order')
        require([timestamp for callback in task.callbacks for timestamp in callback['timestamps']] == delivered,
                'Callback/delivery coverage')
        require(not errors_seen and cpu_reads[0] == 0 and io['pressure_drains'] == 0, 'Errors/fallback/pressure drains')
        require(io['completed_requests'] == io['requests'] == 5 * count and
                io['completed_bytes'] == io['requested_bytes'] == EVENT_BYTES * count, 'Exact GPU I/O')
        require(io['read_size_histogram'] == (histogram(count) if count else {}), 'Read sizes')
        require({row['primary']['path'] for row in handles['kvikio']} == (set(info['files']) if count else set()), 'Actual KvikIO file coverage')
        require({row['path'] for row in handles['dgram']} == set(info['files']), 'Actual DgramManager file coverage')
        require(io['issue_groups_calls'] == len(task.callbacks), 'Unexpected input issue grouping')
        require(manager._closed and not manager.gpu_reader._pending and not manager.event_pool.active_count and
                manager._gpu_budget._held == 0 and manager_reuse == [True], 'Incomplete manager drain')
        if arguments.kind != 'acceptance':
            require(all(callback['size'] == arguments.batch for callback in task.callbacks), 'Unexpected callback splitting')
        require(not task.callbacks or len({callback['stream'] for callback in task.callbacks}) == 1, 'Multiple task streams')
        require(copier.bytes == (0 if arguments.stage == 'S' else count * 32 * 4), 'Unexpected task D2H byte total')
        require(all(defaults.get(key) == value for key, value in settings.items()), 'Settings changed')
        if task.cube is not None:
            task.cube.drain()
        gc.collect()
        require(not any(owner() is not None for owner in task.previous), 'Full image owner survives drain')
        task.sample('drained')
    mark('post_loop_gates_complete')
    cube_verification = inspect_cube(task.cube, delivered) if rank >= 2 and arguments.stage == 'ECC' else None
    if cube_verification is not None:
        task.sample('cube_verified')
    mark('cube_verification_complete', checked=cube_verification is not None)
    host_memory['after_loop'] = snapshot()
    validate_host_memory(host_memory, 4, 4, 16)
    memory_receipts=stop_memory(watches) if node_leader else []
    modules = ('psana', 'psana.dgram', 'psana.eventbuilder', 'psana.smdreader', 'psana.utilsdetector_ext',
               'psana.gpu.gpu_events', 'psana.gpu.gpu_task_batch', 'psana.gpu.gpu_producer', 'psana.psexp.mpi_ds',
               'psana.gpu.gpu_d2h', 'psana.gpu.gpu_task', 'psana.pycalgos.utilsdetector')
    row = dict(rank=rank, host=platform.node(), device=device, affinity=chosen, timestamps=delivered,
               delivery_ticks=delivery_ticks, node_observation=observation if node_leader else None,
               node_memory=memory_receipts, network_before=net_before, network_after=net_after,
               handles=handles, budget_peaks=list(peaks.values()), granted_uuids=granted, mpi=MPI.Get_library_version(),
               drain=dict(closed=manager._closed, pending=len(manager.gpu_reader._pending), active=manager.event_pool.active_count, held=manager._gpu_budget._held, peers=manager._n_bd_per_gpu) if manager else None,
               callbacks=task.callbacks, io=io, trace_window=trace_window, setup_seconds=setup_seconds,
               distributed_constants=distributed_checks, cpu_reads=cpu_reads[0], errors=errors_seen,
               device_constant_checks=getattr(task, 'constant_checks', {}), host_memory=host_memory,
               shared_calibration_buffer_bytes=int(getattr(getattr(run, '_calib_xtc_shared', None), 'nbytes', 0)),
               task_copy=copier.result() if copier else None, sampled_memory=task.memory,
               optional_shared_resources_absent=optional_shared_resources,
               sharing_controls={key: os.environ[key] for key in ('PS_JF_SHARE_DERIVED', 'PS_GEO_SHARE')},
               array_weakref_probe=getattr(task, 'weakref_probe', None),
               memory_ledger=getattr(task, 'ledger', None), cube_verification=cube_verification,
               sampled_numerical_reference=numerical.finish(task.cube, delivered) if numerical else None,
               cuda_intervals=task.interval_results() if rank >= 2 else None,
               threads_before=threads_before, threads_after=thread_snapshot(), runtime_files=identity(modules))
    rows = comm.gather(row, root=0)
    mark('gather_complete')
    if rank == 0:
        require(not rows[0]['timestamps'] and not rows[1]['timestamps'], 'Helper delivered events')
        merged = [timestamp for entry in rows[2:] for timestamp in entry['timestamps']]
        require(len(merged) == len(set(merged)) == arguments.events and sorted(merged) == stamps, 'Global disjoint union')
        from topology import audit_placement
        node_events=audit_placement(rows,placement)
        require(len({(entry['host'],entry['device']['pci']) for entry in rows[2:]}) == active_gpus, 'Physical placement collision')
        if arguments.kind != 'acceptance':
            require(all(entry['timestamps'] for entry in rows[2:]), 'Full case exercises every BD')
        if arguments.kind == 'acceptance':
            sizes = [callback['size'] for entry in rows[2:] for callback in entry['callbacks']]
            require(arguments.batch in sizes and any(size < arguments.batch for size in sizes), 'Full batch/tail unexercised')
    comm.Barrier()
    if rank == 0:
        report = dict(status='pass', stage=arguments.stage, kind=arguments.kind, events=arguments.events,
                      seconds=seconds, hz=arguments.events / seconds,
                      payload_gib_s=arguments.events * EVENT_BYTES / (1 << 30) / seconds,
                      rows=rows, observation=observation, node_events=node_events,
                      node_observations={r['host']:r['node_observation'] for r in rows if r['node_observation'] is not None},
                      cube_policy='gpu_shape_finite_nonzero_v1', cube_exported=False,
                      numerical_reference_checked=False,
                      rate_is_clean=arguments.kind in ('full', 'anchor'),
                      processing_instrumentation=dict(memory_samples_per_callback=2,
                          policy='All stages perform entry and exit CUDA free/pool queries and safety checks inside processing time. Clean means untraced/no CUDA timing, not zero monitoring overhead.'),
                      sampled_numerical_reference_checked=all(r.get('sampled_numerical_reference', {}).get('status') == 'pass' for r in rows[2:]) if os.environ.get('SIM10_NUMERICAL_REFERENCE') == '1' else False,
                      native_caveat='Stage 3 pinned PR331 candidate; clock sampled every 200 delivered events',
                      job=os.environ['SLURM_JOB_ID'], active_gpus=active_gpus,
                      end_to_end_seconds=time.monotonic() - started)
    comm.Barrier()
    run.close_shared_memory()
    mark('shared_memory_close_complete')
    comm.Barrier()
    handles_context.__exit__(None, None, None)
    budget_context.__exit__(None, None, None)
    if rank == 0:
        report['end_to_end_seconds'] = time.monotonic() - started
        write(Path(arguments.output), report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('inputs', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--bds', type=int, choices=(4, 8, 16), required=True)
    parser.add_argument('--batch', type=int, choices=(50, 200), required=True)
    parser.add_argument('--stage', choices=('S', 'E', 'EC', 'ECC'), required=True)
    parser.add_argument('--kind', choices=('acceptance', 'pilot', 'full', 'anchor'), required=True)
    parser.add_argument('--events', type=int, required=True)
    try:
        execute(parser.parse_args())
    except BaseException:
        traceback.print_exc()
        from mpi4py import MPI
        MPI.COMM_WORLD.Abort(1)
