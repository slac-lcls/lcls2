"""One fresh MPI case, with untimed setup, strict gates and bounded export."""
import argparse
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
sys.path.extend([str(HERE.parent / 'sim10_kvikio_bounce_20261113'),
                 str(HERE.parent / 'sim10_storage_gpu_20261114')])
from support import require, write, affinity, EVENT_BYTES, histogram
from cpu_plan import thread_snapshot
from runtime import activate, identity
from storage import cache, counters, evidence
from reference import load_constants, SeedReader, CpuCalibration, CubeReference, digest, samples
from host_memory import snapshot, demand, validate_host_memory


def execute(arguments):
    rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    placement = json.loads(os.environ['SIM10_GPU_PLAN'])
    require(arguments.bds in (1, 4) and arguments.batch == {1: 50, 4: 200}[arguments.bds], 'Topology/batch')
    require(len(placement['rank_uuids']) == arguments.bds + 2 and
            len(set(placement['rank_uuids'][2:])) == arguments.bds, 'One BD per physical GPU')
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
    started = time.monotonic()
    host_memory = dict(startup=snapshot(), demand=demand(arguments.bds, 4, 16))
    require(host_memory['startup']['mem_available_bytes'] >= 32 << 30, 'Insufficient host setup headroom')
    info = json.loads(Path(arguments.inputs).read_text())
    stamps = list(map(int, info['timestamps'][:arguments.events]))
    require(len(stamps) == arguments.events and stamps == sorted(set(stamps)), 'Global source identity')
    require(info['media'] == 'weka', 'This is the primary Weka cohort only')
    if arguments.kind != 'acceptance':
        require(arguments.events == (4000 if arguments.kind == 'pilot' else 100000), 'Frozen event count')
    else:
        require(arguments.events == {1: 201, 4: 1003}[arguments.bds], 'Bounded tail fixture')
    constants, constant_records = ({}, {})
    if arguments.stage in ('EC', 'ECC'):
        constants, constant_records = load_constants()

    def inject(run):
        require(rank == 0, 'Base calibration loader must only run on smd0')
        run._calib_const = {'jungfrau': constants} if constants else {}
        run.dsparms.calibconst = run._calib_const

    adapter.register_reader()
    cpu_reader = cpu_calibration = None
    accepted = []
    if arguments.kind == 'acceptance' and rank >= 2 and arguments.stage != 'S':
        cpu_reader = SeedReader(info['files'], stamps)
        if constants:
            cpu_calibration = CpuCalibration(constants)

    def accept(timestamps, raw, calibrated, errors):
        require(not np.any(errors.get()), 'Acceptance descriptor error')
        for event_index, timestamp in enumerate(timestamps):
            expected_raw = cpu_reader.raw(timestamp)
            observed_raw = raw[event_index].get()
            require(np.array_equal(expected_raw, observed_raw), 'Full expanded words differ')
            record = dict(timestamp=int(timestamp), raw_sha256=digest(observed_raw))
            if calibrated is not None:
                expected_calibrated = cpu_calibration(expected_raw)
                observed_calibrated = calibrated[event_index].get()
                require(np.array_equal(expected_calibrated.view(np.uint32), observed_calibrated.view(np.uint32)),
                        'Full calibration words differ from CPU-v3')
                record['calibrated_sha256'] = digest(observed_calibrated)
            accepted.append(record)

    task = StageTask(arguments.stage, arguments.batch, stamps,
                     acceptance=accept if arguments.kind == 'acceptance' and arguments.stage != 'S' else None,
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
        device = dict(mask=os.environ['CUDA_VISIBLE_DEVICES'], pci=cp.cuda.runtime.deviceGetPCIBusId(0),
                      name=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(), cupy=cp.__version__,
                      kvikio=kvikio.__version__, cuda_runtime=cp.cuda.runtime.runtimeGetVersion())
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
    manager = copier = None
    manager_reuse = []
    if rank >= 2:
        manager = run._make_gpu_event_manager()
        require(manager._n_bd_per_gpu == 1 and manager.gpu_reader.task_size in (None, 1 << 20), 'Pipeline peers/task')
        copier = CopyAudit(manager._output_d2h)
        task.prepare(manager, constant_records)

        def use_prepared():
            require(not manager_reuse and not manager._closed, 'Precreated manager can be consumed only once')
            manager_reuse.append(True)
            return manager

        run._make_gpu_event_manager = use_prepared
        cp.cuda.Device().synchronize()
    cube_reference = None
    if rank == 0 and arguments.stage == 'ECC':
        cpu_reader = SeedReader(info['files'], stamps)
        cube_reference = CubeReference(cpu_reader, constants, arguments.kind != 'full')
    setup_seconds = time.monotonic() - started
    host_memory['before_loop'] = snapshot()
    threads_before = thread_snapshot()
    comm.Barrier()
    observation = None
    if rank == 0:
        leaves = info.get('observed_leaves', [])
        idle_before = counters(leaves)
        time.sleep(2)
        idle_after = counters(leaves)
        conditioning = cache(info['files'], Path(info['directory']), 'weka', True)
        initial = cache(info['files'], Path(info['directory']), 'weka')
        require(all(row['resident_pages'] == 0 for row in initial), 'Cold host-page-cache gate')
        observation = dict(conditioning=conditioning, initial_cache=initial, idle_before=idle_before,
                           idle_after=idle_after, before=counters(leaves))
    comm.Barrier()
    begin = time.perf_counter()
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
    if rank == 0:
        observation.update(seconds=seconds, after=counters(info.get('observed_leaves', [])))
        observation['final_cache'] = cache(info['files'], Path(info['directory']), 'weka')
        if arguments.kind != 'acceptance':
            observation['storage_evidence'] = evidence(observation, 'weka', EVENT_BYTES * arguments.events, seconds)
    if rank >= 2:
        count = len(delivered)
        require(delivered == sorted(set(delivered)), 'Local identity/order')
        require([timestamp for callback in task.callbacks for timestamp in callback['timestamps']] == delivered,
                'Callback/delivery coverage')
        require(not errors_seen and cpu_reads[0] == 0 and io['pressure_drains'] == 0, 'Errors/fallback/pressure drains')
        require(io['completed_requests'] == io['requests'] == 5 * count and
                io['completed_bytes'] == io['requested_bytes'] == EVENT_BYTES * count, 'Exact GPU I/O')
        require(io['read_size_histogram'] == (histogram(count) if count else {}), 'Read sizes')
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
    host_memory['after_loop'] = snapshot()
    validate_host_memory(host_memory, arguments.bds, 4, 16)
    modules = ('psana', 'psana.dgram', 'psana.eventbuilder', 'psana.smdreader', 'psana.utilsdetector_ext',
               'psana.gpu.gpu_events', 'psana.gpu.gpu_task_batch', 'psana.gpu.gpu_producer', 'psana.psexp.mpi_ds',
               'psana.gpu.gpu_d2h', 'psana.gpu.gpu_task', 'psana.pycalgos.utilsdetector')
    row = dict(rank=rank, host=platform.node(), device=device, affinity=chosen, timestamps=delivered,
               callbacks=task.callbacks, io=io, trace_window=trace_window, setup_seconds=setup_seconds,
               distributed_constants=distributed_checks, cpu_reads=cpu_reads[0], errors=errors_seen,
               device_constant_checks=getattr(task, 'constant_checks', {}), host_memory=host_memory,
               shared_calibration_buffer_bytes=int(getattr(getattr(run, '_calib_xtc_shared', None), 'nbytes', 0)),
               task_copy=copier.result() if copier else None, sampled_memory=task.memory,
               optional_shared_resources_absent=optional_shared_resources,
               sharing_controls={key: os.environ[key] for key in ('PS_JF_SHARE_DERIVED', 'PS_GEO_SHARE')},
               array_weakref_probe=getattr(task, 'weakref_probe', None),
               memory_ledger=getattr(task, 'ledger', None), acceptance=accepted,
               cuda_intervals=task.interval_results() if rank >= 2 else None,
               threads_before=threads_before, threads_after=thread_snapshot(), runtime_files=identity(modules))
    rows = comm.gather(row, root=0)
    if rank == 0:
        require(not rows[0]['timestamps'] and not rows[1]['timestamps'], 'Helper delivered events')
        merged = [timestamp for entry in rows[2:] for timestamp in entry['timestamps']]
        require(len(merged) == len(set(merged)) == arguments.events and sorted(merged) == stamps, 'Global disjoint union')
        require(len({entry['device']['pci'] for entry in rows[2:]}) == arguments.bds, 'Physical placement collision')
        if arguments.kind == 'acceptance':
            sizes = [callback['size'] for entry in rows[2:] for callback in entry['callbacks']]
            require(arguments.batch in sizes and any(size < arguments.batch for size in sizes), 'Full batch/tail unexercised')
            if arguments.stage != 'S':
                require(sum(len(entry['acceptance']) for entry in rows[2:]) == arguments.events, 'Missing full-pixel acceptance')
    comm.Barrier()
    exported = None
    export_wall_seconds = 0.0
    if arguments.stage == 'ECC':
        from mpi_cube import export
        export_start = time.monotonic()
        exported = export(comm, task.cube, arguments.local_output, stamps, delivered,
                          validate_bin=cube_reference, identity_gate_passed=True, expected_bds=arguments.bds)
        export_wall_seconds = time.monotonic() - export_start
    if rank == 0:
        report = dict(status='pass', stage=arguments.stage, kind=arguments.kind, events=arguments.events,
                      seconds=seconds, hz=arguments.events / seconds,
                      payload_gib_s=arguments.events * EVENT_BYTES / (1 << 30) / seconds,
                      rows=rows, observation=observation, cube=exported,
                      numerical_checks=cube_reference.records if cube_reference else [],
                      sample_seed=20261115, sample_indices=samples().tolist(),
                      rate_is_clean=arguments.kind in ('full', 'anchor'),
                      processing_instrumentation=dict(memory_samples_per_callback=2,
                          policy='All stages perform entry and exit CUDA free/pool queries and safety checks inside processing time. Clean means untraced/no CUDA timing, not zero monitoring overhead.'),
                      native_caveat='Inherited PR298 receipt has psana/meson.build mismatch; no fresh native build',
                      end_to_end_seconds=time.monotonic() - started)
        if exported is not None:
            reference_seconds = sum(entry['seconds'] for entry in cube_reference.records)
            report['export_timing'] = dict(wall_seconds=export_wall_seconds,
                serialized_reference_seconds=reference_seconds,
                export_excluding_reference_seconds=export_wall_seconds - reference_seconds,
                processing_plus_validated_export_hz=arguments.events / (seconds + export_wall_seconds),
                processing_plus_export_hz=arguments.events / (seconds + export_wall_seconds - reference_seconds),
                interpretation='Second rate is derived by removing serialized root reference/hash intervals from export wall time; bin barriers prevent overlap with later transfers. Not a standalone clean export benchmark.')
    if cpu_reader is not None:
        cpu_reader.close()
    comm.Barrier()
    run.close_shared_memory()
    comm.Barrier()
    if rank == 0:
        report['end_to_end_seconds'] = time.monotonic() - started
        write(Path(arguments.output), report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('inputs', 'output', 'local-output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--bds', type=int, choices=(1, 4), required=True)
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
