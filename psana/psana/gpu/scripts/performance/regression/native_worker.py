"""Rank-side raw workload for the reviewed Perlmutter adapter (no job submission)."""
import argparse
import cProfile
import ctypes
import os
from pathlib import Path
import resource
import re
import sys
import time
import uuid

from native_contract import (artifact, check_rows, check_selected_files, cold_cache, load_contract, nic_delta,
                             nic_snapshot, require, runtime_evidence, write)


class ReaderAccounting:
    """Disjoint pre-reset windows and one final window, including drain work."""
    fields = ('total_bytes', 'total_requests', 'requested_bytes', 'useful_bytes', 'total_ns', 'issue_to_complete_ns')

    def __init__(self, reader_type):
        self.reader_type, self.original = reader_type, reader_type.reset_io_stats
        self.original_init = reader_type.__init__
        self.readers, self.done = {}, False
        self.totals = {k: 0 for k in self.fields}
        self.totals.update(fallback=True, bulk=True, reset_windows=0, tail_windows=0)

        def reset(reader):
            self.readers[id(reader)] = reader
            snapshot = reader.io_stats()
            value = self.original(reader)
            self.record(snapshot, 'reset_windows')
            return value
        def initialize(reader, *args, **kwargs):
            self.original_init(reader, *args, **kwargs)
            self.readers[id(reader)] = reader
        reader_type.__init__ = initialize
        reader_type.reset_io_stats = reset

    def record(self, snapshot, window):
        for key in self.fields:
            self.totals[key] += snapshot[key]
        self.totals['fallback'] &= snapshot['compat_mode'] is True and snapshot['io_path'] == 'CPU-fallback'
        self.totals['bulk'] &= snapshot['bulk_read'] is True
        self.totals[window] += 1

    def finish(self, reader=None):
        if not self.done:
            try:
                if reader is not None:
                    self.readers[id(reader)] = reader
                for reader in self.readers.values():
                    self.record(reader.io_stats(), 'tail_windows')
            finally:
                self.reader_type.reset_io_stats = self.original
                self.reader_type.__init__ = self.original_init
                self.done = True
        return dict(self.totals)


class DeliveryAccounting:
    """Observe EB packets, yielded delivery windows, and the separate finish drain.

    GPU retirement can yield a previous packet's events while processing a later
    packet. Window populations are deliberately NOT called EB batch sizes.
    """
    def __init__(self, events_type, elapsed):
        self.events_type, self.elapsed = events_type, elapsed
        self.original_init, self.original_end = events_type.__init__, events_type._emit_batch_end
        self.records, self.first_nonempty = [], None

        class Source:
            def __init__(source, original, record):
                source.original, source.record = original, record
            def __iter__(source):
                return source
            def __next__(source):
                value = next(source.original)
                source.record['eb_deliveries'] += 1
                return value
            def __getattr__(source, name):
                return getattr(source.original, name)

        def initialize(events, *args, **kwargs):
            self.original_init(events, *args, **kwargs)
            record = dict(eb_deliveries=0, yielded_events_per_delivery_window=[], drain_yielded_events=0)
            events._regression_delivery_record = record
            self.records.append(record)
            if events._batch_source is not None:
                events._batch_source = Source(events._batch_source, record)

        def end(events):
            record = events._regression_delivery_record
            count = events._batch_event_count
            if events._batch_start_time is not None:
                record['yielded_events_per_delivery_window'].append(count)
            elif count:
                # Native finish() drains once; _emit_batch_end leaves its count
                # intact because its start marker is None. Snapshot, don't add.
                record['drain_yielded_events'] = count
            if count and self.first_nonempty is None:
                self.first_nonempty = self.elapsed()
            return self.original_end(events)
        events_type.__init__, events_type._emit_batch_end = initialize, end

    def restore(self):
        self.events_type.__init__, self.events_type._emit_batch_end = self.original_init, self.original_end

    def result(self, delivered):
        result = dict(eb_deliveries=sum(r['eb_deliveries'] for r in self.records),
                      yielded_events_per_delivery_window=[n for r in self.records for n in r['yielded_events_per_delivery_window']],
                      drain_yielded_events=sum(r['drain_yielded_events'] for r in self.records),
                      first_nonempty_yield_window_seconds=self.first_nonempty)
        require(sum(result['yielded_events_per_delivery_window']) + result['drain_yielded_events'] == delivered,
                'delivery-window and drain populations do not cover delivered events')
        return result


def gpu_identity():
    driver = ctypes.CDLL('libcuda.so.1')

    def call(name, types, *values):
        function = getattr(driver, name)
        function.argtypes, function.restype = types, ctypes.c_int
        require(function(*values) == 0, 'CUDA identity query failed: ' + name)

    call('cuInit', [ctypes.c_uint], 0)
    count, device = ctypes.c_int(), ctypes.c_int()
    call('cuDeviceGetCount', [ctypes.POINTER(ctypes.c_int)], ctypes.byref(count))
    require(count.value == 1, 'expected exactly one visible GPU')
    call('cuDeviceGet', [ctypes.POINTER(ctypes.c_int), ctypes.c_int], ctypes.byref(device), 0)
    identifier, bus, name = (ctypes.c_ubyte * 16)(), ctypes.create_string_buffer(32), ctypes.create_string_buffer(256)
    call('cuDeviceGetUuid_v2', [ctypes.c_void_p, ctypes.c_int], identifier, device)
    call('cuDeviceGetPCIBusId', [ctypes.c_void_p, ctypes.c_int, ctypes.c_int], bus, len(bus), device)
    call('cuDeviceGetName', [ctypes.c_void_p, ctypes.c_int, ctypes.c_int], name, len(name), device)
    pci = bus.value.decode()
    numa = Path('/sys/bus/pci/devices', pci.lower(), 'numa_node')
    return dict(uuid='GPU-' + str(uuid.UUID(bytes=bytes(identifier))), pci=pci, model=name.value.decode(),
                numa=int(numa.read_text()) if numa.exists() else None)


def step_gpu_environment(environment):
    step, visible = environment.get('SLURM_STEP_GPUS'), environment.get('CUDA_VISIBLE_DEVICES')
    single = lambda value: bool(value) and re.fullmatch(r'(?:[0-9]+|GPU-[0-9a-f-]{36})', value) is not None
    require(single(step) and single(visible), 'one Slurm step GPU and visibility required')
    job = environment.get('SLURM_JOB_GPUS', '').split(',')
    require(job and all(single(device) for device in job), 'valid job GPU allocation required')
    if step.isdigit() and all(device.isdigit() for device in job):
        require(step in job, 'step GPU outside job allocation')
    return {k: environment[k] for k in ('SLURM_JOB_GPUS', 'SLURM_STEP_GPUS', 'CUDA_VISIBLE_DEVICES')}


def kvikio_settings(kvikio):
    from kvikio import defaults
    expected = dict(num_threads=4, task_size=1048576, bounce_buffer_size=16777216,
                    compat_mode=kvikio.CompatMode.ON, auto_direct_io_read=False, auto_direct_io_write=False)
    actual = {key: defaults.get(key) for key in expected}
    require(actual == expected, 'effective KvikIO settings differ')
    actual['compat_mode'] = 'ON'
    return actual


def physical_cores(affinity):
    cores = []
    for cpu in affinity:
        root = Path('/sys/devices/system/cpu/cpu%d/topology' % cpu)
        cores.append([int((root / 'physical_package_id').read_text()), int((root / 'core_id').read_text())])
    require(len(set(tuple(c) for c in cores)) == len(cores), 'SMT siblings in affinity')
    return cores


def execute(request_path, folder, stage):
    from native_contract import read
    request = read(request_path)
    site, reference, inputs, runtime = load_contract(request)
    require(Path(__file__).resolve() == artifact(request, 'native_worker'), 'unfrozen worker')
    import native_contract
    require(Path(native_contract.__file__).resolve() == artifact(request, 'native_contract'), 'unfrozen helper')
    rank = int(os.environ['SLURM_PROCID'])
    require(rank == int(os.environ['SLURM_LOCALID']), 'single node only')
    case, contract = request['case'], request['contract']
    affinity = sorted(os.sched_getaffinity(0))
    require(affinity == sorted(site['cpu_affinity'][str(case['bds'])][rank]), 'rank CPU binding differs')
    cores = physical_cores(affinity)
    folder = Path(folder)
    if rank == 0:
        write(folder / 'step.json', dict(job=os.environ['SLURM_JOB_ID'], step=os.environ['SLURM_STEP_ID'], node=os.uname().nodename))
    is_gpu = case['workload'] == 'gpu-raw' and rank >= 2
    gpu_environment = step_gpu_environment(os.environ) if is_gpu else None
    if not is_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    from mpi4py import MPI
    from psana import DataSource
    comm = MPI.COMM_WORLD
    require(comm.rank == rank and comm.size == case['bds'] + 2, 'MPI/Slurm topology mismatch')
    require('MPICH' in MPI.Get_library_version(), 'Cray MPICH required')
    gpu, identity, accounting = None, None, None
    if is_gpu:
        import cupy as gpu
        import kvikio
        import kvikio.defaults as defaults
        require(kvikio.__version__.startswith('26.10.'), 'this adapter requires KvikIO 26.10 API')
        require(defaults.get('num_threads') == 4 and defaults.get('task_size') == 1048576 and
                defaults.get('bounce_buffer_size') == 16777216 and defaults.get('compat_mode') == kvikio.CompatMode.ON, 'effective KvikIO settings differ')
        defaults.set(dict(auto_direct_io_read=False, auto_direct_io_write=False))
        require(defaults.get('auto_direct_io_read') is False and defaults.get('auto_direct_io_write') is False, 'direct I/O policy differs')
        require(gpu.cuda.runtime.runtimeGetVersion() == runtime['cuda_runtime_version'] and
                gpu.cuda.runtime.driverGetVersion() == runtime['cuda_driver_version'], 'CUDA runtime/driver differs')
        require(gpu.__version__ == runtime['cupy_version'] and kvikio.__version__ == runtime['kvikio_version'], 'GPU package versions differ')
        identity = gpu_identity()
        require(identity['uuid'] == site['gpu_uuid'] and 'A100' in identity['model'], 'wrong physical GPU')
        gpu.zeros(1).sum().item()
        gpu.cuda.runtime.deviceSynchronize()
        from psana.gpu.gpu_kvikio_read import KvikioGpuReader
        accounting = ReaderAccounting(KvikioGpuReader)
    # Hash libraries/dependencies now as well as after execution, outside loop timing.
    provenance_before = runtime_evidence(runtime)
    expected = reference['events'][:contract['warmup']['events']] if stage == 'warmup' else reference['events']
    callback_sizes = []

    def callback(batch, stream):
        # Dense input declaration schedules gather. No user kernel or D2H output.
        callback_sizes.append(int(batch.size))
        if request['phase'] in ('correctness', 'profile'):
            values = batch.input('jungfrau.raw')
            require(values is not None and values.dtype.name == 'uint16' and values.size > 0, 'no prepared Jungfrau input')

    settings = contract['settings']
    options = dict(exp='mfx100848724', run=51, dir=inputs['root'], live=False,
                   detectors=['jungfrau'], skip_calib_load=['jungfrau'], batch_size=settings['batch_size'], max_events=len(expected))
    if case['workload'] == 'gpu-raw':
        from psana.gpu import GpuTask
        options.update(gpu_det='jungfrau', gpu_fn=GpuTask(callback, ['jungfrau.raw'], []),
                       n_gpu_streams=1, gpu_memory_budget_gb=8, gpu_bulk_read=True, gpu_bulk_target_bytes=1048576)
    setup_start = time.perf_counter()
    datasource = DataSource(**options)
    run = next(datasource.runs())
    setup_seconds = time.perf_counter() - setup_start
    require(run.expt == 'mfx100848724' and run.runnum == 51, 'BeginRun identity differs')
    observed_settings = {k: getattr(run.dsparms, k) for k in ('batch_size',)}
    if case['workload'] == 'gpu-raw':
        observed_settings.update({k: getattr(run.dsparms, k) for k in ('n_gpu_streams', 'gpu_memory_budget_gb', 'gpu_bulk_read', 'gpu_bulk_target_bytes')})
    require(all(settings[k] == v for k, v in observed_settings.items()), 'effective DataSource settings differ')
    check_selected_files(datasource, inputs)
    io_settings = kvikio_settings(kvikio) if gpu else None
    role = run.comms.node_type()
    from psana.psexp.events import Events
    started = None
    delivery = DeliveryAccounting(Events, lambda: time.perf_counter() - started)
    comm.Barrier()
    cache = cold_cache(inputs) if rank == 0 and stage == 'measured' else None
    comm.Barrier()
    before = nic_snapshot(site['nic_interfaces']) if rank == 0 else None
    profiler = cProfile.Profile() if request['phase'] == 'profile' and stage == 'measured' else None
    timestamps, steps, cpu_bytes = [], [], 0
    cpu_start = time.process_time()
    started = time.perf_counter()
    if profiler:
        profiler.enable()
    try:
        iterators = [(None, run.events())] if case['iterator'] == 'events' else ((i, step.events()) for i, step in enumerate(run.steps()))
        for step, events in iterators:
            for event in events:
                timestamp = int(event.timestamp)
                timestamps.append(timestamp)
                if step is not None:
                    steps.append([timestamp, step])
                if case['workload'] == 'cpu-raw':
                    cpu_bytes += sum(d._size for d in event._dgrams if d is not None)
        if gpu is not None:
            gpu.cuda.runtime.deviceSynchronize()
    finally:
        if profiler:
            profiler.disable()
            profiler.dump_stats(str(folder / ('rank-%d.pstats' % rank)))
        delivery.restore()
    local_loop = time.perf_counter() - started
    cpu_seconds = time.process_time() - cpu_start
    comm.Barrier()
    loop_seconds = time.perf_counter() - started
    after = nic_snapshot(site['nic_interfaces']) if rank == 0 else None
    delivery_counts = delivery.result(len(timestamps))
    io = accounting.finish() if accounting else None
    require(not (case['workload'] == 'cpu-raw' or rank < 2) or 'cupy' not in sys.modules, 'CUDA imported on CPU role')
    close_start = time.perf_counter()
    run.close_shared_memory()
    close_seconds = time.perf_counter() - close_start
    if gpu:
        require(kvikio_settings(kvikio) == io_settings, 'KvikIO settings changed during loop')
    provenance_after = runtime_evidence(runtime)
    memory = dict(host_rss_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  gpu_pool_current_bytes=int(gpu.get_default_memory_pool().used_bytes()) if gpu else None,
                  gpu_pool_reserved_bytes=int(gpu.get_default_memory_pool().total_bytes()) if gpu else None,
                  gpu_used_peak_bytes=None, pinned_peak_bytes=None,
                  unavailable=['GPU physical/pool peak', 'pinned-memory peak', 'kernel launch count', 'candidate/live-window visits'])
    row = dict(rank=rank, role=role, node=os.uname().nodename, affinity=affinity, physical_cores=cores,
               gpu=identity, gpu_environment=gpu_environment, timestamps=timestamps, steps=steps, cpu_raw_bytes=cpu_bytes,
               callback_events=sum(callback_sizes), callback_sizes=callback_sizes, kvikio_settings=io_settings,
               delivery=delivery_counts, io=io, memory=memory, cpu_seconds=cpu_seconds,
               setup_seconds=setup_seconds, first_batch_seconds=delivery_counts['first_nonempty_yield_window_seconds'],
               local_loop_seconds=local_loop, loop_seconds=loop_seconds, close_seconds=close_seconds,
               observed_settings=observed_settings, environment={k: os.environ.get(k) for k in
                   list(contract['environment']) + ['PS_PARALLEL', 'PS_SMD_N_EVENTS', 'OMP_NUM_THREADS', 'KVIKIO_NTHREADS', 'KVIKIO_TASK_SIZE', 'KVIKIO_BOUNCE_BUFFER_SIZE', 'KVIKIO_COMPAT_MODE']},
               provenance_before=provenance_before, provenance_after=provenance_after,
               mpi_library=MPI.Get_library_version())
    write(folder / ('rank-%d.json' % rank), row)
    rows = comm.gather(row, root=0)
    if rank == 0:
        work = check_rows(rows, expected, case)
        require(all(r['first_batch_seconds'] is not None for r in rows[2:]), 'first completed batch missing')
        write(folder / 'complete.json', dict(status='pass', work=work, rows=rows, cache=cache,
                                            nic=nic_delta(before, after), nic_before=before, nic_after=after,
                                            diagnostic=bool(profiler), stage=stage))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('request')
    parser.add_argument('folder')
    parser.add_argument('stage', choices=('warmup', 'measured'))
    args = parser.parse_args()
    try:
        execute(args.request, args.folder, args.stage)
    except BaseException:
        import traceback
        traceback.print_exc()
        try:
            from mpi4py import MPI
            MPI.COMM_WORLD.Abort(1)
        finally:
            raise


if __name__ == '__main__':
    main()
