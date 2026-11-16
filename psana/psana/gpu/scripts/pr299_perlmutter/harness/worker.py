import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time

from common import DATA, EXPERIMENTS, check_rows, write_json
from perlmutter_support import rank_environment, progress, physical_cores, validate_rows, runtime_provenance
from io_accounting import observe_reader_io, save_io_evidence, validate_io


input_checks = 0


def noop(batch, stream):
    pass


def check_input(batch, stream):
    global input_checks
    values = batch.input('jungfrau.raw')
    if values is None or values.dtype.name != 'uint16' or values.size == 0:
        raise RuntimeError('missing prepared Jungfrau raw input')
    input_checks += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', choices=('base', 'head'), required=True)
    parser.add_argument('--install', required=True)
    parser.add_argument('--mode', choices=('gpu', 'cpu'), required=True)
    parser.add_argument('--iterator', choices=('events', 'steps'), required=True)
    parser.add_argument('--run', type=int, choices=(51, 77), required=True)
    parser.add_argument('--events', type=int, required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--provenance-script', required=True)
    parser.add_argument('--nic-inventory', required=True)
    parser.add_argument('--diagnostics', action='store_true')
    parser.add_argument('--check-input', action='store_true')
    args = parser.parse_args()
    rank = rank_environment()
    affinity = {rank: sorted(os.sched_getaffinity(0))}
    if args.run != 51:
        raise RuntimeError('This reviewed scope covers run51 only')
    if args.diagnostics:
        import faulthandler
        faulthandler.enable()
        faulthandler.dump_traceback_later(120, repeat=True)
    step_gpu_identity = None
    if args.mode == 'cpu' or rank < 2:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        from gpu_identity import verify_step_gpu
        step_gpu_identity = verify_step_gpu(os.environ['SCREEN_GPU_UUID'])
    from mpi4py import MPI
    from psana import DataSource
    comm = MPI.COMM_WORLD
    if comm.rank != rank or comm.size not in (3, 4):
        raise RuntimeError('MPI/Slurm rank topology mismatch')
    spec = importlib.util.spec_from_file_location('provenance', args.provenance_script)
    provenance_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(provenance_module)
    provenance = provenance_module.verify_install(args.install)
    gpu = None
    if args.mode == 'gpu' and rank >= 2:
        import cupy as gpu
        import kvikio.defaults
        if kvikio.defaults.get_num_threads() != 4 or kvikio.defaults.task_size() != 1048576:
            raise RuntimeError('KvikIO worker/task settings changed')
        if gpu.cuda.runtime.getDeviceCount() != 1:
            raise RuntimeError('expected exactly one GPU visible per BD')
        gpu.zeros(1).sum().item()
        gpu.cuda.runtime.deviceSynchronize()
    runtime = runtime_provenance(gpu)
    runtime['step_gpu_identity'] = step_gpu_identity
    write_json(Path(args.output).with_name(f'runtime-rank-{rank}.json'), runtime)
    expected = json.loads(Path(args.reference).read_text())['events'][:args.events]
    options = dict(exp=EXPERIMENTS[args.run], run=args.run, dir=str(DATA),
                   detectors=['jungfrau', 'timing'] if args.run == 77 else ['jungfrau'],
                   skip_calib_load=['jungfrau'], batch_size=20,
                   max_events=0 if args.run == 77 else args.events)
    if args.mode == 'gpu':
        from psana.gpu import GpuTask
        options.update(gpu_det='jungfrau', gpu_fn=GpuTask(check_input if args.check_input else noop, ['jungfrau.raw'], []),
                       n_gpu_streams=1, gpu_memory_budget_gb=8,
                       gpu_bulk_read=True, gpu_bulk_target_bytes=1 << 20)
    with observe_reader_io(gpu is not None) as accounting:
        execute_worker(args, comm, affinity, provenance, gpu, options, expected, accounting)


def execute_worker(args, comm, affinity, provenance, gpu, options, expected, accounting):
    from psana import DataSource
    rank = comm.Get_rank()
    setup_start = time.perf_counter()
    datasource = DataSource(**options)
    run = next(datasource.runs())
    setup_seconds = time.perf_counter() - setup_start
    role = run.comms.node_type()
    timestamps = []
    step_records = []
    first_event_seconds = None
    manager = None
    reader = None
    first_io = None
    if args.mode == 'gpu' and (run.dsparms.n_gpu_streams != 1 or run.dsparms.gpu_memory_budget_gb != 8):
        raise RuntimeError('GPU settings changed')
    from storage import snapshot, delta
    network = json.loads(Path(args.nic_inventory).read_text())
    before_nic = snapshot(network) if rank == 0 else None
    comm.Barrier()
    started = time.perf_counter()
    loop_complete = False
    cumulative_io = None
    try:
        if args.iterator == 'events':
            for event in run.events():
                if first_event_seconds is None:
                    first_event_seconds = time.perf_counter() - started
                    manager = getattr(run, '_gpu_manager_for_placement', None)
                    reader = getattr(manager, 'gpu_reader', None)
                    first_io = reader.io_stats() if reader is not None else None
                timestamps.append(int(event.timestamp))
                progress(args, rank, timestamps, started)
        else:
            for step_index, step in enumerate(run.steps()):
                for event in step.events():
                    if first_event_seconds is None:
                        first_event_seconds = time.perf_counter() - started
                        manager = getattr(run, '_gpu_manager_for_placement', None)
                        reader = getattr(manager, 'gpu_reader', None)
                        first_io = reader.io_stats() if reader is not None else None
                    timestamp = int(event.timestamp)
                    timestamps.append(timestamp)
                    step_records.append((timestamp, step_index))
                    progress(args, rank, timestamps, started)
        if gpu is not None:
            gpu.cuda.runtime.deviceSynchronize()
        loop_complete = True
    finally:
        loop_seconds = time.perf_counter() - started
        manager = getattr(run, '_gpu_manager_for_placement', None)
        reader = getattr(manager, 'gpu_reader', None)
        if accounting is not None:
            cumulative_io = save_io_evidence(
                Path(args.output).with_name(f'io-rank-{rank}.json'), accounting, reader,
                rank=rank, role=role, loop_complete=loop_complete,
                delivered_events=len(timestamps), input_checks=input_checks,
                first_batch_io=first_io)
    io_stats = reader.io_stats() if reader is not None else None
    if args.mode == 'gpu' and role == 'bd':
        validate_io(cumulative_io)
        if args.check_input and input_checks == 0:
            raise RuntimeError('input preflight callback never executed')
    if (args.mode == 'cpu' or role != 'bd') and 'cupy' in sys.modules:
        raise RuntimeError('unexpected GPU initialization on CPU role')
    row = dict(rank=rank, role=role, timestamps=timestamps, step_records=step_records,
               setup_seconds=setup_seconds, first_event_seconds=first_event_seconds,
               loop_seconds=loop_seconds, first_batch_io=first_io, last_batch_io=io_stats,
               cumulative_io=cumulative_io,
               input_checks=input_checks, affinity=affinity[rank],
               rss_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               provenance=provenance, physical_cores=physical_cores(affinity[rank]),
               local_rank=int(os.environ['SLURM_LOCALID']),
               runtime_provenance=runtime_provenance(gpu), gpu_uuid=None, mpi_library=__import__('mpi4py').MPI.Get_library_version())
    if gpu is not None:
        from gpu_identity import cuda_identity
        row['gpu_uuid'] = cuda_identity()['uuid']
        if row['gpu_uuid'] != os.environ['SCREEN_GPU_UUID']:
            raise RuntimeError('BD GPU UUID differs from allocation')
    comm.Barrier()
    nic_window = delta(before_nic, snapshot(network)) if rank == 0 else None
    rows = comm.gather(row, root=0)
    run.close_shared_memory()
    if rank == 0:
        validate_rows(rows, args.mode)
        digest = check_rows(rows, expected, args.iterator)
        duration = max(row['loop_seconds'] for row in rows)
        for row in rows:
            row['event_count'] = len(row.pop('timestamps'))
            row.pop('step_records')
        write_json(args.output, dict(complete=True, runtime=args.runtime, mode=args.mode,
                                    iterator=args.iterator, run=args.run,
                                    events=len(expected), timestamp_sha256=digest,
                                    loop_seconds=duration, events_per_second=len(expected) / duration,
                                    rows=rows, n_gpu_streams=1, batch_size=20,
                                    gpu_memory_budget_gb=8, gpu_bulk_read=True,
                                    nic_loop_window=nic_window,
                                    io_note='Cumulative I/O sums disjoint pre-reset windows and one final window per reader; first/last snapshots are not run totals. Reset-boundary observation is identical for base/head.'))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        try:
            from mpi4py import MPI
            MPI.COMM_WORLD.Abort(1)
        finally:
            raise
