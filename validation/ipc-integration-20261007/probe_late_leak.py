"""Does the late pinning path leak contexts onto other GPUs?

Item 4 of psana2-early-pinning-note.docx. The note's recommendation -- collapse
pin_device to a single late path and leave isolation to the launcher -- depends
on whether a process that can still *see* every GPU in the mask ends up
allocating on more than one.

Each rank selects its device with Device(i).use(), allocates, runs a kernel,
then reports which GPUs the driver says this PID is resident on. A PID
appearing on one GPU means the late path is self-sufficient; appearing on two
means something defaults to device 0 and launcher-side narrowing is required
rather than optional.

Also answers the note's premise: is the early path reachable at all through a
normal psana import?

Run with several ranks and several GPUs::

    mpirun -n 4 python probe_late_leak.py
"""
import json
import os
import subprocess
import sys


def emit(tag, obj):
    print(f'LEAK_{tag} ' + json.dumps(obj, indent=2, sort_keys=True), flush=True)


def cuda_initialised():
    """True if this process has already called cuInit, via the driver API."""
    import ctypes
    try:
        lib = ctypes.CDLL('libcuda.so.1')
    except OSError:
        return None
    count = ctypes.c_int(-1)
    # 0 = success (already initialised), 3 = CUDA_ERROR_NOT_INITIALIZED
    return lib.cuDeviceGetCount(ctypes.byref(count)) == 0


def compute_apps():
    """(pid, gpu_uuid, used_mib) rows the driver reports for this node."""
    try:
        text = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
             '--format=csv,noheader'],
            capture_output=True, text=True, timeout=30, check=True).stdout
    except Exception as exc:                              # noqa: BLE001
        return None, f'{type(exc).__name__}: {exc}'
    rows = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 3 and parts[0].isdigit():
            rows.append({'pid': int(parts[0]), 'uuid': parts[1],
                         'used': parts[2]})
    return rows, None


def main():
    result = {'pid': os.getpid()}

    # ---- the note's premise: is the early path reachable? ---------------
    # Nothing CUDA-related has been imported yet at this point.
    result['cuda_before_any_import'] = cuda_initialised()
    result['mpi4py_in_modules_before'] = 'mpi4py.MPI' in sys.modules

    import psana                                           # noqa: F401
    result['cuda_after_import_psana'] = cuda_initialised()
    result['mpi4py_in_modules_after_psana'] = 'mpi4py.MPI' in sys.modules
    result['PS_PARALLEL'] = os.environ.get('PS_PARALLEL', '<unset>')

    from psana.gpu.gpu_placement import (
        device_pci, discover_peers, pin_device, select_device,
    )

    # ---- the late path, as production takes it -------------------------
    pinned = pin_device()
    result['pinned'] = pinned.pinned
    result['requested_pci'] = pinned.requested_pci
    result['permitted'] = list(pinned.permitted)
    result['local_rank'] = pinned.local_rank

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    result['rank'] = rank

    placement = discover_peers(pinned, comm, is_gpu_worker=True)
    import cupy as cp
    result['visible_count'] = placement.visible_count
    result['ordinal'] = placement.device_ordinal
    result['device_pci'] = device_pci(cp, placement.device_ordinal)

    # Allocate and compute, so a context definitely exists wherever work goes.
    buffer = cp.arange(1 << 22, dtype=cp.float32)          # 16 MiB
    total = float(cp.asnumpy(cp.sum(buffer, dtype=cp.float64)))
    result['kernel_sum_ok'] = total > 0
    result['allocation_device_id'] = int(buffer.device.id)
    result['allocation_pci'] = device_pci(cp, int(buffer.device.id))

    # The assertion that matters: the allocation must land on the device this
    # rank selected, not on device 0.
    result['allocation_on_selected_device'] = (
        result['allocation_pci'] == pinned.requested_pci)

    # ---- where does the driver say this PID is resident? ---------------
    comm.Barrier()
    rows, error = compute_apps()
    mine = [r for r in (rows or []) if r['pid'] == os.getpid()]
    result['compute_apps_error'] = error
    result['my_gpu_rows'] = mine
    result['my_gpu_count'] = len(mine)
    result['leaked'] = len(mine) > 1

    emit(f'RANK{rank}', result)

    gathered = comm.gather(result, root=0)
    comm.Barrier()
    if rank == 0 and gathered is not None:
        ranks = [r for r in gathered if r]
        by_pid_count = {r['rank']: r['my_gpu_count'] for r in ranks}
        emit('SUMMARY', {
            'n_ranks': len(ranks),
            'early_path_reachable': any(
                r['cuda_after_import_psana'] is False for r in ranks),
            'all_pinned': all(r['pinned'] for r in ranks),
            'gpus_per_pid': by_pid_count,
            'any_leaked': any(r['leaked'] for r in ranks),
            'all_allocations_on_selected_device': all(
                r['allocation_on_selected_device'] for r in ranks),
            'distinct_devices_used': sorted(
                {r['allocation_pci'] for r in ranks}),
            'verdict': ('late path is self-sufficient'
                        if not any(r['leaked'] for r in ranks)
                        and all(r['allocation_on_selected_device']
                                for r in ranks)
                        else 'launcher-side narrowing is REQUIRED'),
        })
    return 0


if __name__ == '__main__':
    sys.exit(main())
