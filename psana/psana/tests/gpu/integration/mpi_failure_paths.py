"""Failure paths on real MPI: do they abort, or do they hang?

Two cases the unit tests structurally cannot cover. ``FakeComm`` drives peers
sequentially, so it has no notion of two ranks blocked in *different*
collectives -- which is the failure mode seven bugs in this work shared, every
one of which hung rather than erroring.

``--inject verify-pin``
    One rank is given a device it did not select, so ``verify_pin`` fails
    there and nowhere else. The ``allreduce`` inside ``discover_peers`` covers
    only GPU workers on this node, so EB and smd0 ranks can only be reached by
    ``gpu_error_handler`` aborting the communicator. This is the only test of
    that, and it is the reason a discovery failure costs a diagnostic rather
    than a whole allocation's wall time.

``--inject follower-import``
    One follower's ``ipcOpenMemHandle`` raises. The group must degrade to
    private copies *together*: if the owner falls back and a follower does
    not, the follower reads memory the owner has freed -- silent corruption
    rather than a hang. Also exercises the owner's half of the fallback
    ordering (importers close, Barrier, owner frees), which the unit test
    scopes away because the owner has already returned there.

Timing is part of the contract. A correct abort and a regressed deadlock
produce near-identical logs; the difference is two seconds versus the whole
timeout. The launcher asserts elapsed time, so a hang cannot pass as an abort.

Run::

    mpirun -n 3 python mpi_failure_paths.py --inject follower-import
"""
import argparse
import json
import os
import sys
import time

import numpy as np


START = time.monotonic()


def emit(tag, obj):
    obj = dict(obj)
    obj['elapsed'] = round(time.monotonic() - START, 3)
    print(f'FAILURE_{tag} ' + json.dumps(obj, indent=2, sort_keys=True),
          flush=True)


SHAPE = (3, 4, 128, 128)          # ~768 KiB float32 per constant


def constants():
    peds = np.arange(int(np.prod(SHAPE)), dtype=np.float32).reshape(SHAPE)
    return {'jf': {'pedestals': peds,
                   'pixel_gain': (peds * 0.5 + 1.0).astype(np.float32)}}


def world_rank():
    for name in ('OMPI_COMM_WORLD_RANK', 'PMIX_RANK', 'PMI_RANK',
                 'SLURM_PROCID'):
        value = os.environ.get(name)
        if value is not None and value.strip().isdigit():
            return int(value)
    return None


# ---------------------------------------------------------------------------
# Case 1: verify_pin fails on one rank only
# ---------------------------------------------------------------------------

def case_verify_pin(rank, eb_ranks):
    """Give one GPU worker a device it did not select.

    Expected: every rank dies promptly. The injected rank raises from
    discover_peers; its node peers learn through the allreduce; EB and smd0
    can only be reached by the communicator abort.
    """
    import dataclasses

    from psana.gpu.gpu_placement import PinnedDevice, discover_peers, pin_device

    is_worker = rank >= eb_ranks
    pinned = pin_device() if is_worker else PinnedDevice()

    # The first GPU worker claims a PCI id that does not exist on this node.
    target = eb_ranks
    if rank == target and pinned.pinned:
        pinned = dataclasses.replace(pinned, requested_pci='0:ff:00.0')
        emit(f'RANK{rank}', {'injected': 'bogus requested_pci',
                             'note': 'verify_pin must fail here'})

    from mpi4py import MPI
    from psana.gpu.gpu_mpi import gpu_error_handler

    comm = MPI.COMM_WORLD
    emit(f'RANK{rank}', {'phase': 'entering discovery', 'worker': is_worker})

    # Wrapped exactly as mpi_ds._discover_gpu_placement wraps it. Without the
    # abort, non-GPU ranks block in their next psana_comm collective.
    with gpu_error_handler(comm):
        discover_peers(pinned, comm, is_gpu_worker=is_worker)

    # Reaching here means no rank aborted, which is the bug.
    emit(f'RANK{rank}', {'phase': 'SURVIVED discovery',
                         'failure': 'a rank with the wrong device was accepted'})
    return 1


# ---------------------------------------------------------------------------
# Case 2: a follower's IPC import fails
# ---------------------------------------------------------------------------

def case_follower_import(rank, eb_ranks):
    """Break ipcOpenMemHandle on one follower and require a uniform degrade."""
    from psana.gpu.gpu_placement import PinnedDevice, discover_peers, pin_device

    is_worker = rank >= eb_ranks
    pinned = pin_device() if is_worker else PinnedDevice()

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    placement = discover_peers(pinned, comm, is_gpu_worker=is_worker)
    if not is_worker:
        return 0

    peers = placement.device_comm
    if placement.n_device_peers < 2 or peers is None:
        emit(f'RANK{rank}', {'skipped': 'needs >= 2 peers on one device'})
        return 0

    import psana.gpu.gpu_shared_constants as gsc
    from psana.gpu.gpu_budget import _GpuBudget

    # Inject on the LAST follower, so the owner and at least one other
    # follower have already imported successfully when it fails.
    victim = placement.n_device_peers - 1
    injected = peers.Get_rank() == victim
    if injected:
        original = gsc._ImportedBlock

        class Failing(original):
            def __init__(self, *args, **kwargs):
                raise RuntimeError('injected ipcOpenMemHandle failure')

        gsc._ImportedBlock = Failing
        emit(f'RANK{rank}', {'injected': 'ipcOpenMemHandle raises',
                             'device_rank': victim})

    host = constants()
    declared = [('jf', 'pedestals'), ('jf', 'pixel_gain')]
    budget = _GpuBudget(limit_bytes=placement.usable_bytes
                        // placement.n_device_peers)
    shared = gsc.SharedRequestedConstants(declared, budget, placement)

    failures, checks = [], {}
    info = {'rank': rank, 'device_rank': peers.Get_rank(),
            'is_owner': placement.is_owner, 'injected': injected}

    shared.refresh(host)

    info['shared_selectors'] = [list(s) for s in shared.shared_selectors]
    info['private_selectors'] = [list(s) for s in shared.private_selectors]
    info['ipc_error'] = shared._ipc_error
    info['imported_bytes'] = int(shared.imported_bytes)
    info['charged_bytes'] = int(budget.committed())

    # THE assertion: every peer must have reached the same conclusion. A
    # partial degrade leaves a follower reading memory the owner has freed.
    counts = peers.allgather(len(shared.shared_selectors))
    info['shared_counts'] = counts
    checks['degraded_uniformly'] = len(set(counts)) == 1
    if not checks['degraded_uniformly']:
        failures.append(f'peers degraded inconsistently: {counts}; a follower '
                        'may be reading memory the owner has freed')

    checks['fell_back_to_private'] = len(shared.shared_selectors) == 0
    if shared.shared_selectors:
        failures.append('sharing survived an injected IPC failure')

    # Values must still be correct, through the private path.
    for selector in declared:
        got = np.asarray(shared.get(*selector).get())
        if not np.array_equal(got, host[selector[0]][selector[1]]):
            failures.append(f'{selector}: wrong values after degrading')
    checks['values_correct_after_degrade'] = not failures

    # Every rank owns its own copies now, so each is charged and none imports.
    checks['charged_after_degrade'] = budget.committed() > 0
    checks['nothing_imported'] = shared.imported_bytes == 0
    if not checks['charged_after_degrade']:
        failures.append('no bytes charged after falling back to private copies')
    if shared.imported_bytes:
        failures.append(f'{shared.imported_bytes} bytes still imported after '
                        'the fallback')

    # Teardown must still be clean: the fallback already closed and freed, so
    # close() must not double-free.
    peers.Barrier()
    try:
        shared.close()
        checks['close_after_degrade_safe'] = True
    except Exception as exc:                      # noqa: BLE001
        checks['close_after_degrade_safe'] = False
        failures.append(f'close() after degrading raised {type(exc).__name__}: {exc}')

    info['checks'] = checks
    info['failures'] = failures
    info['PASS'] = not failures
    emit(f'RANK{rank}', info)

    gathered = peers.gather(info, root=0)
    if placement.is_owner and gathered is not None:
        members = [r for r in gathered if r]
        emit('SUMMARY', {
            'peers': len(members),
            'all_pass': all(m['PASS'] for m in members),
            'failures': [f for m in members for f in m.get('failures', [])],
            'shared_counts': counts,
            'checks_by_device_rank': {m['device_rank']: m['checks']
                                      for m in members},
        })
    return 0 if not failures else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inject', required=True,
                    choices=('verify-pin', 'follower-import'))
    ap.add_argument('--eb-ranks', type=int, default=1)
    args = ap.parse_args()

    rank = world_rank()
    if rank is None:
        print('cannot determine world rank', file=sys.stderr)
        return 2

    if args.inject == 'verify-pin':
        return case_verify_pin(rank, args.eb_ranks)
    return case_follower_import(rank, args.eb_ranks)


if __name__ == '__main__':
    sys.exit(main())
