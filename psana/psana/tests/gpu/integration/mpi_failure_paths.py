"""Failure paths on real MPI: do they abort, or do they hang?

Two cases the unit tests structurally cannot cover. ``FakeComm`` drives peers
sequentially, so it has no notion of two ranks blocked in *different*
collectives -- which is the failure mode seven bugs in this work shared, every
one of which hung rather than erroring.

``--inject select-device``
    ``Device.use()`` is made to fail on exactly one GPU worker. The ``allreduce`` inside ``discover_peers`` covers
    only GPU workers on this node, so EB and smd0 ranks can only be reached by
    ``gpu_error_handler`` aborting the communicator. This is the only test of
    that, and it is the reason a discovery failure costs a diagnostic rather
    than a whole allocation's wall time.

``--inject follower-import``
    One follower's ``ipcOpenMemHandle`` raises. Every peer must abort
    *together*, including those whose own import succeeded: the capability
    flag is reduced over the group, so a rank that carried on would read an
    allocation its peers have already freed -- silent corruption rather than
    a hang. Also exercises the release ordering before the raise (importers
    close, Barrier, owner frees), which the unit test scopes away because the
    owner has already returned there.

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
# Case 1: device selection fails on one rank only
# ---------------------------------------------------------------------------

def case_select_device(rank, eb_ranks):
    """Make Device.use() fail on exactly one GPU worker.

    Expected: every rank dies promptly. The injected rank raises from
    discover_peers; its node peers learn through the node allreduce; EB and
    smd0 can only be reached by the communicator abort, so this is the only
    test of that reach.

    Injected at the CuPy boundary rather than by corrupting a requested
    device id. Selection now happens inside discovery -- there is no
    caller-supplied device to corrupt -- and failing the real call is closer
    to what a wrong device or a busy context actually does.
    """
    from psana.gpu.gpu_placement import discover_peers

    is_worker = rank >= eb_ranks

    from mpi4py import MPI
    from psana.gpu.gpu_mpi import gpu_error_handler

    comm = MPI.COMM_WORLD
    cp = None
    target = eb_ranks                   # the first GPU worker
    if rank == target and is_worker:
        import cupy

        class FailsToSelect:
            """CuPy whose Device(...).use() raises, like a device this rank
            cannot actually take."""

            def __init__(self, real):
                self.cuda = _Cuda(real)

        class _Cuda:
            def __init__(self, real):
                self._real = real
                self.runtime = real.cuda.runtime

            def Device(self, ordinal=0):
                raise RuntimeError(
                    'injected: cudaSetDevice failed for this rank')

        cp = FailsToSelect(cupy)
        emit(f'RANK{rank}', {'injected': 'Device.use() raises',
                             'note': 'device selection must fail here'})

    emit(f'RANK{rank}', {'phase': 'entering discovery', 'worker': is_worker})

    # Wrapped exactly as mpi_ds._discover_gpu_placement wraps it. Without the
    # abort, non-GPU ranks block in their next psana_comm collective.
    with gpu_error_handler(comm):
        discover_peers(comm, is_gpu_worker=is_worker, cp=cp)

    # Reaching here means no rank aborted, which is the bug.
    emit(f'RANK{rank}', {'phase': 'SURVIVED discovery',
                         'failure': 'a rank that could not select a device '
                                    'was accepted'})
    return 1


# ---------------------------------------------------------------------------
# Case 2: a follower's IPC import fails
# ---------------------------------------------------------------------------

def case_follower_import(rank, eb_ranks):
    """Break ipcOpenMemHandle on one follower and require a uniform abort."""
    from psana.gpu.gpu_placement import discover_peers

    is_worker = rank >= eb_ranks

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    placement = discover_peers(comm, is_gpu_worker=is_worker)
    if not is_worker:
        return 0

    peers = placement.device_comm
    if placement.n_device_peers < 2 or peers is None:
        emit(f'RANK{rank}', {'skipped': 'needs >= 2 peers on one device'})
        return 0

    # Record the collectives so the abort path can be checked for divergence.
    # This is the case where it matters most: one rank fails its import and
    # the others do not, so if the release-then-raise sequence were not
    # identical across them the job would hang instead of aborting -- and the
    # sequential unit-test fake cannot model that.
    sys.path.insert(0, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    from collective_recorder import RecordingComm
    recorder = RecordingComm(peers, 'failure-paths')
    placement.device_comm = recorder
    peers = recorder

    import psana.gpu.gpu_shared_constants as gsc
    from psana.gpu.gpu_budget import _GpuBudget

    # Inject on the LAST follower, so the owner and at least one other
    # follower have already imported successfully when it fails -- which is
    # what makes "every peer aborts" a real assertion rather than one rank
    # reporting its own exception.
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

    # EVERY peer must raise, including the ones whose own import succeeded:
    # the capability flag is reduced over the group, so a rank that carried on
    # would be reading an allocation its peers have already freed. A rank that
    # raised alone would instead leave the others in the next collective.
    raised = None
    try:
        shared.refresh(host)
    except gsc.SharedConstantsError as exc:       # noqa: BLE001
        raised = f'{type(exc).__name__}: {exc}'
    except Exception as exc:                      # noqa: BLE001
        raised = f'WRONG TYPE {type(exc).__name__}: {exc}'

    info['raised'] = raised
    info['shared_selectors'] = [list(s) for s in shared.shared_selectors]
    info['ipc_error'] = shared._ipc_error
    info['imported_bytes'] = int(shared.imported_bytes)
    info['charged_bytes'] = int(budget.committed())

    # THE assertion: every peer reached the same conclusion, and that
    # conclusion was to stop.
    verdicts = peers.allgather(bool(raised))
    info['raised_by_peer'] = verdicts
    checks['every_peer_aborted'] = all(verdicts)
    if not all(verdicts):
        failures.append(
            f'peers disagreed about aborting: {verdicts}; a rank that carried '
            'on is reading memory its peers have freed')
    checks['raised_the_documented_error'] = (
        raised is not None and not raised.startswith('WRONG TYPE'))
    if raised is None:
        failures.append('an injected IPC failure did not raise on this rank')
    elif raised.startswith('WRONG TYPE'):
        failures.append(f'raised the wrong exception type: {raised}')

    # The message has to name the way to run without sharing, or a user whose
    # system cannot do IPC between ranks discovery grouped as peers has
    # nothing to try. (It is not a MIG remedy -- on MIG the peer count and
    # budget stay wrong -- so the message describes the cost, not a supported
    # MIG configuration.)
    checks['message_names_the_opt_out'] = (
        raised is not None and 'PSANA_GPU_SHARE_CONSTANTS=0' in raised)
    if not checks['message_names_the_opt_out']:
        failures.append(f'the abort does not name the opt-out: {raised}')

    # Nothing may be left mapped or allocated: _settle releases through
    # _release_shared, whose barriers every peer reaches, before raising.
    checks['nothing_left_imported'] = shared.imported_bytes == 0
    if shared.imported_bytes:
        failures.append(f'{shared.imported_bytes} bytes still imported after '
                        'the abort')
    checks['nothing_left_charged'] = budget.committed() == 0
    if budget.committed():
        failures.append(f'{budget.committed()} bytes still charged after the '
                        'abort')

    # Teardown must still be safe: the release already happened, so close()
    # must not double-free or barrier again.
    peers.Barrier()
    try:
        shared.close()
        checks['close_after_abort_safe'] = True
    except Exception as exc:                      # noqa: BLE001
        checks['close_after_abort_safe'] = False
        failures.append(f'close() after the abort raised '
                        f'{type(exc).__name__}: {exc}')

    # The abort must leave every peer having made the same calls in the same
    # order -- a release that ran on some ranks and not others is the hang
    # this whole case exists to rule out.
    checks['collective_order_agreed'] = recorder.check_agreement(
        'failure-paths/driver-end')
    if not checks['collective_order_agreed']:
        failures.append('peers made different collectives on the abort path; '
                        'see the "collective order diverged" line above')
    info['collectives'] = recorder.calls

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
            'raised_by_peer': verdicts,
            'checks_by_device_rank': {m['device_rank']: m['checks']
                                      for m in members},
        })
    # Hold every peer until the summary is complete. A rank with failures
    # calls MPI_Abort as soon as main() returns, which would otherwise kill
    # the summarising rank mid-emit and leave the launcher parsing a
    # truncated record.
    peers.Barrier()
    return 0 if not failures else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inject', required=True,
                    choices=('select-device', 'follower-import'))
    ap.add_argument('--eb-ranks', type=int, default=1)
    args = ap.parse_args()

    rank = world_rank()
    if rank is None:
        print('cannot determine world rank', file=sys.stderr)
        return 2

    if args.inject == 'select-device':
        return case_select_device(rank, args.eb_ranks)
    return case_follower_import(rank, args.eb_ranks)


if __name__ == '__main__':
    sys.exit(main())
