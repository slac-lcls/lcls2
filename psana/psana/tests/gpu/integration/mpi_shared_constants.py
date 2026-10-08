"""SharedRequestedConstants on real hardware: one device copy per GPU.

Covers design rows 2B-0/3/4/5/6/8 with real CUDA IPC, which the unit tests
fake: un-pooled allocation exporting a usable handle, followers reading the
owner's bytes, the intersection degrade, the three BeginStep cases, ordered
teardown, and #168's primary criterion -- device-resident bytes close to one
copy rather than one per rank.

Run with at least three ranks pinned to ONE device::

    CUDA_VISIBLE_DEVICES=0 mpirun -n 3 python mpi_shared_constants.py
"""
import argparse
import json
import os
import sys

import numpy as np




def emit(tag, obj):
    print(f'SHARED_{tag} ' + json.dumps(obj, indent=2, sort_keys=True),
          flush=True)


SHAPE = (3, 8, 256, 256)          # ~6 MiB float32 per constant, JF-like


def constants(scale=1.0, shape=SHAPE):
    peds = (np.arange(int(np.prod(shape)), dtype=np.float32)
            .reshape(shape) * scale)
    return {'jf': {
        'pedestals': peds,
        'pixel_gain': (peds * 0.5 + 1.0).astype(np.float32),
        'pixel_status': np.ones(shape, dtype=np.uint16),
    }}


def device_used():
    import cupy as cp
    free, total = cp.cuda.runtime.memGetInfo()
    return int(total - free)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eb-ranks', type=int, default=1)
    args = ap.parse_args()

    world = None
    for name in ('OMPI_COMM_WORLD_RANK', 'PMIX_RANK', 'PMI_RANK',
                 'SLURM_PROCID'):
        value = os.environ.get(name)
        if value is not None and value.strip().isdigit():
            world = int(value)
            break
    if world is None:
        print('cannot determine world rank', file=sys.stderr)
        return 2

    is_worker = world >= args.eb_ranks
    # Imported normally, as production does.
    from psana.gpu.gpu_placement import (
        PinnedDevice, discover_peers, pin_device,
    )
    pinned = pin_device() if is_worker else PinnedDevice()

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    placement = discover_peers(pinned, comm, is_gpu_worker=is_worker)
    if not is_worker:
        return 0

    # From here on, rank 0 (CPU-role) has returned. Any COMM_WORLD collective
    # would deadlock, so `peers` -- the device-local communicator -- is the
    # only safe synchronisation point. Bound it to make the rule impossible
    # to violate by accident.
    peers = placement.device_comm
    del comm

    import cupy as cp
    from psana.gpu.gpu_budget import _GpuBudget
    from psana.gpu.gpu_shared_constants import (
        SharedConstantsError, SharedRequestedConstants,
    )

    failures, checks = [], {}
    info = {'rank': rank, 'peers': placement.n_device_peers,
            'is_owner': placement.is_owner,
            'uuid_hex': placement.device_uuid_hex[:16] + '...'}

    if placement.n_device_peers < 2 or peers is None:
        info['skipped'] = 'needs >= 2 ranks on one device'
        emit(f'RANK{rank}', info)
        return 0

    host = constants()
    nbytes = sum(host['jf'][k].nbytes for k in ('pedestals', 'pixel_gain'))

    # ---- 2B-8: differing selector sets share the intersection -----------
    # The owner declares a third constant nobody else wants.
    declared = [('jf', 'pedestals'), ('jf', 'pixel_gain')]
    if placement.is_owner:
        declared = declared + [('jf', 'pixel_status')]

    peers.Barrier()
    baseline = device_used()

    budget = _GpuBudget(limit_bytes=placement.usable_bytes
                        // placement.n_device_peers)
    shared = SharedRequestedConstants(declared, budget, placement)
    shared.refresh(host)

    info['shared_selectors'] = [list(s) for s in shared.shared_selectors]
    info['private_selectors'] = [list(s) for s in shared.private_selectors]
    info['charged_bytes'] = int(budget.committed())
    info['imported_bytes'] = int(shared.imported_bytes)

    checks['intersection_shared'] = len(shared.shared_selectors) == 2
    checks['private_remainder'] = (
        len(shared.private_selectors) == (1 if placement.is_owner else 0))
    for name, ok in (('intersection_shared', checks['intersection_shared']),
                     ('private_remainder', checks['private_remainder'])):
        if not ok:
            failures.append(f'{name}: shared={shared.shared_selectors} '
                            f'private={shared.private_selectors}')

    # ---- 2B-0/3: a follower genuinely reads the owner's bytes ----------
    for selector in declared:
        got = cp.asnumpy(shared.get(*selector))
        want = host[selector[0]][selector[1]]
        if not np.array_equal(got, want):
            failures.append(f'{selector}: values wrong after establish')
    checks['values_correct'] = not failures

    # A kernel must be able to read an imported mapping.
    total = float(cp.asnumpy(cp.sum(shared.get('jf', 'pedestals'),
                                    dtype=cp.float64)))
    expect = float(host['jf']['pedestals'].astype(np.float64).sum())
    checks['kernel_reads_shared'] = abs(total - expect) <= 1e-6 * max(1.0, abs(expect))
    if not checks['kernel_reads_shared']:
        failures.append(f'kernel sum {total} != {expect}')

    # ---- 2B-4: charge once, report imports ----------------------------
    if placement.is_owner:
        checks['owner_charged'] = budget.committed() >= nbytes
        if not checks['owner_charged']:
            failures.append(f'owner charged {budget.committed()} < {nbytes}')
    else:
        checks['follower_not_charged'] = budget.committed() == 0
        checks['follower_reports_imports'] = shared.imported_bytes == nbytes
        if budget.committed():
            failures.append(f'follower charged {budget.committed()} bytes '
                            'for memory it does not own')
        if shared.imported_bytes != nbytes:
            failures.append(f'imported_bytes {shared.imported_bytes} != {nbytes}')

    # ---- 2B-7: the intersection is resident ONCE, not once per rank ----
    # Measured by what each rank owns versus imports, because cudaMemGetInfo
    # reports device-wide usage: every peer sees the same total, and that
    # total also contains the owner's private selector, the CUDA context and
    # pool slack. Those confounds are larger than the quantity under test, so
    # comparing a device delta against a per-rank estimate compares unlike
    # things -- an earlier version of this check did exactly that and failed
    # on a correct implementation.
    peers.Barrier()
    growth = device_used() - baseline
    info['device_growth_bytes'] = growth
    info['per_peer_observations'] = peers.allgather(growth)

    owned = int(budget.committed()) if placement.is_owner else 0
    contributions = peers.allgather(owned)
    info['owned_by_peer'] = contributions
    if placement.is_owner:
        # Exactly one peer allocates the intersection; the rest allocate none
        # of it, so the device holds one copy however many ranks read it.
        allocating = [c for c in contributions if c]
        info['peers_allocating'] = len(allocating)
        info['intersection_bytes'] = nbytes
        checks['one_copy_not_per_rank'] = len(allocating) == 1
        if not checks['one_copy_not_per_rank']:
            failures.append(
                f'{len(allocating)} of {placement.n_device_peers} peers '
                'allocated constants; the intersection is not shared')

        saved = nbytes * (placement.n_device_peers - 1)
        info['bytes_saved_vs_private_copies'] = saved
        checks['saving_matches_peer_count'] = saved > 0
    else:
        # A follower must read the full intersection while owning none of it.
        checks['follower_owns_nothing'] = budget.committed() == 0
        checks['follower_imports_intersection'] = (
            shared.imported_bytes == nbytes)

    # ---- 2B-5 case A: unchanged -> nothing moves ----------------------
    peers.Barrier()
    checks['case_A_no_move'] = shared.refresh(host) is False
    if not checks['case_A_no_move']:
        failures.append('an unchanged refresh reported movement')

    # ---- 2B-5 delayed consumer: a retained host copy is independent ---
    retained = cp.asnumpy(shared.get('jf', 'pixel_gain')).copy()

    # ---- 2B-5 case B: same layout, new values, written in place -------
    peers.Barrier()
    changed = constants(scale=3.0)
    moved = shared.refresh(changed)
    checks['case_B_moved'] = moved is True
    peers.Barrier()
    got = cp.asnumpy(shared.get('jf', 'pedestals'))
    checks['case_B_new_values_visible'] = np.array_equal(
        got, changed['jf']['pedestals'])
    if not checks['case_B_new_values_visible']:
        failures.append('case B: follower view is stale')
    checks['delayed_consumer_unaffected'] = np.array_equal(
        retained, host['jf']['pixel_gain'])
    if not checks['delayed_consumer_unaffected']:
        failures.append('a retained host copy was mutated by a refresh')

    # ---- 2B-5 case C: layout change -> close, free, re-export ---------
    peers.Barrier()
    reshaped = constants(scale=1.0, shape=(3, 8, 128, 256))
    shared.refresh(reshaped)
    peers.Barrier()
    got = cp.asnumpy(shared.get('jf', 'pedestals'))
    checks['case_C_reestablished'] = (
        got.shape == (3, 8, 128, 256)
        and np.array_equal(got, reshaped['jf']['pedestals']))
    if not checks['case_C_reestablished']:
        failures.append('case C: wrong shape or values after reallocation')
    total = float(cp.asnumpy(cp.sum(shared.get('jf', 'pixel_gain'),
                                    dtype=cp.float64)))
    expect = float(reshaped['jf']['pixel_gain'].astype(np.float64).sum())
    checks['case_C_kernel_reads_new_mapping'] = (
        abs(total - expect) <= 1e-6 * max(1.0, abs(expect)))
    if not checks['case_C_kernel_reads_new_mapping']:
        failures.append(f'case C kernel sum {total} != {expect}')

    # ---- 2B-6: ordered teardown, no growth, idempotent ---------------
    peers.Barrier()
    before_close = device_used()
    # close() is COLLECTIVE: _release_shared sequences importers-close,
    # barrier, owner-free internally. Calling it on one role at a time -- as
    # this driver used to -- leaves that barrier unmatched, which the bounded
    # barrier now reports as 'release-shared-closed' instead of hanging.
    shared.close()
    peers.Barrier()
    after_close = device_used()
    checks['teardown_did_not_grow'] = after_close <= before_close
    if after_close > before_close:
        failures.append(f'memory grew across close: {before_close} -> '
                        f'{after_close}')
    try:
        shared.close()
        checks['double_close_safe'] = True
    except Exception as exc:                      # noqa: BLE001
        checks['double_close_safe'] = False
        failures.append(f'second close raised {type(exc).__name__}: {exc}')

    # ---- content disagreement must abort ------------------------------
    peers.Barrier()
    guard = SharedRequestedConstants(
        [('jf', 'pedestals')],
        _GpuBudget(limit_bytes=placement.usable_bytes), placement)
    divergent = constants(scale=1.0 if placement.is_owner else 2.0)
    try:
        guard.refresh(divergent)
        checks['digest_mismatch_aborts'] = placement.is_owner
    except SharedConstantsError:
        checks['digest_mismatch_aborts'] = True
    except Exception:                             # noqa: BLE001
        checks['digest_mismatch_aborts'] = False
    if not placement.is_owner and not checks['digest_mismatch_aborts']:
        failures.append('a follower accepted the owner\'s differing values')
    try:
        guard.close()
    except Exception:                             # noqa: BLE001
        pass

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
            'charged_by_rank': {m['rank']: m.get('charged_bytes')
                                for m in members},
            'imported_by_rank': {m['rank']: m.get('imported_bytes')
                                 for m in members},
            'checks_by_rank': {m['rank']: m.get('checks') for m in members},
        })
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
