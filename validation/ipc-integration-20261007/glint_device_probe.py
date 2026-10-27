"""Would a GLINT adapter actually get what it needs from run.gpu_device?

GLINT #256's contract on the psana side: keep the mask, follow psana's
device, never assume cuda:0, verify physical identity. Checked against a real
DataSource, which the unit tests cannot do -- they use fakes, so they cannot
show the bus id names the card this rank's allocations land on.

Not in run_cases.py: it needs two masks to mean anything, so it is run by
hand twice and compared. Committed because b0103a3ad's message cites its
measurement.

Run on an A100 node, once wide and once reversed::

    mpirun -n 5 python validation/ipc-integration-20261007/glint_device_probe.py
    CUDA_VISIBLE_DEVICES=1,0 mpirun -n 5 \\
        python validation/ipc-integration-20261007/glint_device_probe.py

Measured on sdfampere017: under the reversed mask the ordinal moved 0 -> 1
while `pci_bus_id` stayed `0:81:00.0`, and a 1 MiB CuPy allocation landed on
that same card both times. smd0 and EB reported None, as a CPU role must.
"""
import json
import os
import sys

from mpi4py import MPI
from psana import DataSource
from psana.gpu import GpuTask, GpuDevice


def unused(batch, stream):
    raise AssertionError('this probe checks setup, not callbacks')


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    task = GpuTask(unused, ['jungfrau.raw'], [('jungfrau', 'pedestals')])
    ds = DataSource(exp='mfx100848724', run=51,
                    dir='/sdf/data/lcls/ds/prj/public01/xtc',
                    detectors=['jungfrau'], gpu_det='jungfrau', gpu_fn=task,
                    max_events=1)
    run = next(ds.runs())
    role = run.comms.node_type()

    dev = run.gpu_device
    fail = []
    info = {'rank': rank, 'role': role,
            'mask': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'has_device': dev is not None}

    if role != 'bd':
        # A CPU role has nothing to follow, and must say so rather than
        # reporting a device a framework would then try to use.
        if dev is not None:
            fail.append(f'a {role} rank reported a device; must be None')
    elif dev is None:
        fail.append('a BD worker got None; a framework has nothing to follow')
    else:
        info.update(pci=dev.pci_bus_id, ordinal=dev.ordinal, peers=dev.peers,
                    owner=dev.is_owner, host=dev.hostname)

        if not isinstance(dev, GpuDevice):
            fail.append('not the GpuDevice type exported from psana.gpu')

        # THE check: is the bus id it reports the card this rank's
        # allocations really land on? A view that names a device nothing was
        # selected on is the failure mode a framework cannot detect itself.
        import cupy as cp
        from psana.gpu.gpu_placement import device_pci
        probe = cp.empty(1 << 20, dtype=cp.uint8)
        actual = device_pci(cp, int(probe.device.id))
        del probe
        info['allocation_pci'] = actual
        if actual != dev.pci_bus_id:
            fail.append(f'gpu_device says {dev.pci_bus_id} but allocations '
                        f'land on {actual}')

        # Read-only, so a framework cannot diverge from the agreed placement.
        try:
            dev.peers = 99
        except AttributeError:
            pass
        else:
            fail.append('the view is writable')

    info['failures'] = fail
    info['PASS'] = not fail
    print('GLINT_PROBE ' + json.dumps(info, sort_keys=True), flush=True)

    rows = comm.gather(info, root=0)
    comm.Barrier()
    run.close_shared_memory()
    if rank == 0:
        bad = [f for r in rows if r for f in r['failures']]
        workers = [r for r in rows if r and r['role'] == 'bd']
        print('GLINT_SUMMARY ' + json.dumps(dict(
            ranks=len(rows), gpu_workers=len(workers),
            all_pass=not bad, failures=bad[:5],
            devices_by_rank={r['rank']: r.get('pci') for r in workers},
            peers_by_rank={r['rank']: r.get('peers') for r in workers}),
            sort_keys=True), flush=True)
    comm.Barrier()
    return 1 if fail else 0


if __name__ == '__main__':
    try:
        status = main()
    except BaseException:
        import traceback
        traceback.print_exc()
        if MPI.COMM_WORLD.size > 1:
            MPI.COMM_WORLD.Abort(1)
        raise
    if status and MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)
    sys.exit(status)
