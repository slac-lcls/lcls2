"""
MPI-based test for testing timestamp-based access and standard event iteration using
prefetched Jungfrau calibration data in a distributed setup.

PREREQUISITES BEFORE RUNNING:
--------------------------------
1. Allocate 3 exclusive nodes for MPI jobs:
    salloc -N3 --exclusive --account=lcls:data -p milano
   Add the 3 nodes in slurm_host_test with this format:
   node0 slots=2
   node1 slots=120
   node2 slots=120

2. Create softlinks for *only* run 45 to avoid loading all runs:
   For smalldata:
       for i in $(seq 0 9); do
           ln -s /sdf/data/lcls/ds/mfx/mfx100852324/xtc/smalldata/mfx100852324-r0045-s00$i-c000.smd.xtc2 \
                 mfx100852324-r0045-s00$i-c000.smd.xtc2
       done

   For bigdata:
       for i in $(seq 0 9); do
           ln -s /sdf/data/lcls/ds/mfx/mfx100852324/xtc/bigdata/mfx100852324-r0045-s00$i-c000.xtc2 \
                 mfx100852324-r0045-s00$i-c000.xtc2
       done

3. Prefetch Jungfrau calibration data on a shared filesystem:
    calib_prefetch --xtc-dir /sdf/home/m/monarin/tmp/lcls2/psana/psana/tests/tmp_xtc \
                   -e mfx100852324 --log-level DEBUG --detectors jungfrau

4. Copy calibration files to each allocated node:
    scp /dev/shm/*.pkl sdfmilan048:/dev/shm
    # Repeat for each node in your salloc allocation (e.g., sdfmilan049, sdfmilan050, etc.)

USAGE:
-------
- Standalone:
    mpirun -n 242 --hostfile=slurm_host_test python run_build_table_performance.py
"""

import os
import numpy as np
from psana2 import DataSource
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
group = comm.Get_group()

n_bd_ranks = size - int(os.environ.get('PS_EB_NODES', '1')) - 1
xtc_dir = "/sdf/data/lcls/ds/mfx/mfx100852324/xtc"

def test_normal_ds():
    """Triggers subprocess MPI run for normal psana2 event loop test."""
    ds = DataSource(exp='mfx100852324', run=45, dir=xtc_dir, max_events=0,
                    use_calib_cache=True, cached_detectors=["jungfrau"],
                    log_level='INFO', batch_size=100)
    run = next(ds.runs())
    det = run.Detector('jungfrau')

    n_counters = 1
    sendbuf = np.zeros(n_counters, dtype="i")
    recvbuf = np.empty([size, n_counters], dtype="i") if rank == 0 else None

    comm.Barrier()
    t0 = MPI.Wtime()
    for i_evt, evt in enumerate(run.events()):
        img = det.raw.raw(evt)
        assert img is not None
        sendbuf[:] = [i_evt + 1]

    comm.Gather(sendbuf, recvbuf, root=0)
    comm.Barrier()
    if rank == 0:
        sum_events = np.sum(recvbuf, axis=0)[0]
        total_time = MPI.Wtime() - t0
        print(f"[test_normal_ds] Total time: {total_time:.2f}s. Events: {sum_events} Rate: {(sum_events / total_time) * 1e-3:.2f}kHz")


def test_ts_access_ds():
    ds = DataSource(exp='mfx100852324', run=45, dir=xtc_dir, max_events=0,
                    use_calib_cache=True, cached_detectors=["jungfrau"],
                    log_level='INFO')
    run = next(ds.runs())
    det = run.Detector('jungfrau')

    comm.Barrier()
    t0 = MPI.Wtime()

    bd_group = group.Excl([0, 1])
    bd_comm = comm.Create(bd_group)
    sum_events = 0

    with run.build_table() as success:
        if success:
            bd_offset = int(os.environ.get('PS_EB_NODES', '1')) + 1
            bd_rank = rank - bd_offset

            if bd_rank >= 0:
                bd_comm_rank = bd_comm.Get_rank()
                bd_comm_size = bd_comm.Get_size()
                t_read_start = MPI.Wtime()
                cn_events = 0

                sendbuf = np.zeros(1, dtype="i")
                recvbuf = np.empty([bd_comm_size, 1], dtype="i") if bd_comm_rank == 0 else None

                if bd_comm_rank == 0:
                    valid_ts = sorted(k for k in run._ts_table if run._ts_table[k])
                    print(f'build_table took {MPI.Wtime() - t0:.2f}s. {len(valid_ts)=}')
                    for ts in valid_ts:
                        recv_rank = bd_comm.recv(source=MPI.ANY_SOURCE)
                        bd_comm.send(ts, dest=recv_rank)

                    for _ in range(1, bd_comm_size):
                        recv_rank = bd_comm.recv(source=MPI.ANY_SOURCE)
                        bd_comm.send(None, dest=recv_rank)

                else:
                    while True:
                        bd_comm.send(bd_comm_rank, dest=0)
                        ts = bd_comm.recv(source=0)
                        if ts is None:
                            break
                        evt = run.event(ts)
                        assert evt is not None and len(evt._dgrams) > 0
                        img = det.raw.raw(evt)
                        assert img is not None
                        cn_events += 1

                    sendbuf[:] = [cn_events]

                t_read_end = MPI.Wtime()
                print(f"[Rank {rank}] Processed {cn_events} events in {t_read_end - t_read_start:.2f}s. "
                      f"Rate={(cn_events / (t_read_end - t_read_start)) * 1e-3:.4f}kHz")

                bd_comm.Gather(sendbuf, recvbuf, root=0)
                if bd_comm_rank == 0:
                    sum_events = np.sum(recvbuf, axis=0)[0]
                    total_time = MPI.Wtime() - t0
                    print(f"[test_ts_access_ds] Total time: {total_time:.2f}s. Events: {sum_events} Rate: {(sum_events / total_time) * 1e-3:.2f}kHz")


if __name__ == "__main__":
    if os.path.exists(xtc_dir):
        if rank == 0:
            print("Running build_table performance")
        test_normal_ds()
        test_ts_access_ds()
    else:
        if rank == 0:
            print(f"XTC directory {xtc_dir} not found. Skipping tests.")
