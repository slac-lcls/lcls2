from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank == 2:
    import cupy as cp

    value = cp.arange(1024, dtype=cp.float32).sum()
    cp.cuda.get_current_stream().synchronize()
    props = cp.cuda.runtime.getDeviceProperties(0)
    print(
        "gpu_test_ok "
        f"rank={rank} name={props['name'].decode()} bus={props['pciBusID']} "
        f"sum={float(value.get())}",
        flush=True,
    )
comm.Barrier()
print(f"rank_done={rank}", flush=True)
