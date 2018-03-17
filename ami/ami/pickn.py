from mpi4py import MPI

comm = MPI.COMM_WORLD
worker_id = comm.Get_rank() - 1
size = comm.Get_size()
nworkers = size - 1


reqevent_per_s = 50
hb_per_s = 10
events_per_s = 100

events_per_hb = event_per_s//hb_per_s

# switch from units of Hz and total-events to "per-worker" and "heartbeats"
worker_event_per_s = event_per_s//nworker  # number of events per second per worker
worker_event_per_hb = worker_event_per_s//hb_per_s
worker_reqevent_per_s = reqevent_per_s//nworker  # number of events per second per worker
worker_reqevent_per_hb = worker_reqevent_per_s//hb_per_s

# when worker_event_per_hb is small, need to choose based on rank, but round-robin it to different ranks
if worker_event_per_hb < 1:
    # we don't send on every heartbeat
    worker_hb_send_period =  hb_per_s // worker_event_per_s # e.g. 3hb per event per worker
else:
    worker_hb_send_period = 1

nevent=0

for hb in range(30):
    # this line emulates gathering
    nevent += min(worker_event_per_hb,worker_reqevent_per_hb)

    if hb % worker_hb_send_period != 0: continue
    if worker_event_per_hb < 1:
        if hb % nworker == worker_id:
            print('hb send', worker_id, hb)
