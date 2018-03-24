from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
worker_id = rank - 1
size = comm.Get_size()
nworkers = size - 1

# issues:
# - when n workers send on a heartbeat ideally shouldn't all be on a node
# - how can we learn "events_per_s" for each detector?  maybe readout groups?
# - there is no guarantee of precise number of events because of load-balancing
#   deadtime. in principle the collector could know about the statistics of
#   all workers and do something more precise.

def worker():

    reqevent_per_s = 5    # user defined
    event_per_s    = 100  # per-detector.  can we reliably learn this?
    hb_per_s       = 10   # fixed by the daq

    num_hb         = 30

    event_per_hb = event_per_s/hb_per_s

    # switch from units of Hz and total-events to "per-worker" and "heartbeats"
    worker_event_per_s     = event_per_s/nworkers
    worker_event_per_hb    = worker_event_per_s/hb_per_s
    worker_reqevent_per_s  = reqevent_per_s/nworkers
    worker_reqevent_per_hb = worker_reqevent_per_s/hb_per_s

    worker_nevent_to_send_per_hb = min(worker_event_per_hb,worker_reqevent_per_hb)
    nsend_all_workers_per_hb = round(worker_nevent_to_send_per_hb*nworkers)
    if worker_id == 0: print ('worker_nevent_to_send_per_hb',worker_nevent_to_send_per_hb,'nsend_all_workers_per_hb',nsend_all_workers_per_hb)
    for hb in range(num_hb):
        if nsend_all_workers_per_hb==0:
            hb_period = hb_per_s/reqevent_per_s
            if hb%hb_period==0:
                if (hb/hb_period)%nworkers==worker_id: print('worker',worker_id,'hb',hb,1)
        elif worker_nevent_to_send_per_hb<1:
            first_sending_worker_id = (hb*nsend_all_workers_per_hb)%nworkers
            # ideally the senders would be "sparse" so not only one node is sending
            # at a time
            tmp = range(first_sending_worker_id,first_sending_worker_id+nsend_all_workers_per_hb)
            worker_ids = [id%nworkers for id in tmp]
            if worker_id in worker_ids:
                print('worker',worker_id,'hb',hb,1)
        else:
            print (worker_id,hb,(round(worker_nevent_to_send_per_hb)))

if rank!=0:
    worker()

MPI.Finalize()
