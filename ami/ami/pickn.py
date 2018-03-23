from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
worker_id = rank - 1
size = comm.Get_size()
nworkers = size - 1

def worker():

    reqevent_per_s = 5
    hb_per_s       = 10
    event_per_s    = 100
    num_hb         = 10

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
                if worker_id==0: print('worker',worker_id,'hb',hb,1)
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
