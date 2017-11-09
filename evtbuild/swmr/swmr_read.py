from mpi4py import MPI
import h5py, glob, os, sys
import numpy as np, time
import argparse
from load_config import load_config
#logistical MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-n','--nersc', action='store_true', help="Copy to NERSC", dest="nersc", default=False)
args = parser.parse_args()

cfg = load_config('sconfig')


class master(object):
    def __init__(self):
        self.master_msg = msg()
        self.total_read = np.zeros(size-1)
        self.total_events = np.zeros(size-1)
        self.total_file_size = 0
        self.cum_event_num = 0
        self.file0_eof = False
        if args.nersc:
            self.hit_probability = 1
        else:
            self.hit_probability = float(cfg['hit_probability'])

        self.diode_lp = np.zeros(size-1)

    def find_evts(self,mfile, start_index, request_rank):
        client_rank = request_rank -1
        diode_dst = mfile['small_data']['diode_values']
        ts_dst = mfile['small_data']['timestamps']
        diode_dst.id.refresh()
        ts_dst.id.refresh()

        if request_rank == -1:
            dd = 0
        else:
            dd = self.diode_lp[client_rank]

        dd = int(dd)
        
#        dd=0
        number_of_events = diode_dst.shape[0]            
        print('Client %i dd is %i/%i ' % (client_rank, dd, number_of_events))
      #  print('Diode values', diode_dst[dd:])
        # Location of high diode values (>(1-hit_probability))
        notable_inds = np.where(np.array(diode_dst[dd:]) > 1-self.hit_probability)[0]
        notable_inds += dd
        
#        print('Notable inds', notable_inds)
 #       print('Length of timestamps', len(ts_dst))
        # Find timestamps of interesting events
        timestamps = np.take(np.array(ts_dst,dtype='uint32'), notable_inds, mode='clip').ravel()
  #      print('Timestamps', timestamps)
        ts_start = np.searchsorted(timestamps, start_index)+(np.sign(start_index)+1)/2

        # Check if the small data has finished writing
        # If it has, pass the final length to the clients
        # so they may finish reading
        if self.file0_eof:
            final_length = len(np.where(np.array(diode_dst) > 1-self.hit_probability)[0])
            print('Final length is %i' % final_length)
        else:
            final_length = -1

        if request_rank == -1:
            self.diode_lp[:] = number_of_events
        else:
            self.diode_lp[client_rank] = number_of_events
            
        print('Sending to client', number_of_events, timestamps, final_length)    
        return number_of_events, timestamps, final_length


    def master(self):
        num_clients = size-1
        client_list = np.arange(num_clients) + 1

        master_file_name = '%s/swmr_small_data.h5' % (cfg['path'])

        total_read = 0
        with h5py.File(master_file_name, 'r', swmr=True) as master_file:
            # Find the interesting (Diode high) events in the small data
            self.master_msg.file_evt_num, self.master_msg.timestamps, \
                self.master_msg.cum_event_num = self.find_evts(master_file,-1,-1)
            
            for rank in client_list:
                client_obj = comm.recv(source=MPI.ANY_SOURCE)
                comm.send(self.master_msg, dest=client_obj.rank)
            while True:
                if num_clients == 0:
                    break

                client_obj = comm.recv(source=MPI.ANY_SOURCE)

                if client_obj.file0_eof:
                    self.file0_eof = True

                if client_obj.client_exit_flag:
                    cu_time = time.strftime("%H:%M:%S")
                    client_rank = client_obj.rank -1
                    print('Client %i exited from master at %s' % (client_rank,cu_time))
                    client_list = np.delete(client_list, np.searchsorted(client_list, client_obj.rank))
                    num_clients = len(client_list)
                    self.total_read[client_obj.rank-1] = client_obj.read_mb
                    self.total_events[client_obj.rank-1] = client_obj.total_events
                    self.total_file_size += client_obj.total_file_size
                    continue

                if client_obj.client_refresh:
                    self.master_msg.file_evt_num, self.master_msg.timestamps, \
                        self.master_msg.cum_event_num = self.find_evts(master_file, client_obj.last_timestamp, client_obj.rank)
                    comm.send(self.master_msg, dest=client_obj.rank)

                    
# Declare messaging object
class msg(object):
    def __init__(self):
        self.rank = rank
        self.client_refresh = False
        self.file_evt_num = 0
        self.timestamps = []
        self.last_timestamp = -1
        self.client_exit_flag = False
        self.read_mb = 0
        self.total_file_size = 0
        self.rank_active = True
        self.eof = False
        self.total_events = 0
        self.file0_eof = False

def client():
    client_rank = rank-1
    client_msg = msg()
    client_total_events = 0
    read_events = []
    file_name = '%s/swmr_file%i.h5' % (cfg['path'],rank-1)

    retained_evts = []
    eof = False

    print('Client %i reading file %s' % (client_rank, file_name))
     
    with h5py.File(file_name, 'r', swmr=True) as client_file:
        dset = client_file["data"]
        clt_tsmps = client_file["small_data"]["timestamps"]
        total_read_in = 0    
        while True:                        
            comm.send(client_msg, dest=0)
          #  if client_msg.client_exit_flag:
               # print('Client %i exiting' % client_rank)
           #     break

            evt_obj = comm.recv(source=0)
       
            if client_msg.client_refresh:
                dset.id.refresh()
                clt_tsmps.id.refresh()
            
            clt_tsmps_arr = np.array(clt_tsmps)
            evt_inds = evt_obj.timestamps
     #       evt_inds = clt_tsmps_arr[np.in1d(clt_tsmps_arr, evt_obj.timestamps)]
      #      if len(evt_inds) != len(evt_obj.timestamps):
              #  print('Client %i found %i our of %i total' % (client_rank, len(evt_inds), len(evt_obj.timestamps)))
       
              # Append any events that failed to read from the last loop
            # This can happen if the file opened lags the small data synchronized to file0 
            evt_inds = np.append(retained_evts, evt_inds)
         #   print(evt_inds)
            retained_evts = []

            client_start = time.time()
            # Read in the filtered events 
          #  re_evt=[]
            print('Client events %i' % client_rank, evt_inds)
            for evt_ct,evt in enumerate(evt_inds):
           #     print(evt)
                try:
                    read_in = dset[evt][:]
                    client_total_events += 1
                    read_events.append(evt)
                except ValueError:
                    retained_evts.append(evt)
#                    print('Event %i out of range, held over for rank %i' % (evt, client_rank))
                rin_mb = read_in.nbytes/10**6
                total_read_in += rin_mb
            client_end = time.time()
            read_events.append(-1)
#            read_events.append(re_evt)

            average_speed = rin_mb*len(evt_inds)/(client_end-client_start)
            if len(evt_inds) > 0:
                pass
               # print('Client %i read in %i events at %.2f MB/s' % (rank-1, len(evt_inds),average_speed))

            # Test if we've reached the end of the file
            if np.sum(dset[-1][:1000]) == 0:
                eof = True
                if client_rank==0:
                    client_msg.file0_eof = True
                
            # Done with reading the batch that the master provided
            # Wait for the reader to move ahead x GB
            # This is to avoid contention at the EOF
            
            # Wait for new data to be written out
            # Use an exit count. If the file size hasn't changed over 3 seconds
            # assume that the writer has finished
            # If the reader also runs out of events, close it. 
            
            exit_count = 0
            prev_size = os.stat(file_name).st_size/10**6

            while client_msg.rank_active and not eof:
                if exit_count > 1:
                    #print('File %i stopped writing' % client_rank)
                    client_msg.rank_active = False
                    break

                size_file = os.stat(file_name).st_size/10**6
             
                if size_file - 1000 > prev_size:
                    break
                
                time.sleep(0.1)
                size_change = os.stat(file_name).st_size/10**6 - size_file

                if size_change == 0:
                    exit_count+=1
                else:
                    exit_count = 0
                    
            # If we've reached the end of the file and read out all the events
            # designated by the master, pass the close event flags to master
            if eof and client_total_events == evt_obj.cum_event_num:
                cu_time = time.strftime("%H:%M:%S")
                exit_obj = msg()
                exit_obj.file0_eof = client_msg.file0_eof
                print('Client %i passing exit flag at %s' % (client_rank, cu_time))
                exit_obj.client_exit_flag = True
                exit_obj.read_mb = total_read_in
                exit_obj.total_file_size = os.stat(file_name).st_size/10**6
                exit_obj.total_events = client_total_events
                comm.send(exit_obj, dest = 0)
                break
                
            
            # Else ask for more events from the master
            client_msg.client_refresh = True
            try:
                client_msg.last_timestamp = int(evt_inds[-1])
            except IndexError:
                pass
    # When the client has finished, quit the process
    sys.exit()
            # red = np.array(read_events)
    # red = red.ravel()
    # np.savetxt('client_%i_events.txt' % client_rank, red)


comm.Barrier()

try:
    if rank == 0:
      
        global_start = time.time()
        rm = master()
        rm.master()
    else:
        client()
  #  comm.Barrier()
finally:
    if rank == 0:
        global_end = time.time()
        elapsed_time = global_end - global_start
        
        read_gb = float(np.sum(rm.total_read))/1000
        total_read = float(rm.total_file_size)/1000

        av_spd = read_gb/elapsed_time
        total_spd = total_read/elapsed_time

        print('\n'+'-'*40)
        print('Elapsed time %f s' % elapsed_time)
        print('Number of clients %i' % (size-1))
        print('Read %.2f GB at an average of %.2f GB/s' % (read_gb, av_spd))
        print('Filtered %.2f GB at an average of %.2f GB/s' % (total_read, total_spd))
        print('-'*40+'\n')
        for ct, (client_event,client_total) in enumerate(zip(rm.total_events, rm.total_read)):
            client_gb = client_total/1000
            print('Client %i read in %.2f GB and %i events' % (ct, client_gb, client_event))
