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
        self.file0_length = 0
        self.last_index = 0
        self.final_length = -1
        self.sd_eof = False
        self.master_dump = open('dump/master_dump.txt', 'w')
        self.eof_lock = False
        self.last_dd = 0
        self.last_timestamps = []
        if args.nersc:
            self.hit_probability = 1
        else:
            self.hit_probability = float(cfg['hit_probability'])

        self.diode_lp = np.zeros(size-1)

    def find_evts(self,mfile, start_index, request_rank, num_read_events):
        
        client_rank = request_rank -1

        
        if not self.sd_eof:
            self.diode_dst.id.refresh()
            self.ts_dst.id.refresh()

        # check if EOF
        if self.diode_dst[-1] == -1 and not self.sd_eof:
            print('Client %i found the end of the small data file' % client_rank)
            self.sd_eof = True

        print('Client %i read %i events' % (client_rank, num_read_events))
        try:
            if self.sd_eof and num_read_events == self.final_length:
                self.master_dump.write('Client %i at end of file. Skipping small data\n' % client_rank)
                return [],[],self.file0_length
        except IndexError:
            pass

        if request_rank == -1:
            dd = 0
        else:
            dd = self.diode_lp[client_rank]

        dd = int(dd)
        
#        dd=0
        number_of_events = self.diode_dst.shape[0]            
      # print('Client %i dd is %i/%i ' % (client_rank, dd, number_of_events))
      #  print('Diode values', diode_dst[dd:])
        # Location of high diode values (>(1-hit_probability))
        notable_inds = np.where(np.array(self.diode_dst[dd:]) > 1-self.hit_probability)[0]
        notable_inds += dd
        
#        print('Notable inds', notable_inds)
 #       print('Length of timestamps', len(ts_dst))
        # Find timestamps of interesting events
        timestamps = np.take(np.array(self.ts_dst,dtype='uint32'), notable_inds, mode='clip').ravel()
        
        # Check if the small data has finished writing
        # If it has, pass the final length to the clients
        # so they may finish reading

        if request_rank == -1:
            self.diode_lp[:] = number_of_events
            self.file0_length = len(timestamps)
        else:
            self.diode_lp[client_rank] = number_of_events

        if request_rank == 1:
               self.file0_length += len(timestamps)


        if self.sd_eof and not self.eof_lock:
            self.final_length = len(np.where(np.array(self.diode_dst) > 1-self.hit_probability)[0])

            print('Final length is %i' % self.final_length)
            self.eof_lock = True
   
        return number_of_events, timestamps, self.final_length


    def master(self):
        num_clients = size-1
        client_list = np.arange(num_clients) + 1

        master_file_name = '%s/swmr_small_data.h5' % (cfg['path'])

        total_read = 0
               
        with h5py.File(master_file_name, 'r', swmr=True) as master_file:

            self.diode_dst = master_file['small_data']['diode_values']
            self.ts_dst = master_file['small_data']['timestamps']

        
            # Find the interesting (Diode high) events in the small data
            self.master_msg.file_evt_num, self.master_msg.timestamps, \
                self.master_msg.cum_event_num = self.find_evts(master_file,-1,-1,0)
            
            for rank in client_list:
                client_obj = comm.recv(source=MPI.ANY_SOURCE)
                comm.send(self.master_msg, dest=client_obj.rank)
            while True:
                if num_clients == 0:
                    break

                client_obj = comm.recv(source=MPI.ANY_SOURCE)
                client_rank = client_obj.rank -1
                
                self.master_dump.write('Received from rank %i at time %s\n' % (client_rank,time.strftime("%H:%M:%S")))
                self.master_dump.write('\t Rank %i has read %i events' % (client_rank,client_obj.num_read_events))
                if client_obj.file0_eof:
                    self.file0_eof = True

                if client_obj.client_exit_flag:
                    cu_time = time.strftime("%H:%M:%S")
                    self.master_dump.write('\tClient %i is exiting at %s\n' % (client_rank, cu_time))
                    print('Client %i exited from master at %s' % (client_rank,cu_time))
                    client_list = np.delete(client_list, np.searchsorted(client_list, client_obj.rank))
                    num_clients = len(client_list)
                    self.total_read[client_obj.rank-1] = client_obj.read_mb
                    self.total_events[client_obj.rank-1] = client_obj.total_events
                    self.total_file_size += client_obj.total_file_size
                    continue

                if client_obj.client_refresh:
                    self.master_msg.file_evt_num, self.master_msg.timestamps, \
                        self.master_msg.cum_event_num = self.find_evts(master_file, client_obj.last_timestamp, client_obj.rank, client_obj.num_read_events)
                    comm.send(self.master_msg, dest=client_obj.rank)
                    
                self.master_dump.write('\tSent %i events to rank %i at time %s\n' % (len(self.master_msg.timestamps), client_rank,time.strftime("%H:%M:%S")))
        self.master_dump.close()
        
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
        self.num_read_events = 0

def client():
    client_rank = rank-1
    client_msg = msg()
    client_total_events = 0
    read_events = []
    file_name = '%s/swmr_file%i.h5' % (cfg['path'],rank-1)

    retained_evts = []
    eof = False

#\    print('Client %i reading file %s' % (client_rank, file_name))
    txt_dump  = open('dump/%i_rank_dump.txt' % client_rank, 'w') 
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
           # print('Client events %i' % client_rank, evt_inds)

            for evt_ct,evt in enumerate(evt_inds):
           #     print(evt)
                try:
                    read_in = dset[evt][:]
                    rin_mb = read_in.nbytes/10.0**6
                    client_total_events += 1
                    read_events.append(evt)
                except ValueError:
                    retained_evts.append(evt)
                    rin_mb = 0
#                    print('Event %i out of range, held over for rank %i' % (evt, client_rank))
                total_read_in += rin_mb
                #print('Client %i read in event %i' % (rank-1, evt_ct))
                
            client_end = time.time()
            client_msg.num_read_events = client_total_events
            
            read_events.append(-1)
#            read_events.append(re_evt)

            average_speed = rin_mb*len(evt_inds)/(client_end-client_start)
            if len(evt_inds) > 0:
                pass
               # print('Client %i read in %i events at %.2f MB/s' % (rank-1, len(evt_inds),average_speed))
            cu_time = time.strftime("%H:%M:%S")
            txt_dump.write('%s: read in %i events\n' % (cu_time, client_total_events))
            
            # Test if we've reached the end of the file
            try:
                if np.sum(dset[-1][:1000]) == 0:
                    eof = True
                    if client_rank==0:
                        client_msg.file0_eof = True
            except ValueError:
                pass
            # Done with reading the batch that the master provided
            # Wait for the reader to move ahead x GB
            # This is to avoid contention at the EOF
            
            # Wait for new data to be written out
            # Use an exit count. If the file size hasn't changed over 3 seconds
            # assume that the writer has finished
            # If the reader also runs out of events, close it. 
            
            exit_count = 0
            prev_size = os.stat(file_name).st_size/10**6

            # Loop to keep the readers away from the end of the file
            # There is a special case when the file is done writing
            # The file size isn't increasing anymore, so we need to
            # exit from the loop
            while client_msg.rank_active and not eof:
                if exit_count > 10:
                    print('File %i stopped writing' % client_rank)
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
                txt_dump.write('Passing exit flag at %s\n' % cu_time)
                
                exit_obj = msg()
                exit_obj.num_read_events = client_msg.num_read_events
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
    
    # red = np.array(read_events)
    # red = red.ravel()
    # np.savetxt('client_%i_events.txt' % client_rank, red)
    txt_dump.write('Done at' + time.strftime("%H:%M:%S")+'\n')
    txt_dump.close()
    sys.exit()

comm.Barrier()
if rank == 0:
    #Wait until the hdf files appear, and thus have the lock released
    while True:
        files = glob.glob(cfg['path']+'/swmr_*')
        file_size = [os.path.getsize(file) for file in files]

        if len(files) == size and np.prod(file_size)>0:
            print('-'*40+'\n')
            print('Done waiting for files to appear')
            print('-'*40+'\n')
            # Some kind of race condition here
            # Waiting for the files to be created isn't enough
            # Read process accesses the file before the writer has done
            # initializing it. No safe exception within h5py
#            time.sleep(8)
            break
        else:
            print('There are %i files' % len(files))
            time.sleep(0.2)
    
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
        print('Read completed at %s' % time.strftime("%H:%M:%S"))
        print('Elapsed time %f s' % elapsed_time)
        print('Number of clients %i' % (size-1))
        mode_String = 'Filtered'
        if args.nersc:
            mode_String = 'Copied'
        
        print('%s %.2f GB at an average of %.2f GB/s' % (mode_String, read_gb, av_spd))
#        print('Filtered %.2f GB at an average of %.2f GB/s' % (total_read, total_spd))
        print('-'*40+'\n')
        for ct, (client_event,client_total) in enumerate(zip(rm.total_events, rm.total_read)):
            client_gb = client_total/1000
            print('Client %i read in %.2f GB and %i events' % (ct, client_gb, client_event))
