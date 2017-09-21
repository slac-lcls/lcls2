'''
Here is the analysis script in the offline event builder trilogy
This code will distribute the matched indices from the pickle file
to client cores that will retrieve the event data corresponding to
the indices.
'''
#from psana import *

from mpi4py import MPI
import h5py, time, sys, pickle, os, glob
import numpy as np
from numba import jit
from picklecreate import load_config

#logistical MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
assert size>1, 'At least 2 MPI ranks required'



global batch_size
batch_size = int(sys.argv[1])
try:
  scr_dir = str(sys.argv[2])
except IndexError:
#  print('No options')
  sys.exit()

comm.Barrier()

#load the pickle
global data_dict
config_dict = load_config()

path = config_dict['path'] + '/' + scr_dir

file_Name = path+"/eventpickle"
data_dict = pickle.load(open(file_Name, 'r'))


#open every h5 file
files=[]
total_file_size = 0

file_list = glob.glob(path+'/*.h5')
file_list = np.sort(file_list)
for filename in file_list:
  file_size = float(os.stat(filename)[6])
  total_file_size += file_size
  f = h5py.File(filename)
  files.append(f)


#this is the master core that distributes the contents of the pickle
def master():
  #variable definitions and opening of h5 files
  time_now = time.time()
  
  evts = data_dict.keys()
  num_evts = len(evts)

  n_batches = int(np.ceil(float(num_evts)/batch_size))
  print('Number of events %i' % num_evts)
  print('Number of batches %i' % n_batches)
  #n_chunks --> n_batch

  key_batches = np.array_split(evts,n_batches)
  
  for ct,batch in enumerate(key_batches):
     # print('sending chunk %i' % ct)
      rankreq = comm.recv(source=MPI.ANY_SOURCE)
      comm.send(batch, dest=rankreq)

  for rankreq in range(size-1):
 #   print 'TOTAL ANALYSIS TIME FOR CLIENT CORES:', time.time()-time_now
    rankreq = comm.recv(source=MPI.ANY_SOURCE)
    comm.send('endrun', dest=rankreq)

#this is what the client cores are responsible for    


def client():
    while True:
      comm.send(rank, dest=0)
      keys = comm.recv(source=0)
  
      #if the endrun signal has been received, print the analysis time and exit
      if str(keys) == 'endrun': 
        break

      for key in keys:        
        data_locs = data_dict[key]
        cspad_image=[]
        cspad_red_image=[]

        for file_num, evt_loc in zip(data_locs[0], data_locs[1]):
          
#          print(rank, file_num, evt_loc)
        
          #Find all the tiles from the current image
          cspad_tile = files[file_num]['cspad_data/image_data'][evt_loc]#[:]
   #       cspad_image = np.r_[cspad_image, cspad_tile]
          
          #and the reduced size image
          cspad_red_tile = files[file_num]['cspad_data/image_data'][evt_loc]#  
    #      cspad_red_image = np.r_[cspad_red_image,cspad_red_tile] 
        cspad_image = None
        cspad_red_image = None
        #event.append([cspad_image,cspad_red_image])
comm.Barrier()
tstart = time.time()
if rank == 0:
  master()
else:
  client()

comm.Barrier()

if rank == 0:
  tend = time.time()
  telapsed = tend - tstart

  # Averaged speed in GB/s
  total_file_size /= 10**9
  average_speed = total_file_size/(telapsed)
  
  print("Batch size: %i" % batch_size)
  print("Time elapsed: %.2f s" % telapsed)
  print("Total file size: %.2f GB" % total_file_size)

  print("Average speed: %.2f GB/s" % average_speed)  

#  with open("log.txt", "w") as text_file:
#    text_file.write("Batch size: %i \n" % batch_size)
#    text_file.write("Time elapsed: %i \n" % telapsed)
#    text_file.write("Average speed: %.2f \n" % average_speed) 

