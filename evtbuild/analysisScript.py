
'''
Here is the analysis script in the offline event builder trilogy
This code will distribute the matched indices from the pickle file
to client cores that will retrieve the event data corresponding to
the indices.
'''
#from psana import *

from mpi4py import MPI
import cPickle as pickle
import json
import h5py, time, sys, os, glob
import numpy as np
from numba import jit
from picklecreate import load_config

#logistical MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
assert size>1, 'At least 2 MPI ranks required'

#print(rank, size)

global batch_size
global files, total_file_size
batch_size = int(sys.argv[1])
try:
  scr_dir = str(sys.argv[2])
except IndexError:
#  print('No options')
  sys.exit()

load_all = 0
try:
  load_all = sys.argv[3]
except IndexError:
  pass


comm.Barrier()

#load the pickle
global data_dict
config_dict = load_config()

path = config_dict['path'] + '/' + scr_dir +'/' +'nstripes_18'

file_Name = path+"/eventpickle"
if load_all:
  file_Name += '_all'

#Loading a json file is 30 time faster than a pickle 
#data_dict = pickle.load(open(file_Name, 'r'))
data_dict = {int(k):v for k,v in json.load(open(file_Name, 'r')).iteritems()}

#open every h5 file
def load_files():
  files=[]
  total_file_size = 0

 # print('loading files')
  file_list = glob.glob(path+'/*.h5')
  file_list = np.sort(file_list)
  for filename in file_list:
    file_size = float(os.stat(filename)[6])
    total_file_size += file_size
    f = h5py.File(filename, mode = 'r')#, driver = 'mpio', comm=comm)
    files.append(f)
  return files, total_file_size




#this is the master core that distributes the contents of the pickle
def master():
  #variable definitions and opening of h5 files
  
  evts = data_dict.keys()
    
#  evts = np.sort(evts)
  num_evts = len(evts)
 # files, _ = load_files()

  n_batches = int(np.ceil(float(num_evts)/batch_size))
  print('Number of events %i' % num_evts)
  print('Number of batches %i' % n_batches)
 



  key_batches = np.array_split(evts,n_batches)
#  print(key_batches)
  for ct,batch in enumerate(key_batches):
  #  print(ct)
    rankreq = comm.recv(source=MPI.ANY_SOURCE)
    comm.send(batch, dest=rankreq[0])
    

 # print('Done')
  tot_size = 0
  for rank in range(1,size):
#    print('Ending %i' % rank)
    comm.send('endrun', dest=rank)
    cl_size = comm.recv(source=rank)
  #  print('cl_size is %i' % cl_size[3])
    tot_size += cl_size[3]

#  print('Total size %i' % tot_size)
  return tot_size
#this is what the client cores are responsible for    

def client():
  files,_ = load_files()
  tile_ct = 0
  while True:
     # print('loading files')
      
      comm.send([rank,0,0,tile_ct], dest=0)
      keys = comm.recv(source=0)
      
      #if the endrun signal has been received, print the analysis time and exit
      if str(keys) == 'endrun': 
        print('Client %i done' % rank)
        break
      
      for key in keys:
        data_locs = data_dict[key]
                
        for file_num, evt_loc in zip(data_locs[0], data_locs[1]):
          cspad_tile = files[file_num]['cspad_data/image_data'][evt_loc]#[:]
          tile_ct += cspad_tile.nbytes

  [x.close() for x in files]



comm.Barrier()
#files,_ = load_files()

timestart=MPI.Wtime()
if rank == 0:
  print('start')  
  tstart = time.time()
  ts = master()
else:
  client()

comm.Barrier()
timeend=MPI.Wtime()

if rank == 0:

  tend = time.time()
  telapsed = tend - tstart
  telapsed = timeend-timestart
  # Averaged speed in GB/s
 # _, total_file_size = load_files()
  #close_files(files)
  ts /= 10**9
  average_speed = ts/(telapsed)
  print("Batch size: %i" % batch_size)
  print("Time elapsed: %.2f s" % telapsed)
  print("Total file size: %.2f GB" % ts)

  print("Average speed: %.2f GB/s" % average_speed)  
  

