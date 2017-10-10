
'''
Here is the analysis script in the offline event builder trilogy
This code will distribute the matched indices from the pickle file
to client cores that will retrieve the event data corresponding to
the indices.
'''
#from psana import *

from mpi4py import MPI
#import cPickle as pickle
import json
import h5py, time, sys, os, glob
import numpy as np
#from numba import jit
from picklecreate import load_config, create_json
import argparse

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-a','--all events', action='store_true', help = "Read all available events", dest="load_all", default = False)
parser.add_argument('-s','--subset', action='store_true', help = "Read a subset of files", dest="subset", default = False)
parser.add_argument('-d','--directory', action='store', help = "", dest = "dir_name")
parser.add_argument('-b','--batch', action='store', help = "Batch size", dest = "batch_size", type = int, default = 100)
parser.add_argument('-bd','--bb_directory_name', action='store', help = "Directory name for BB", dest = "bb_dir_name", type = str, default='')

args = parser.parse_args()


#logistical MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
assert size>1, 'At least 2 MPI ranks required'


#load the pickle
global data_dict
config_dict = load_config()

path = config_dict['path'] + '/' + str(args.dir_name) +'/' +'nstripes_11'
if args.bb_dir_name:
  path = args.bb_dir_name
  path = path[:-1]

comm.Barrier()


file_Name = path+"/eventpickle"
#print(file_Name)
if args.load_all:
  file_Name += '_all'

#Loading a json file is 30 time faster than a pickle 
#data_dict = pickle.load(open(file_Name, 'r'))
data_dict = {int(k):v for k,v in json.load(open(file_Name, 'r')).iteritems()}

#open every h5 file

def load_files(path):
  files=[]
  total_file_size = 0
  pth = path
 # print('loading files')
  if args.bb_dir_name:
    new_path = args.bb_dir_name
    new_path = new_path[:-1]
    pth = new_path
  

  file_list = glob.glob(pth+'/*.h5')
  file_list = np.sort(file_list)
  #print(file_list)
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
 # files, _ = load_files(path)

  n_batches = int(np.ceil(float(num_evts)/args.batch_size))
  print('Number of events %i' % num_evts)
  print('Number of batches %i' % n_batches)
 

  key_batches = np.array_split(evts,n_batches)
#  print(key_batches)
  for ct,batch in enumerate(key_batches):
  #  print(ct)
    rankreq = comm.recv(source=MPI.ANY_SOURCE)
    comm.send(batch, dest=rankreq[0])

  #Signal the clients to end
  tot_size = 0
  rnk = size
  while rnk >1:
    rank = comm.recv(source = MPI.ANY_SOURCE)
    #print('Ending rank %i' % rank[0])
    comm.send('endrun', dest = rank[0])
    tot_size += rank[3]
    rnk-=1
  return tot_size


#this is what the client cores are responsible for    

def client():
  files,_ = load_files(path)
  tile_ct = 0
  while True:
     # print('loading files')
      
      comm.send([rank,0,0,tile_ct], dest=0)
      keys = comm.recv(source=0)
#      tile_ct = 0
      #if the endrun signal has been received, print the analysis time and exit
      if str(keys) == 'endrun': 
       # print('Client %i done' % rank)
        break
      
      for key in keys:
        data_locs = data_dict[key]
                
        for file_num, evt_loc in zip(data_locs[0], data_locs[1]):
          if args.subset and (file_num not in config_dict['subset']):
            continue
        
          cspad_tile = files[file_num]['cspad_data/image_data'][evt_loc]#[:]
          tile_ct += cspad_tile.nbytes
     

  [x.close() for x in files]



comm.Barrier()
#files,_ = load_files()

timestart=MPI.Wtime()
if rank == 0:
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
  ts /= 10**9
  average_speed = ts/(telapsed)
  print('\n'+'='*40)
  if args.subset:
    print("Subset [%s]" % ''.join(map(str, config_dict['subset'])))
  if args.load_all:
    print('Sequential access')
  else:
    print('Random access')

  print("Batch size: %i" % args.batch_size)
  print("Time elapsed: %.2f s" % telapsed)
  print("Total file size: %.2f GB" % ts)

  print("Average speed: %.2f GB/s" % average_speed)  
  print('+'*40+'\n')

