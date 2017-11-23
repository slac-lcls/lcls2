from mpi4py import MPI
from swmr_write2 import write_files
from swmr_read3 import read_files
import glob,os
from load_config import load_config
import numpy as np
import time

cfg = load_config('sconfig')

#logistical MPI setup
world_comm = MPI.COMM_WORLD
world_rank = world_comm.Get_rank()
world_size = world_comm.Get_size()
#assert size>1, 'At least 2 MPI ranks required'

def clear_files(path):
    files = glob.glob(path)
    for fil in files:
        os.remove(fil)

    #check if the files have actually been deleted
    while True:
        files = glob.glob(path)
        if len(files) == 0:
            break
        else:
            time.sleep(0.2)

cores_per_group = world_size / 3
node_count =  3
array_inds = np.array_split(np.arange(node_count),3)

if world_rank % node_count in array_inds[0]:
    color = 0 # reader
elif world_rank % node_count in array_inds[1]:
    color = 1 # filter
elif world_rank % node_count in array_inds[2]:
    color = 2 # copier

key = world_rank % cores_per_group

try:
    comm = world_comm.Split(color, key)
    rank = world_rank % cores_per_group
    size = world_size/3
except:
    print(world_rank)

world_comm.Barrier()

if world_rank == 0:
    clear_files(cfg['path']+'/*.h5')

world_comm.Barrier()
    
if color == 0:
 #   pass
    write_files(comm) # write
elif color == 1:
    #pass
    #    #comm_test(color,comm,rank,size)
    read_files(comm, 1) # filter
elif color == 2:
  #  pass
    read_files(comm, 0) # copy

