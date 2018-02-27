from mpi4py import MPI
from read_xtc_lite import do_read
from write_xtc_lite import do_write
import glob,os
from load_config import load_config
import numpy as np
import time


cfg = load_config('sconfig')

world_comm = MPI.COMM_WORLD
world_rank = world_comm.Get_rank()
world_size = world_comm.Get_size()

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


num_tasks = 2
cores_per_group = world_size / num_tasks
node_count =  num_tasks
array_inds = np.array_split(np.arange(node_count),num_tasks)



key = world_rank % cores_per_group


if world_rank % node_count in array_inds[0]:
    color = 0 # reader
elif world_rank % node_count in array_inds[1]:
    color = 1 # filter
#elif world_rank % node_count in array_inds[2]:
#    color =  2

try:
    comm = world_comm.Split(color, key)
except Exception as e:
    print(e)
    print('exception', world_rank)



if world_rank == 0:
#    print('Removing files')
    clear_files(cfg['path']+'/*.xtc')

world_comm.Barrier()



if color == 0:
 #   pass
    do_write(comm) # writ e
elif color == 1:
   # pass
    #    #comm_test(color,comm,rank,size)
    do_read(comm,0) # copy
elif color == 2:
    do_read(comm, 1) # filter
