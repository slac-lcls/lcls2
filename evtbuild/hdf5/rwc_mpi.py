
from mpi4py import MPI
from write_hdf5 import write_files
from read_hdf5 import read_files
import glob,os
from load_config import load_config
import numpy as np
import time

#print('Start rwc_mpi')
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

task_count = 2
cores_per_group = world_size / task_count
array_inds = np.array_split(np.arange(task_count),task_count)

if world_rank % task_count in array_inds[0]:
    color = 0 # reader
    if world_rank == 0:
        color = 4
elif world_rank % task_count in array_inds[1]:
    color = 1 # copier
# Ask for forgiveness, not permission
try:
    if world_rank % task_count in array_inds[2]:
        color = 2 # filter
except IndexError:
    pass

key = world_rank % cores_per_group

try:
    comm = world_comm.Split(color, key)
  #  rank = world_rank % cores_per_group
   # size = world_size/3
except:
    print(world_rank)

world_comm.Barrier()

if world_rank == 0:
    clear_files(cfg['path']+'/*.h5')

world_comm.Barrier()

#print('Passing to read/write')
if color == 0:
    write_files(comm) # write

if int(cfg['sequential']):
    world_comm.Barrier()
    time.sleep(5)


if color == 1:
    read_files(comm, 0) # copy

if color == 2:
    read_files(comm, 0) # filter

