# This file is to test collective io in h5py
 

import pickle, glob, os, random, subprocess
from mpi4py import MPI
import numpy as np
import h5py
import time
import sys
from picklecreate import load_config

import socket
sock = int(socket.gethostname()[-1])
sock = sock%2

nevents = str(sys.argv[1])

config_dict = load_config()
path = config_dict['path'] + '/' + nevents

file_Name = path+"/eventpickle"
data_dict = pickle.load(open(file_Name, 'r'))


comm =MPI.COMM_WORLD
nproc = comm.Get_size()
rank = comm.Get_rank()

#open the dataset, get the handle, shape

def load_files():
  filenames = glob.glob(path +'/*.h5')
  files = []
  for file in filenames:
    f = h5py.File(file,mode='r', driver='mpio', comm=MPI.COMM_WORLD)
    f.atomic = False
    files.append(f)


def master():
  pass

def client():
  pass
#   print('Client %i started' % rank)
# #  num = rank%2
  
#   print(filename)
  
#   print('File open')
#   f.atomic = False
#   f.close()


comm.Barrier()
timestart=MPI.Wtime()
files = load_files()

if rank ==0:
  master()
else:
  client()

comm.Barrier()
timeend=MPI.Wtime()


