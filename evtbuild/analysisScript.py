"
Here is the analysis script in the offline event builder trilogy
This code will distribute the matched indices from the pickle file
to client cores that will retrieve the event data corresponding to
the indices.
"
from psana import *
from mpi4py import MPI
import h5py, time, sys, pickle
import numpy as np

#logistical MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
assert size>1
global nbatch
nbatch = int(sys.argv[1])
comm.Barrier()

#INSERT NUMBER OF FILES HERE (Currently only supports 10)
number_of_h5_files = 10

#global variables to be called in the core definitions
global runs
runs = []
runs = np.arange(1, number_of_h5_files+1)
global files
files = []
global event
event = []

#open every h5 file
for r in runs:
  filename = 'file'+str(r)+'.h5'
  f = h5py.File(filename)
  files.append(f)

#this is the master core that distributes the contents of the pickle
def master():
  #variable definitions and opening of h5 files
  master_index = 0
  global indices
  global nbatch
  file_Name = "eventpickle"
  loop = 0
  fileObject = open(file_Name, 'r')
  b = pickle.load(fileObject)
  time_now = time.time()
  blocks = range(0, len(b), nbatch)
  loop = 0
  i = 0
  #looping thorugh the pickle in blocks and 
  while loop <= blocks[-1]:
    if loop+nbatch < len(b):
      sending = b[loop:loop+nbatch]
    else:
      sending = b[loop:]
    #determine which cores are free and send them batches of indices for analysis
    rankreq = comm.recv(source=MPI.ANY_SOURCE)
    comm.send(sending, dest=rankreq)
    loop = loop + nbatch
  #send 'endrun' when entire pickle has been sent and terminate the script
  for rankreq in range(size-1):
    print 'TOTAL ANALYSIS TIME FOR CLIENT CORES:', time.time()-time_now
    rankreq = comm.recv(source=MPI.ANY_SOURCE)
    comm.send('endrun', dest=rankreq)

#this is what the client cores are responsible for    
def client():
    while True:
      comm.send(rank, dest=0)
      results = comm.recv(source=0)
      #if the endrun signal has been received, print the analysis time and exit
      if str(results) == 'endrun': 
        break
      a = 0
      #loop through the files and congregate event data based on the indices received
      batchedevents = len(results)
      while a < batchedevents:
        c = 0
        while c < len(results[a]):
          #append the matched event data to an array called 'event'
          matchno = results[a][c]
          fileno = results[a][c][0]
          evtindex = results[a][c][1]
          data = files[fileno]['bigdata%s' %(str(fileno+1))][evtindex]
          event.append(data)
          c += 1
        a += 1

if rank == 0:
  master()
else:
  client()

comm.Barrier()
