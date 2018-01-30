import sys
sys.path.append('../../build/psana')
from dgram import Dgram
sys.path.append('../')
from DataSource import DataSource

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
assert size>1, 'Parallel read requires at least 2 ranks.'

###
### ClientPkg stores data being passed to clients.
###
class ClientPkg():
  def __init__(self, filename, offset):
    self.filename = filename
    self.offset = offset

def master():
  ds = DataSource('smd.xtc');
  for i, evt in enumerate(ds):
    rankreq = comm.recv(source=MPI.ANY_SOURCE)
    comm.send(ClientPkg('data.xtc', evt.hsd1.fex.intOffset), dest=rankreq)
  for rankreq in range(size-1):
    rankreq = comm.recv(source=MPI.ANY_SOURCE)
    comm.send('endrun', dest=rankreq)

def client():
  f2dsLookup = {}
  while True:
    comm.send(rank, dest=0)
    myPkg = comm.recv(source=0)
    if myPkg == 'endrun': break

    ### Find the right ds in f2dsLookup.
    ### If not found, open a ds and add this filename 
    ### with the ds to the lookup table.
    if myPkg.filename not in f2dsLookup:
      ds = DataSource(myPkg.filename)
      f2dsLookup[myPkg.filename] = ds
    else:
      ds = f2dsLookup[myPkg.filename]
    e = ds.__jump__(myPkg.offset)
    print("Rank %d reads %s with offset %d"%(rank, myPkg.filename, myPkg.offset))
    show(e)
      
def show(evt):
  for var_name in sorted(vars(evt)):
    e = getattr(evt, var_name)
    print("  %s: %s" %(var_name, e))
    for key in sorted(e.__dict__.keys()):
      print("    %s: %s" % (key, e.__dict__[key]))
    print("")

if __name__ == "__main__":
  if rank == 0:
    master()
  else:
    client()

