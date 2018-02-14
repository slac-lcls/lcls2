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
    def __init__(self, event):
        self.filename = 'data.xtc'
        self.event = event

def master():
    ds = DataSource('exp=test:run=0');
    for i, evt in enumerate(ds):
        rankreq = comm.recv(source=MPI.ANY_SOURCE)
        comm.send(ClientPkg(evt), dest=rankreq)
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

        ### Load dgrams in the event object
        ### For now, assume that the first dgram holds
        ### the offset to bigdata
        for i, pydgram in enumerate(myPkg.event):
            if i == 0: 
                bigevt = ds.jump(pydgram.hsd1.fex.intOffset)
                print("Rank %d reads %s with offset %d"%(rank, myPkg.filename, pydgram.hsd1.fex.intOffset))
            else:
                show(pydgram)

def show(pydgram):
    for var_name in sorted(vars(pydgram)):
        attr = getattr(pydgram, var_name)
        print(" var_name  %s: attr %s" %(var_name, attr))
        for key in sorted(attr.__dict__.keys()):
            print("    key %s: val %s" % (key, attr.__dict__[key]))
        print("***")

if __name__ == "__main__":
    if rank == 0:
        master()
    else:
        client()

