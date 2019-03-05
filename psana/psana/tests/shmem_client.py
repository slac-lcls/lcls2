import os, shutil
import subprocess
import sys, os
from psana import DataSource

def launch_client(pid):
    dg_count = 0
    ds = DataSource('shmem','shmem_test_'+pid)
    run = next(ds.runs())
    for evt in run.events():
        if not evt:
            break
        if not evt._dgrams:
            break
        if not len(evt._dgrams):
            break
        # check for L1 accept transition ID 12
        if evt._dgrams[0].seq.service() == 12:
            # immediately release datagram to server
            del evt._dgrams[0]
            dg_count += 1
    return dg_count  

#------------------------------

def main() :
    sys.exit(launch_client(sys.argv[1]))

#------------------------------

if __name__ == '__main__':
    main()

#------------------------------
