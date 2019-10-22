import os, shutil
import subprocess
import sys, os
from psana import DataSource
import numpy as np
import vals

def launch_client(pid):
    dg_count = 0
    ds = DataSource(shmem='shmem_test_'+pid)
    run = next(ds.runs())
    cspad = run.Detector('xppcspad')
    hsd = run.Detector('xpphsd')
    for evt in run.events():
        if not evt:
            break
        if not evt._dgrams:
            break
        if not len(evt._dgrams):
            break
        # check for L1 accept transition ID 12
        assert(hsd.raw.calib(evt).shape==(5,))
        assert(hsd.fex.calib(evt).shape==(6,))
        padarray = vals.padarray
        assert(np.array_equal(cspad.raw.calib(evt),np.stack((padarray,padarray))))
        assert(np.array_equal(cspad.raw.image(evt),np.vstack((padarray,padarray))))
        if evt._dgrams[0].seq.service() == 12:
            # immediately release datagram to server
            # if this "del" isn't done then the client receives fewer
            # datagrams, perhaps because the shmem buffer doesn't
            # guarantee delivery of all dgrams, but I'm not certain - cpo
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
