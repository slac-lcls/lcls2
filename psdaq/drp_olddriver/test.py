from __future__ import print_function
import numpy as np
import time
from subprocess import call, Popen

for i in range(100):
    print('i', i)

    drp = Popen('build/drp')
    #time.sleep(5)
    #pgp_sender = Popen('build/pgp-sender')

    #pgp_sender.wait()
    drp.wait()

    x = np.loadtxt('test.dat', dtype=np.float32)
    unique, counts = np.unique(x, return_counts=True)

    if np.allclose(counts, 1):
        print('All values correct')
        print(x.shape)
    else:
        ind, = np.where(counts != 1)
        print(ind.shape[0], ' Wrong values')
        print('unique', unique[ind])
        print('counts', counts[ind])
        break
