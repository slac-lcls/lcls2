import numpy as np
import dgramCreate as dc
import os, subprocess

try:
    os.remove('data.xtc')
except:
    pass

FILE_NAME = 'data.xtc'
NUM_ELEM = 1

def generate_event(num_elem, zero):
    # data = np.array([0])*num_elem
    data = [np.full((1,1), zero)]*num_elem
    # data = [np.random.random((2,3,3)) for x in range(num_elem)]
    names = ['det%i' % zero for i in range(num_elem)]
    algs = [alg]*num_elem

    event_dict = dict(zip(names, zip(data, algs)))
    return event_dict
 
alg = dc.alg('alg', [0, 0, 0])
alg2 = dc.alg('alg2', [0, 0, 0])

ninfo = dc.nameinfo('xpphsd', 'cspad', 'detnum1234', 0)
ninfo2 = dc.nameinfo('xpphsd', 'cspad', 'detnum1234', 1)


cydgram = dc.CyDgram()



with open(FILE_NAME, 'wb') as f:
    for _ in range(10):
        cydgram.addDet(ninfo, alg, generate_event(NUM_ELEM, 0))
        cydgram.addDet(ninfo2, alg2, generate_event(NUM_ELEM, 0))
        # pydgram.writeToFile()
        xtc_byte = cydgram.get()
        f.write(xtc_byte)

