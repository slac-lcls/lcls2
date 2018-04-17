import numpy as np
import dgramCreate as dc
import os

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
# ninfo3 = dc.nameinfo('xpphsd3', 'cspad', 'detnum1236', 2)

pydgram = dc.writeDgram(FILE_NAME)
pydgram.addDet(ninfo, alg, generate_event(NUM_ELEM, 0))
pydgram.writeToFile()
# pydgram.addDet(ninfo2, alg2, generate_event(NUM_ELEM, 0))
# for _ in range(10):
#     pydgram.addDet(ninfo, alg, generate_event(NUM_ELEM, 1))
#     pydgram.addDet(ninfo2, alg2, generate_event(NUM_ELEM, 1))
#     pydgram.writeToFile()
