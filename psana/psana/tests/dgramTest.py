import numpy as np
import dgramCreate as dc
import os

try:
    os.remove('data.xtc')
except:
    pass

DETNAME = b'xpphsd'
DETTYPE = b'cspad'
DETID = b'detnum1234'
NAMESID = 0
NUM_ELEM = 3
FILE_NAME = b'data.xtc'

def generate_event(num_elem, zero):
    data = [np.full((2,3,3), zero)]*num_elem
    # data = [np.random.random((2,3,3)) for x in range(num_elem)]
    names = [b'arrayRaw', b'det0', b'det1']
    event_dict = dict(zip(names, data))
    return event_dict

alg = dc.alg(b"raw", [2, 3, 42])
ninfo = dc.nameinfo(b'xpphsd', b'cspad', b'detnum1234', 0)
ninfo2 = dc.nameinfo(b'xpphsd2', b'cspad', b'detnum1234', 0)


pydgram = dc.writeDgram2(FILE_NAME)

pydgram.addDet(ninfo, alg, generate_event(NUM_ELEM, 0))
# pydgram.addDet(ninfo2, alg, generate_event(NUM_ELEM, 0))
pydgram.writeToFile()

 
pydgram.addDet(ninfo, alg, generate_event(NUM_ELEM, 1))
# pydgram.addDet(ninfo2, alg, generate_event(NUM_ELEM, 0))
pydgram.writeToFile()

pydgram.addDet(ninfo, alg, generate_event(NUM_ELEM, 2))
# pydgram.addDet(ninfo2, alg, generate_event(NUM_ELEM, 0))
pydgram.writeToFile()

 
