import numpy as np
import dgramCreate as dc

class alg:
    def __init__(self, name,version):
        self.algname = name
        [self.major,self.minor,self.micro] = version




dataarr=np.full((2,2), 128.0, dtype=np.double)
num_elem = 10

data=[]
for i in range(num_elem):  
    data.append([[b"name%i" % i,alg(b"alg%i" % i, [0,1,2])],dataarr])

verbose = True
dc.blockcreate(data)
 
with open('data.xtc', 'rb') as f:
	data =f.read()

data_m = np.resize(np.frombuffer(data[-160:], dtype = dataarr.dtype), (num_elem, *dataarr.shape))

if verbose:
    print("Data array is: \n ", data_m)
