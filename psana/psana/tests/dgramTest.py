import numpy as np
import dgramCreate as dc

class alg:
    def __init__(self, name,version):
        self.algname = name
        [self.major,self.minor,self.micro] = version

type=5
version=7 

algor = dc.PyAlg(b'foo',0,1,2)
 # n = dc.PyName(b"foo", alg)


# dataarr=np.full((3,3), 1, dtype=np.float)
# dataarr = np.random.random((3,3))

data=[]
for i in range(10):
    data.append([[b"name%i" % i,alg(b"alg%i" % i, [0,1,2])],np.random.random((3,3))])
     # data.append([[b"name%i" % i,alg(b"alg%i" % i, [0,1,2])],dataarr])

dc.blockcreate(data, True)
