import sys
import os
sys.path.append('../build/xtcdata')
from dgram import Dgram

class DataSource:
    def __init__(self,fname):
        self.fd = os.open(fname, os.O_RDONLY|os.O_LARGEFILE)
        self.config = Dgram(self.fd)
    def __iter__(self):
        return self
    def __next__(self):
        return Dgram(self.fd, self.config)

if __name__ == '__main__':
    ds = DataSource('data.xtc')
    for evt in ds:
        print(evt.float_pgp,evt.int_pgp,evt.array0_pgp)
