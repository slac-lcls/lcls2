import sys
import os
sys.path.append('../build/xtcdata')
from dgram import Dgram

class DataSource:
    def __init__(self,fname,verbose=0):
        self.fd = os.open(fname, os.O_RDONLY|os.O_LARGEFILE)
        self.verbose=verbose
        self.config = self.__next__()
    def __iter__(self):
        return self
    def __next__(self):
        return Dgram(self.fd, self.verbose)

if __name__ == '__main__':
    ds = DataSource('data.xtc')
    for evt in ds:
        print(evt.float0,evt.float1)
