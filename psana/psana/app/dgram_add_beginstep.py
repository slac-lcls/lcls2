import numpy as np
import sys

class Dgram:
    def __init__(self,f):
        headerwords = 6 # 32-bit words. 3 for Dgram, 3 for Xtc 
        self._header = np.fromfile(f,dtype=np.uint32,count=headerwords)
        self._xtcsize = 12 # bytes
        self._payload = np.fromfile(f,dtype=np.uint8,count=self.extent()-self._xtcsize)
    def timelow(self): return self._header[0]
    def timehigh(self): return self._header[1]
    def env(self): return self._header[2]
    def transitionId(self): return (self.env()>>24)&0xf
    def control(self): return (self.env()>>24)&0xff
    def extent(self): return self._header[5]
    def next(self): return self.extent()+self._xtcsize
    def data(self): return self._header
    def write(self,outfile):
        self._header.tofile(outfile)
        self._payload.tofile(outfile)

assert len(sys.argv)==3
infname = sys.argv[1]
outfname = sys.argv[2]

infile = open(infname,'r')
outfile = open(outfname,'w')
try:
    ndg = 0
    while(1):
        dg = Dgram(infile)
        #print('----',dg.transitionId(),dg.extent())
        dg.write(outfile)
        if dg.transitionId()==4: # beginrun
            old = dg._header[2]
            dg._header[2] = old&0xf0ffffff | (6<<24) # switch to beginstep
            dg.write(outfile)
        ndg += 1
        if ndg%100 == 0: print('Event:',ndg)
except Exception as e: # happens on end of file
    print('done')
    infile.close()
    outfile.close()
