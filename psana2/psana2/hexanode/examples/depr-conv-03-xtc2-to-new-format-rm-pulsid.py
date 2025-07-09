import numpy as np
import sys

class OldDgram:
    def __init__(self,f):
        headerwords = 8 # 32-bit words.  5 for Dgram with pulseId, 3 for Xtc 
        self._header = np.fromfile(f,dtype=np.uint32,count=headerwords)
        self._xtcsize = 12 # bytes.
        self._payload = np.fromfile(f,dtype=np.uint8,count=self.extent()-self._xtcsize)
    def pulseidlow(self): return self._header[0]
    def pulseidhigh(self): return self._header[1]
    def timelow(self): return self._header[2]
    def timehigh(self): return self._header[3]
    def env(self): return self._header[4]
    def transitionId(self): return (self.pulseidhigh()>>24)&0xf
    def control(self): return (self.pulseidhigh()>>24)&0xff
    def extent(self): return self._header[7]
    def next(self): return self.extent()+self._xtcsize
    def data(self): return self._header
    def writeNoPulseId(self,outfile):
        # put the control byte in the top part of env
        self._header[4] = (self._header[4]&0xffffff)|(self.control()<<24)
        # remove the 64-bit pulseid/control word
        self._header[2:].tofile(outfile)
        self._payload.tofile(outfile)

assert len(sys.argv)==3
infname = sys.argv[1]
outfname = sys.argv[2]

infile = open(infname,'r')
outfile = open(outfname,'w')
try:
    while(1):
        dg = OldDgram(infile)
        #print(dg.transitionId(),dg.extent())
        dg.writeNoPulseId(outfile)
except: # happens on end of file
    print('done')
    infile.close()
    outfile.close()
