import time
from psdaq.seq.seq import *
from psdaq.cas.pvedit import *
from threading import Lock

class SeqUser:
    def __init__(self, base):
        prefix = base
        self.ninstr   = Pv(prefix+':INSTRCNT')
        self.desc     = Pv(prefix+':DESCINSTRS')
        self.instr    = Pv(prefix+':INSTRS')
        self.idxseq   = Pv(prefix+':SEQ00IDX')
        self.seqname  = Pv(prefix+':SEQ00DESC')
        self.seqbname = Pv(prefix+':SEQ00BDESC')
        self.idxseqr  = Pv(prefix+':RMVIDX')
        self.seqr     = Pv(prefix+':RMVSEQ')
        self.insert   = Pv(prefix+':INS')
        self.idxrun   = Pv(prefix+':RUNIDX')
        self.start    = Pv(prefix+':SCHEDRESET')
        self.reset    = Pv(prefix+':FORCERESET')
        self.running  = Pv(prefix+':RUNNING', self.changed)
        self._idx     = 0
        self.lock     = None

    def changed(self,err=None):
        q = self.running.__value__
        if q==0 and self.lock!=None:
            self.lock.release()
            self.lock=None

    def stop(self):
        self.idxrun.put(0)  # a do-nothing sequence
        self.reset .put(1)
        self.reset .put(0)

    def clean(self):
        # Remove existing sub sequences
        ridx = -1
        print( 'Remove %d'%ridx)
        if ridx < 0:
            idx = self.idxseq.get()
            while (idx>0):
                print( 'Removing seq %d'%idx)
                self.idxseqr.put(idx)
                self.seqr.put(1)
                self.seqr.put(0)
                time.sleep(1.0)
                idx = self.idxseq.get()
        elif ridx > 1:
            print( 'Removing seq %d'%ridx)
            self.idxseqr.put(ridx)
            self.seqr.put(1)
            self.seqr.put(0)

    def load(self, title, instrset, descset=None):
        self.desc.put(title)

        encoding = [len(instrset)]
        for instr in instrset:
            encoding = encoding + instr.encoding()

        print( encoding)

        self.instr.put( tuple(encoding) )

        time.sleep(1.0)

        ninstr = self.ninstr.get()
        if ninstr != len(instrset):
            print( 'Error: ninstr invalid %u (%u)' % (ninstr, len(instrset)))
            return

        print( 'Confirmed ninstr %d'%ninstr)

        self.insert.put(1)
        self.insert.put(0)

        #  How to handshake the insert.put -> idxseq.get (RPC?)
        time.sleep(1.0)

        #  Get the assigned sequence num
        idx = self.idxseq.get()
        if idx < 2:
            print( 'Error: subsequence index  invalid (%u)' % idx)
            raise RuntimeError("Sequence failed")

        print( 'Sequence '+self.seqname.get()+' found at index %d'%idx)

        #  (Optional for XPM) Write descriptions for each bit in the sequence
        if descset!=None:
            self.seqbname.put(descset)

        self._idx = idx

    def begin(self, wait=False):
        self.idxrun.put(self._idx)
        self.start .put(0)
        self.reset .put(1)
        self.reset .put(0)
        if wait:
            self.lock= Lock()
            self.lock.acquire()

    def execute(self, title, instrset, descset=None):
        self.insert.put(0)
        self.stop ()
        self.clean()
        self.load (title,instrset,descset)
        self.begin()

def main():
    parser = argparse.ArgumentParser(description='sequence pva programming')
    parser.add_argument("seq", help="sequence script")
    parser.add_argument("pv" , help="sequence engine pv; e.g. XPM:0:SEQ_ENG:0")
    args = parser.parse_args()
    
    config = {'title':'TITLE', 'descset':None, 'instrset':None}

    exec(compile(open(args.seq).read(), args.seq, 'exec'), {}, config)

    seq = SeqUser(args.pv)
    seq.execute(config['title'],config['instrset'],config['descset'])

if __name__ == 'main':
    main()
