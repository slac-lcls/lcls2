import time
from psdaq.seq.seq import *
from psdaq.cas.pvedit import *
from threading import Lock
import argparse

tmo = 5.0  # epics pva timeout

class SeqUser:
    def __init__(self, base):
        prefix = base
        self.ninstr   = Pv(prefix+':INSTRCNT')
        self.desc     = Pv(prefix+':DESCINSTRS')
        self.instr    = Pv(prefix+':INSTRS')
        self.idxseq   = Pv(prefix+':SEQIDX')  # array of indices
        self.seqnames = Pv(prefix+':SEQDESC') # array of names
        self.idxseq0  = Pv(prefix+':SEQ00IDX')
        self.seqname  = Pv(prefix+':SEQ00DESC')
        self.seqbname = Pv(prefix+':SEQ00BDESC')
        self.idxseq0r  = Pv(prefix+':RMVIDX')
        self.seqr     = Pv(prefix+':RMVSEQ')
        self.insert   = Pv(prefix+':INS')
        self.idxrun   = Pv(prefix+':RUNIDX')
        #  SCHEDRESET values are:
        #  0 = no reset
        #  1 = schedule reset once
        #  2 = don't schedule reset, but queue to reset on next linkUp
        #  3 = schedule reset once, and queue to reset on next linkUp
        #  4 = dont schedule reset, but next force reset also queues reset on linkUp
        self.start    = Pv(prefix+':SCHEDRESET')
        self.reset    = Pv(prefix+':FORCERESET')
        self.running  = Pv(prefix+':RUNNING', self.changed)
        self._idx     = 0
        self.lock     = None

        xpmpf = ':'.join(prefix.split(':')[:4])
        self.seqcodes = Pv(xpmpf+':SEQCODES',isStruct=True)
        self.eng      = int(prefix.split(':')[-1])

    def changed(self,err=None):
        q = self.running.__value__
        if q==0 and self.lock!=None:
            self.lock.release()
            self.lock=None

    def stop(self):
        self.idxrun.put(0,wait=tmo)  # a do-nothing sequence
        self.reset .put(1,wait=tmo)
        self.reset .put(0,wait=tmo)

    def clean(self, ridx=None):
        if ridx is None:
            self.idxseq0r.put(-1,wait=tmo)
            self.seqr.put(1,wait=tmo)
        else:
            aidx = self.idxseq.get()
            for idx in aidx:
                if idx==0 or idx==ridx:
                    continue
                print( 'Removing seq %d'%idx)
                self.idxseq0r.put(idx,wait=tmo)
                self.seqr.put(1,wait=tmo)

    def load(self, title, instrset, descset=None):
        self.desc.put(title,wait=tmo)

        encoding = [len(instrset)]
        for instr in instrset:
            encoding = encoding + instr.encoding()

        #print( encoding)

        self.instr.put( tuple(encoding),wait=tmo)

        time.sleep(1.0)

        ninstr = self.ninstr.get()
        if ninstr != len(instrset):
            print( 'Error: ninstr invalid %u (%u)' % (ninstr, len(instrset)))
            return

        print( 'Confirmed ninstr %d'%ninstr)

        self.insert.put(1,wait=tmo)

        #  How to handshake the insert.put -> idxseq0.get (RPC?)
        time.sleep(1.0)

        #  Get the assigned sequence num
        idx = self.idxseq0.get()
        if idx < 2:
            print( 'Error: subsequence index  invalid (%u)' % idx)
            raise RuntimeError("Sequence failed")

        print( 'Sequence '+self.seqname.get()+' found at index %d'%idx)

        #  (Optional for XPM) Write descriptions for each bit in the sequence
        if descset!=None:
            self.seqbname.put(descset,wait=tmo)
            
            seqcodes = self.seqcodes.get()
            desc     = seqcodes.value.Description
            for e in range(4*self.eng,4*self.eng+4):
                desc[e] = ''
            for i,d in enumerate(descset):
                desc[4*self.eng+i] = d

            v = seqcodes.value
            v.Description = desc
            seqcodes.value = v
            self.seqcodes.put(seqcodes,wait=tmo)
                
        self._idx = idx

    def begin(self, wait=False, refresh=False):
        self.start .put(0,wait=tmo) # noop
        self.idxrun.put(self._idx,wait=tmo)
        self.reset .put(0,wait=tmo)
        self.reset .put(1,wait=tmo)
        if wait:
            self.lock= Lock()
            self.lock.acquire()

    def sync(self,refresh=False):
        self.start .put(0,wait=tmo) # noop
        self.idxrun.put(self._idx,wait=tmo)
        self.reset .put(0,wait=tmo)
        self.start .put(2 if not refresh else 4,wait=tmo)

    #  Move from one set to the next without stopping
    def execute(self, title, instrset, descset=None, sync=False, refresh=False):
        self.clean(self.idxrun.get())
        self.load (title,instrset,descset)
        if sync:
            self.sync(refresh)  # schedule the reset
        else:
            self.begin(refresh) # reset now
        #self.idxrun.put(self._idx,wait=tmo)
        #self.start.put(2,wait=tmo)

def main():
    parser = argparse.ArgumentParser(description='sequence pva programming')
    parser.add_argument('--pv', type=str, required=True, help="sequence engine pv; e.g. DAQ:NEH:XPM:0")
    parser.add_argument("--seq", required=True, nargs='+', type=str, help="sequence engine:script pairs; e.g. 0:train.py")
    parser.add_argument("--start", action='store_true', help="start the sequences")
    parser.add_argument("--reset", action='store_true', help="reset the sequences (async)")
    parser.add_argument("--verbose", action='store_true', help="verbose output")
    args = parser.parse_args()

    files = []
    engineMask = 0

    seqcodes_pv = Pv(f'{args.pv}:SEQCODES',isStruct=True)
    seqcodes = seqcodes_pv.get()
    desc = seqcodes.value.Description

    for s in args.seq:
        sengine,fname = s.split(':',1)
        engine = int(sengine)
        print(f'** engine {engine} fname {fname} **')

        config = {'title':'TITLE', 'descset':None, 'instrset':None, 'seqcodes':None, 'repeat':False}
        seq = 'from psdaq.seq.seq import *\n'
        seq += open(fname).read()
        exec(compile(seq, fname, 'exec'), {}, config)
        
        print(f'descset  {config["descset"]}')
        print(f'seqcodes {config["seqcodes"]}')
        if 'refresh' not in config:
            config['refresh']=False
        print(f'refresh  {config["refresh"]}')
        if args.verbose:
            print('instrset:')
            for i in config["instrset"]:
                print(i)

        seq = SeqUser(f'{args.pv}:SEQENG:{engine}')
        seq.execute(config['title'],config['instrset'],config['descset'],sync=not args.reset,refresh=config['refresh'])
        del seq

        engineMask |= (1<<engine)

        for e in range(4*engine,4*engine+4):
            desc[e] = ''
        for e,d in config['seqcodes'].items():
            desc[4*engine+e] = d
        print(f'desc {desc}')

    if args.start:
        v = seqcodes.value
        v.Description = desc
        seqcodes.value = v
        seqcodes_pv.put(seqcodes,wait=tmo)

        pvSeqReset = Pv(f'{args.pv}:SeqReset')
        pvSeqReset.put(engineMask,wait=tmo)

        

if __name__ == '__main__':
    main()

