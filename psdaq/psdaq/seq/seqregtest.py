import time
from psdaq.seq.seq import *
from psdaq.cas.pvedit import *
from threading import Lock
import argparse
import logging

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

    def execute(self, title, instrset, descset=None):
        self.ninstr.get()
        self.desc  .get()
        self.idxseq.get()
        self.seqname.get()
        self.seqbname.get()
        self.idxrun.get()
        self.running.get()
        self.reset.get()
        self.start.get()

def main():
    parser = argparse.ArgumentParser(description='sequence pva programming')
    parser.add_argument('--engine', type=int, default=0, help="sequence engine")
    parser.add_argument("seq", help="sequence script")
    parser.add_argument("pv" , help="sequence engine pv; e.g. NEH:DAQ:XPM:0")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    config = {'title':'TITLE', 'descset':None, 'instrset':None, 'seqcodes':None}

    exec(compile(open(args.seq).read(), args.seq, 'exec'), {}, config)

    print(f'descset  {config["descset"]}')
    print(f'instrset {config["instrset"]}')
    print(f'seqcodes {config["seqcodes"]}')

    seq = SeqUser(f'{args.pv}:SEQENG:{args.engine}')
    seq.execute(config['title'],config['instrset'],config['descset'])

    seqcodes_pv = Pv(f'{args.pv}:SEQCODES',isStruct=True)
    seqcodes = seqcodes_pv.get()

    desc = seqcodes.value.Description
    for e,d in config['seqcodes'].items():
        desc[4*args.engine+e] = d
    print(f'desc {desc}')

    v = seqcodes.value
    v.Description = desc
    seqcodes.value = v

    print(f'seqcodes_pv {seqcodes}')

    seqcodes_pv.put(seqcodes)

if __name__ == '__main__':
    main()

