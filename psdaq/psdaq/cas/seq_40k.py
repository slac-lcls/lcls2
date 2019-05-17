#from evtsel import *
import sys
import argparse
from sequser import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='simple exp seq setup')
    parser.add_argument('--pv' , help="TPG pv base", default='TPG:SYS2:2')
    args = parser.parse_args()

    sync_marker = 6
    
    instrset = []
    #  Insert global sync instruction (1Hz?)
    instrset.append(FixedRateSync(marker=sync_marker,occ=1))

    NP = 22
    #  Setup a 22 pulse sequence to repeat 40000 times each second
    for i in range(NP):
        bits = 0
        for j in range(16):
            if i*(j+1)%NP < j+1:
                bits = bits | (1<<j)
        instrset.append(ControlRequest(bits))
        instrset.append(FixedRateSync(marker=0,occ=1))

    instrset.append(Branch.conditional(line=1, counter=0, value=199))
    instrset.append(Branch.conditional(line=1, counter=1, value=199))
    instrset.append(Branch.unconditional(line=0))

    descset = []
    for j in range(16):
        descset.append('%d kHz'%((j+1)*40))

    i=0
    for instr in instrset:
        print( 'Put instruction(%d): '%i), 
        print( instr.print_())
        i += 1


    title = 'LoopTest'

    seq = SeqUser(args.pv)

    seq.execute(title,instrset,descset)
