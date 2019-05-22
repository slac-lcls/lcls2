#from evtsel import *
import sys
import argparse
from sequser import *
#from seq_ca_new import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='simple exp seq setup')
    parser.add_argument('--pv' , help="TPG pv base", default='TPG:SYS2:2:EXP00')
    args = parser.parse_args()

    sync_marker = 6
    
    instrset = []
    #  Insert global sync instruction (1Hz?)
    instrset.append(FixedRateSync(marker=sync_marker,occ=1))

    for i in range(4):
        sh = i*4

        b0 = len(instrset)
        instrset.append(ControlRequest(0xf<<sh))
        instrset.append(FixedRateSync(marker=0,occ=i+1))
        instrset.append(Branch.conditional(line=b0, counter=0, value=1))

        b0 = len(instrset)
        instrset.append(ControlRequest(0xe<<sh))
        instrset.append(FixedRateSync(marker=0,occ=i+1))
        instrset.append(Branch.conditional(line=b0, counter=0, value=1))

        b0 = len(instrset)
        instrset.append(ControlRequest(0xc<<sh))
        instrset.append(FixedRateSync(marker=0,occ=i+1))
        instrset.append(Branch.conditional(line=b0, counter=0, value=3))

        b0 = len(instrset)
        instrset.append(ControlRequest(0x8<<sh))
        instrset.append(FixedRateSync(marker=0,occ=i+1))
        instrset.append(Branch.conditional(line=b0, counter=0, value=7))

    instrset.append(CheckPoint(0))

    b0 = len(instrset)
    instrset.append(Branch.unconditional(line=b0))

    descset = []
    for j in range(16):
        descset.append('%d x %fus'%(2**(1+(j%4)),1.08*(j/4)))

    i=0
    for instr in instrset:
        print('Put instruction(%d): '%i),
        print(instr.print_())
        i += 1

    title = 'BurstTest'

    seq = SeqUser(args.pv)

#    seq.execute(title,instrset,descset)

    seq.stop()
    seq.clean()
    seq.load(title,instrset,descset)
    seq.begin(wait=True)
    
