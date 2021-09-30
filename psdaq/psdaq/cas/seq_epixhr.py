#from evtsel import *
import sys
import argparse
from sequser import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='simple exp seq setup')
    parser.add_argument('--pv' , help="XPM pv base", default='XPM:LAB2:XPM:2:SEQ00')
    args = parser.parse_args()

    #  Generate a sequence that pulses 41 times for each 120 Hz AC rate marker
    #  with each pulse separated by ~200 usec

    instrset = []
    # 60Hz x timeslots 1,4
    instrset.append(ACRateSync((1<<0)|(1<<3),0,1))
    instrset.append(ControlRequest(0x7))

    #  Set three bits: one at the initial 120 Hz, 
    #                  one at 4 kHz (3960 Hz) intervals (233 spacing),
    #                  one at 5 kHz (4920 Hz) intervals (186 spacing)

    n_5k = 40
    n_4k = 32
    i_5k = 186
    i_4k = 233
    while (n_5k>0 and n_4k>0):
        if i_5k < i_4k:
            instrset.append(FixedRateSync(marker=0,occ=i_5k))
            instrset.append(ControlRequest(0x4))
            n_5k -= 1
            i_4k -= i_5k
            i_5k  = 186
        elif i_4k < i_5k:
            instrset.append(FixedRateSync(marker=0,occ=i_4k))
            instrset.append(ControlRequest(0x2))
            n_4k -= 1
            i_5k -= i_4k
            i_4k  = 233
        else:
            instrset.append(FixedRateSync(marker=0,occ=i_4k))
            instrset.append(ControlRequest(0x6))
            n_4k -= 1
            n_5k -= 1
            i_5k  = 186
            i_4k  = 233

    instrset.append(Branch.unconditional(line=0))

    descset = []
    for rate in (120,3960,4920):
        descset.append('%d Hz'%rate)

    i=0
    for instr in instrset:
        print( 'Put instruction(%d): '%i), 
        print( instr.print_())
        i += 1


    title = 'ePixHR'

    seq = SeqUser(args.pv)

    seq.execute(title,instrset,descset)
