#from evtsel import *
import sys
import math
import argparse
#from sequser import *
from psdaq.configdb.tsdef import *
from psdaq.seq.seq import *
from psdaq.seq.seqprogram import *

def auto_int(val):
    return int(val,0)

def main():

    parser = argparse.ArgumentParser(description='Sequence for EpixHR DAQ/Run triggering')
    parser.add_argument('--rate', help="Run trigger rate (Hz)", type=float, default=4.9e3)
    parser.add_argument('--tgt' , help="DAQ trigger time (sec)", type=float, default=0.834e-3)
    parser.add_argument('--daq', help='DAQ trigger at RUN trigger rate/N', type=int, default=None)
    parser.add_argument('--full', help='DAQ trigger at RUN trigger rate', action='store_true')
    parser.add_argument('--f360', help='DAQ trigger at 360 Hz', action='store_true')
    parser.add_argument('--minAC', help='Minimum 120H interval in 119MHz clocks', default=0x5088c, type=auto_int )
    parser.add_argument('--pv' , help="XPM pv base", default='DAQ:NEH:XPM:7:SEQENG:2')
    parser.add_argument('--test', help="Calculate only", action='store_true')
    parser.add_argument('--verbose', help="Verbose", action='store_true')
    args = parser.parse_args()

    nsel = 0
    if args.full:
        nsel += 1
    if args.daq:
        nsel += 1
    if args.f360:
        nsel += 1

    if nsel > 1:
        raise RunTimeError('Only one (or none) of --full, --daq, --f360 can be used')

    if args.full:
        args.daq = 1

    #  Generate a sequence that repeats at each 120 Hz AC marker
    #  One pulse is targeted for the DAQ trigger time.  Pulses before and after
    #  are added as they fit for "rate" spacing between the AC markers

    fbucket = 13.e6/14
    ac_period = 3*args.minAC/119.e6  # a minimum period (about 1/120.25 Hz)
    if args.f360:
        ac_period /= 3
    ac_periodb = int(ac_period*fbucket)
    spacing = int(math.ceil(fbucket/args.rate))
    rate    = fbucket/spacing
    targetb = int(args.tgt*fbucket+0.5)
    npretrig = int(targetb/spacing)
    startb  = targetb - npretrig*spacing
    nafter  = int((ac_periodb - startb)/spacing) - npretrig
    avgrate = (npretrig+nafter+1)*120.
    if args.f360:
        avgrate *= 3

    if args.verbose:
        print(f' spacing [{spacing}]  rate [{rate} Hz]')
        print(f' npretrig [{npretrig}]  nposttrig [{nafter}]  avg rate [~{avgrate} Hz]')
        print(f' first [{startb} bkt  {startb/fbucket*1.e6} usec]')
        print(f' beam [{startb+npretrig*spacing} bkt]  {(startb+npretrig*spacing)/fbucket*1.e6} usec]')
        print(f' last [{startb+(npretrig+nafter)*spacing} bkt  {(startb+(npretrig+nafter)*spacing)/fbucket*1.e6} usec]')

    RUN = 0
    DAQ = 1
    PARENT = 2
    RUN_rate = int(rate)
    DAQ_rate = int(rate/args.daq) if args.daq else 360 if args.f360 else 120
    PARENT_rate = DAQ_rate
    if startb or (npretrig and not args.daq):
        PARENT_rate += 360 if args.f360 else 120 
    print(f' eventcode 0: {RUN_rate} Hz run trigger')
    print(f' eventcode 1: {DAQ_rate} Hz daq trigger')
    print(f' eventcode 2: {PARENT_rate} Hz parent group trigger')

    instrset = []
    # 60Hz x timeslots 1,4
    tsmask = 0x3f if args.f360 else 0x9
#    instrset.append(ACRateSync(tsmask,0,1))  # hardcoded to (wrong) AC rate marker until xtpg fixed
    instrset.append(ACRateSync(tsmask,5,1))

    #  Parent trigger comes first
    if startb:
        instrset.append(ControlRequest([PARENT]))
        instrset.append(FixedRateSync(marker='910kH',occ=startb-1))

    if npretrig:
        if args.daq:
            for i in range(npretrig,1,-1):
                instrset.append(ControlRequest([RUN,DAQ,PARENT] if (i%args.daq)==0 else [RUN]))
                instrset.append(FixedRateSync(marker='910kH',occ=spacing))

        elif startb:
            line = len(instrset)
            instrset.append(ControlRequest([RUN]))
            instrset.append(FixedRateSync(marker='910kH',occ=spacing))
            if npretrig>1:
                instrset.append(Branch.conditional(line,counter=0,value=npretrig-1))

        else:
            instrset.append(ControlRequest([RUN,PARENT]))
            instrset.append(FixedRateSync(marker='910kH',occ=spacing))
            if npretrig>1:
                line = len(instrset)
                instrset.append(ControlRequest([RUN]))
                instrset.append(FixedRateSync(marker='910kH',occ=spacing))
                if npretrig>2:
                    instrset.append(Branch.conditional(line,counter=0,value=npretrig-2))

    instrset.append(ControlRequest([RUN,DAQ,PARENT]))

    if nafter:
        if not args.daq or args.daq==1:
            line = len(instrset)
            instrset.append(FixedRateSync(marker='910kH',occ=spacing))
            instrset.append(ControlRequest([RUN,DAQ,PARENT] if args.daq else [RUN]))
            instrset.append(Branch.conditional(line=line,counter=0,value=nafter-1))
        else:  # brute force
            for i in range(1,nafter):
                instrset.append(FixedRateSync(marker='910kH',occ=spacing))
                instrset.append(ControlRequest([RUN,DAQ,PARENT] if (i%args.daq)==0 else [RUN]))

    instrset.append(Branch.unconditional(line=0))

    descset = [f'epixhr run trig','epixhr daq trig','daq parent group trig']

    if args.verbose:
        i=0
        for instr in instrset:
            print(f' {i}: {instr.print_()}')
            i += 1


    title = 'ePixHR'

    if not args.test:
        seq = SeqUser(args.pv)
        tmo = 0
        while(tmo<10):
            try:
                seq.execute(title,instrset,descset)
                break
            except TimeoutError:
                tmo += 1
                print(f'--- Timeout #{tmo}')

        print(f'--- Done: {"Success" if tmo<11 else "Fail"}')

if __name__ == '__main__':
    main()
