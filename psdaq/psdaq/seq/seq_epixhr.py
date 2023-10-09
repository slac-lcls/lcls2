#from evtsel import *
import sys
import math
import argparse
#from sequser import *
from psdaq.seq.seq import *
from psdaq.seq.seqprogram import *

def main():

    parser = argparse.ArgumentParser(description='Sequence for EpixHR DAQ/Run triggering')
    parser.add_argument('--rate', help="Run trigger rate (Hz)", type=float, default=5.0e3);
    parser.add_argument('--tgt' , help="DAQ trigger time (sec)", type=float, default=0.8e-3)
    parser.add_argument('--pv' , help="XPM pv base", default='DAQ:NEH:XPM:7:SEQENG:2')
    parser.add_argument('--f360', help="Resync at 360Hz", action='store_true')
    parser.add_argument('--test', help="Calculate only", action='store_true')
    args = parser.parse_args()

    #  Generate a sequence that repeats at each 120 Hz AC marker
    #  One pulse is targeted for the DAQ trigger time.  Pulses before and after
    #  are added as they fit for "rate" spacing between the AC markers

    fbucket = 13.e6/14
    ac_period = 3*0x5088c/119.e6  # a minimum period (about 1/120.25 Hz)
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
        avgrate *= 3.

    print(f' spacing [{spacing}]  rate [{rate} Hz]')
    print(f' npretrig [{npretrig}]  nposttrig [{nafter}]  avg rate [~{avgrate} Hz]')
    print(f' first [{startb} bkt  {startb/fbucket*1.e6} usec]')
    print(f' beam [{startb+npretrig*spacing} bkt]  {(startb+npretrig*spacing)/fbucket*1.e6} usec]')
    print(f' last [{startb+(npretrig+nafter)*spacing} bkt  {(startb+(npretrig+nafter)*spacing)/fbucket*1.e6} usec]')

    if args.test:
        sys.exit()

    instrset = []
    # 60Hz x timeslots 1,4
    instrset.append(ACRateSync((1<<0)|(1<<3),0,1))
    if startb:
        instrset.append(FixedRateSync(marker=6,occ=startb-1))

    if npretrig:
        line = len(instrset)
        instrset.append(ControlRequest([0]))
        instrset.append(FixedRateSync(marker=6,occ=spacing))
        if npretrig>1:
            instrset.append(Branch.conditional(line,counter=0,value=npretrig-1))
                     
    instrset.append(ControlRequest([0,1]))

    line = len(instrset)
    instrset.append(FixedRateSync(marker=6,occ=spacing))
    instrset.append(ControlRequest([0]))
    instrset.append(Branch.conditional(line,counter=0,value=nafter-1))

    instrset.append(Branch.unconditional(line=0))

    descset = [f'{int(fbucket/spacing)} Hz run trig','daq trig']

    i=0
    for instr in instrset:
        print( 'Put instruction(%d): '%i), 
        print( instr.print_())
        i += 1


    title = 'ePixHR'

    seq = SeqUser(args.pv)

    seq.execute(title,instrset,descset)

if __name__ == '__main__':
    main()
