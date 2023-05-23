import argparse
import json
import os
import sys
import numpy
from itertools import chain
from functools import reduce
import operator
from .globals import *

verbose = False

#
#  Need a control sequence with periodic triggers at different rates and phasing
#  The periods will be the minimum bunch spacing for SXR and HXR delivery
#  The 'args' parameter shall have .period,.start_bucket lists for each 
#  sequence bit
#
def gcd(a,b):
    d = min(a,b)
    y = max(a,b)
    while True:
        r = y%d
        if r == 0:
            return d
        d = r

def lcm(a,b):
    return a*b // gcd(a,b)

def myunion(s0,s1):
    return set(s0) | set(s1)

class PeriodicGenerator(object):
    def __init__(self, period, start, charge=None, repeat=-1, notify=False, marker='\"910kH\"'):
        self.charge = charge
        self.init(period, start, marker, repeat, notify)

    def init(self, period, start, marker='\"910kH\"', repeat=-1, notify=False):
        self.async_start       = 0
        if isinstance(period,list):
            self.period    = period
            self.start     = start
        else:
            self.period    = [period]
            self.start     = [start]
        self.marker = marker
        self.repeat = repeat
        self.notify = notify

#        if not numpy.less(start,period).all():
#            raise ValueError('start must be less than period')

        self.desc = 'Periodic: period[{}] start[{}]'.format(period,start)
        self.instr = ['instrset = []']
        self.ninstr = 0
        self._fill_instr()
        if self.ninstr > 1024:
            raise RuntimeError('Instruction cache overflow [{}]'.format(self.ninstr))
        
    def _wait(self, intv):
        if intv <= 0:
            raise ValueError
        if intv >= 2048:
            self.instr.append('iinstr = len(instrset)')
            #  _Wait for 2048 intervals
            self.instr.append(f'instrset.append( FixedRateSync(marker={self.marker}, occ=2048) )')
            self.ninstr += 1
            if intv >= 4096:
                #  Branch conditionally to previous instruction
                self.instr.append('instrset.append( Branch.conditional(line=iinstr, counter=3, value={}) )'.format(int(intv/2048)-1))
                self.ninstr += 1

        rint = intv%2048
        if rint:
            self.instr.append(f'instrset.append( FixedRateSync(marker={self.marker}, occ={rint} ) )' )
            self.ninstr += 1

    def _fill_instr(self):
        #  Common period (subharmonic)
        period = numpy.lcm.reduce(self.period)
        #period = reduce(lcm,self.period)

#        self.repeat *= TPGSEC//period
#        if ((TPGSEC % period) and self.repeat > 0):
#            raise ValueError(f'TPGSEC ({TPGSEC}) is not a multiple of common period {period}')

        #  Brute force it to see how far we get (when will it fail?)
        print('# period {}  args.period {}'.format(period,self.period))
        reps   = [period // p for p in self.period]
        bkts   = [range(self.start[i],period,self.period[i]) 
                  for i in range(len(self.period))]
        bunion = sorted(reduce(myunion,bkts))  # set of buckets with a request
        reqs   = []  # list of request values for those buckets
        for b in bunion:
#            req = 0
            req = []
            for i,bs in enumerate(bkts):
                if b in bs:
#                    req |= (1<<i)
                    req.append(i)
            reqs.append(req)

        blist  = [0] + list(bunion)
        bsteps = list(map(operator.sub,blist[1:],blist[:-1]))
        rem    = period - blist[-1]  # remainder to complete common period

        if verbose:
            print('common period {}'.format(period))
            print('bkts {}'.format(bkts))
            print('bunion {}'.format(bunion))
            print('blist {}  bsteps {}  reqs {}  rem {}'.format(blist,bsteps,reqs,rem))

        #  Reduce common steps+requests into loops
        breps = []
        nreps = 0
        for i in range(1,len(bsteps)):
            if bsteps[i]==bsteps[i-1] and reqs[i]==reqs[i-1]:
                nreps += 1
            else:
                breps.append(nreps)
                nreps = 0
        breps.append(nreps)

        i = 0
        j = 0
        for r in breps:
            if r > 0:
                del bsteps[j:j+r]
                del reqs  [j:j+r]
                if verbose:
                    print('del [{}:{}]'.format(j,j+r))
                    print('bsteps {}'.format(bsteps))
                    print('reqs   {}'.format(reqs))
            j += 1

        if verbose:
            print('breps  {}'.format(breps))
            print('bsteps {}'.format(bsteps))
            print('reqs   {}'.format(reqs))

        #  Now step to each bucket, make the request, and repeat if necessary
        for i,n in enumerate(breps):
            if n > 0:
                self.instr.append('# loop: req {} of step {} and repeat {}'.format(reqs[i],bsteps[i],n))
                self.instr.append('start = len(instrset)')
                if bsteps[i]>0:
                    self._wait(bsteps[i])
                if self.charge is not None:
                    self.instr.append('instrset.append( BeamRequest({}) )'.format(self.charge))
                else:
                    self.instr.append('instrset.append( ControlRequest({}) )'.format(reqs[i]))
                self.ninstr += 1
                self.instr.append('instrset.append( Branch.conditional(start, 0, {}) )'.format(n))
                self.ninstr += 1
            else:
                if bsteps[i]>0:
                    self._wait(bsteps[i])
                if self.charge is not None:
                    self.instr.append('instrset.append( BeamRequest({}) )'.format(self.charge))
                else:
                    self.instr.append('instrset.append( ControlRequest({}) )'.format(reqs[i]))
                self.ninstr += 1
        
        #  Step to the end of the common period and repeat
        if rem > 0:
            self._wait(rem)

        if self.repeat < 0:
            self.instr.append('instrset.append( Branch.unconditional(0) )')
            self.ninstr += 1
        else:
            if self.repeat > 0:
                #  Conditional branch (opcode 2) to instruction 0 (1Hz sync)
                self.instr.append('instrset.append( Branch.conditional(0, 2, {}) )'.format(self.repeat))
                self.ninstr += 1

            if self.notify:
                self.instr.append('instrset.append( CheckPoint() )')
                self.ninstr += 1

            self.instr.append('last = len(instrset)')
            self.instr.append('instrset.append( FixedRateSync(marker="1H",occ=1) )')
            self.instr.append('instrset.append( Branch.unconditional(last) )')
            self.ninstr += 2

def main():
    parser = argparse.ArgumentParser(description='Periodic sequence generator')
    parser.add_argument("-p", "--period"            , required=True , nargs='+', type=int, 
                        help="buckets between start of each train")
    parser.add_argument("-s", "--start_bucket"      , required=True , nargs='+', type=int,
                        help="starting bucket for first train")
    parser.add_argument("-d", "--description"       , required=True , nargs='+', type=str,
                        help="description for each event code")
    parser.add_argument("-r", "--repeat"            , default=-1 , type=int,
                        help="number of times to repeat 1 second sequence (default: indefinite)")
    parser.add_argument("-n", "--notify"            , action='store_true',
                        help="assert SeqDone PV when repeats are finished")
    args = parser.parse_args()
    print('# periodicgenerator args {}'.format(args))
    gen = PeriodicGenerator(period=args.period, start=args.start_bucket, repeat=args.repeat, notify=args.notify)
    if (gen.ninstr > 1000):
        sys.stderr.write('*** Sequence has {} instructions.  May be too large to load. ***\n'.format(gen.ninstr))
    print('# {} instructions'.format(gen.ninstr))

    if len(args.description):
        seqcodes = {i:s for i,s in enumerate(args.description)}
    else:
        seqcodes = {}

    print('')
    print('seqcodes = {}'.format(seqcodes))
    print('')
    for i in gen.instr:
        print('{}'.format(i))

if __name__ == '__main__':
    main()
