import argparse
import json
import os
import sys
import numpy
from itertools import chain
from functools import reduce
import operator
import psdaq.configdb.tsdef as ts # for marker/interval mapping

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
    '''period : value or list of values representing the interval to be repeated.
       start  : value or list of values of the bucket on which to start the period(s).
       charge : electron bunch charge for beam requests.  Only used by TPG.
       repeat -1 : repeat forever
               0 : don't repeat
               n : repeat n times
       notify : insert a notify instruction when the entire sequence is completed
       merge  : output ControlRequest([0]) on any of the periods, else ControlRequest([n..]) for each period n.
       marker : marker id for period counts
       resync : insert a slow fixed rate marker at the end of the sequence to keep it aligned in cases of XPM 
                transmission/receive drops.  This can only happen if the repeating part of the sequence does not cross
                the slow marker boundaries.
    '''
    def __init__(self, period, start, charge=None, repeat=-1, notify=False, merge=False, marker=None, resync=True):
        self.charge = charge
        self.merge  = merge
        if marker is None:
            for k,v in ts.FixedIntvsDict.items():
                if v["intv"]==1:
                    marker = k
                    break
        self.resync = resync
        self.init(period, start, marker, repeat, notify)

    def init(self, period, start, marker='910kH', repeat=-1, notify=False):
        self.async_start       = 0
        if isinstance(period,list):
            if len(period) != len(start):
                raise ValueError('period and start lists must be equal length')
            self.period    = period
            self.start     = start
        else:
            self.period    = [period]
            self.start     = [start]
        if marker in ts.FixedIntvsDict.keys():
            self.syncins = f'Wait( marker=\"{marker}\"'
        elif marker[0]=='a':
            rate, tslots = marker[1:].split('t')
            tsm = 0
            for t in tslots:
                tsm |= 1<<(int(t)-1)
            self.syncins = f'WaitA( {tsm}, \"{rate}\"'
        else:
            options = list(ts.FixedIntvsDict.keys())
            options.append( f'a{ts.acRates}t[1..6]')
            raise ValueError(f'marker {marker} not recognized. Options are {options}')

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
        #  Use the WaitX macro to simplify the output
        self.instr.append(f'instrset.append( {self.syncins}, occ={intv} ) )' )
        self.ninstr += 1

    def _fill_instr(self):
        #  Common period (subharmonic)
        period = numpy.lcm.reduce(self.period)

        #  Brute force it to see how far we get (when will it fail?)
        print('# period {}  args.period {}'.format(period,self.period))
        reps   = [period // p for p in self.period]
        last_start = numpy.max(self.start)

        #  First handle the interval up to the last start bucket
        if last_start > 0:
            bkts = [range(self.start[i],last_start,self.period[i]) for i in range(len(self.period))]
            self.fill_bkts(bkts,last_start)

        #  Then, repeat the period starting after the last start bucket
        start_repeat = self.ninstr
        bkts = []
        for i in range(len(self.start)):
            if self.start[i]<last_start:
                np = 1+(last_start-self.start[i]-1)//self.period[i]
                bkts.append(range(np*self.period[i]+self.start[i]-last_start,period,self.period[i]))
            else:
                bkts.append(range(0,period,self.period[i]))
        if self.fill_bkts(bkts,period,resync=self.resync,start=last_start):
            start_repeat = 0

        if self.repeat < 0:
            self.instr.append(f'instrset.append( Branch.unconditional({start_repeat}) )')
            self.ninstr += 1
        else:
            if self.repeat > 0:
                self.instr.append(f'instrset.append( Branch.conditional({start_repeat}, 2, {self.repeat}) )')
                self.ninstr += 1

            if self.notify:
                self.instr.append('instrset.append( CheckPoint() )')
                self.ninstr += 1

            self.instr.append('last = len(instrset)')
            self.instr.append('instrset.append( Wait(marker="1H",occ=1) )')
            self.instr.append('instrset.append( Branch.unconditional(last) )')
            self.ninstr += 2


    def fill_bkts(self,bkts,period,resync=False,start=None):
        bunion = sorted(reduce(myunion,bkts))  # set of buckets with a request
        reqs   = []  # list of request values for those buckets
        for b in bunion:
            req = []
            for i,bs in enumerate(bkts):
                if b in bs:
                    req.append(i)
            reqs.append(req)

        blist  = [0] + list(bunion)
        bsteps = list(map(operator.sub,blist[1:],blist[:-1]))
        rem    = period - blist[-1]  # remainder to complete common period

        if verbose:
            print('#common period {}'.format(period))
            print('#bkts {}'.format(bkts))
            print('#bunion {}'.format(bunion))
            print('#blist {}  bsteps {}  reqs {}  rem {}'.format(blist,bsteps,reqs,rem))

        if len(bsteps)==0:
            self._wait(rem)
            return

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
                    print('#del [{}:{}]'.format(j,j+r))
                    print('#bsteps {}'.format(bsteps))
                    print('#reqs   {}'.format(reqs))
            j += 1

        if verbose:
            print('#breps  {}'.format(breps))
            print('#bsteps {}'.format(bsteps))
            print('#reqs   {}'.format(reqs))

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
                    self.instr.append('instrset.append( ControlRequest({}) )'.format([0] if self.merge else reqs[i]))
                self.ninstr += 1
                self.instr.append('instrset.append( Branch.conditional(start, 0, {}) )'.format(n))
                self.ninstr += 1
            else:
                if bsteps[i]>0:
                    self._wait(bsteps[i])
                if self.charge is not None:
                    self.instr.append('instrset.append( BeamRequest({}) )'.format(self.charge))
                else:
                    self.instr.append('instrset.append( ControlRequest({}) )'.format([0] if self.merge else reqs[i]))
                self.ninstr += 1
        
        #  Step to the end of the common period and repeat
        if rem > 0:
            #  This is an opportunity to resync
            if resync:
                for k,v in ts.FixedIntvsDict.items():
                    #  Check that period is a marker interval and pattern does not exceed the marker interval
                    if period==v['intv'] and start<rem:
                        print(f'PeriodicGenerator: Filling remainder {rem} with sync to {k}, start {start}, period {period}')
                        self.instr.append(f'instrset.append( Wait(marker="{k}",occ=1) )')
                        return True
            #  No resync
            self._wait(rem)

        return False

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
    parser.add_argument("-m", "--merge"            , action='store_true',
                        help="merge all triggers onto one code")
    parser.add_argument("-M", "--marker"            , default=None, type=str,
                        help="Marker for counting buckets (910kH, 1H, for example).\nAC markers are specified as aRRtXX, where RR=(60H,30H,10H,5H,0.5H) and XX is a combination of up to 6 digits in the range 1-6.\nFor example, a60Ht123456 is 360Hz.")
    args = parser.parse_args()
    print('# periodicgenerator args {}'.format(args))
    #marker = None if args.marker is None else f'\"{args.marker}\"'
    marker = None if args.marker is None else f'{args.marker}'
    gen = PeriodicGenerator(period=args.period, start=args.start_bucket, repeat=args.repeat, marker=marker, notify=args.notify, merge=args.merge, resync=False)
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
