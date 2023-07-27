import argparse
import os
import sys
from .globals import *

#
#  Arguments are:
#     start_bucket      : starting bucket of first train within the pattern
#     bunch_spacing     : buckets between bunches within the train
#     bunches_per_train : bunches within each train 
#     train_spacing     : buckets from last bunch of previous train to
#                         first bunch of the next train
#     repeat            : # of times to repeat (-1 = indefinite)
#     notify            : assert checkpoint when done
#
class TrainGenerator(object):
#    COUNT_RANGE=256
    COUNT_RANGE=1024

    def __init__(self, start_bucket=0, 
                 train_spacing=TPGSEC,
                 bunch_spacing=1, bunches_per_train=1, 
                 charge=0, repeat=0, notify=False, 
                 rrepeat=False, rpad=None):
        self.start_bucket      = start_bucket
        self.train_spacing     = train_spacing
        self.bunch_spacing     = bunch_spacing
        self.bunches_per_train = bunches_per_train
        self.request           = 'ControlRequest([0])' if charge is None else 'BeamRequest({})'.format(charge)
        self.repeat            = repeat
        self.notify            = notify
        self.rrepeat           = rrepeat
        self.rpad              = rpad
        self.async_start       = None if repeat else 0
# Need to add self.reqs[i] as an argument for ControlRequest({})
#        print(vars(self))

        self.instr = ['instrset = []']
        self._fill_instr()

    def _train(self, cc):
        intb = self.bunch_spacing
        nb   = self.bunches_per_train
        w    = 0
        self.instr.append('#   {} bunches / _train'.format(nb))
        self.instr.append('instrset.append({})'.format(self.request))

        rb = nb-1
        if rb:
            if rb > 0xfff:
                self.instr.append('iinstr=len(instrset)')
                self._wait(intb)
                self.instr.append('instrset.append({})'.format(self.request))
                self.instr.append('instrset.append(Branch.conditional(line=iinstr,counter={},value={}))'.format(cc[0],0xfff))
                self.instr.append('instrset.append(Branch.conditional(line=iinstr,counter={},value={}))'.format(cc[1],(rb//0x1000) -1))
                rb = rb & 0xfff

            if rb:
                self.instr.append('iinstr=len(instrset)')
                self._wait(intb)
                self.instr.append('instrset.append({})'.format(self.request))
                if rb > 1:
                    self.instr.append('instrset.append(Branch.conditional(line=iinstr,counter={},value={}))'.format(cc[0],rb-1))
            w = intb*(nb-1)
        return w

    def _wait(self, intv):
        if intv <= 0:
            raise ValueError
        if intv >= 0xfff:
            self.instr.append('iinstr = len(instrset)')
            #  _Wait for 4095 intervals
            self.instr.append('instrset.append( FixedRateSync(marker="910kH", occ=4095) )')
            if intv >= 0x1ffe:
                #  Branch conditionally to previous instruction
                self.instr.append('instrset.append( Branch.conditional(line=iinstr, counter=3, value={}) )'.format(int(intv/0xfff)-1))

        rint = intv%0xfff
        if rint:
            self.instr.append('instrset.append( FixedRateSync(marker="910kH", occ={} ) )'.format(rint))

    def _trains(self,intv,nint):
        rint = nint % self.COUNT_RANGE
        if rint:
            self.instr.append('# loop A: {} _trains'.format(rint))
            self.instr.append('startreq = len(instrset)')
            self._wait(intv-self._train([1,2]))
            if rint > 1:
                self.instr.append('instrset.append( Branch.conditional(startreq, 0, {}) )'.format(rint-1))
            self.instr.append('# end loop A')
            nint = nint - rint

        rint = (nint/self.COUNT_RANGE) % self.COUNT_RANGE
        if rint:
            self.instr.append('# loop B: {} _trains'.format(rint*self.COUNT_RANGE))
            self.instr.append('startreq = len(instrset)')
            self._wait(intv-self._train([1]))  # don't need 2 counters because trains_per_bunch>4096 can't have trains_per_second>self.COUNT_RANGE
            self.instr.append('instrset.append( Branch.conditional(startreq, 0, self.COUNT_RANGE-1) )')
            if rint > 1:
                self.instr.append('instrset.append( Branch.conditional(startreq, 2, {}) )'.format(rint-1))
            self.instr.append('# end loop B')
            nint = nint - rint*self.COUNT_RANGE

        rint = (nint / (self.COUNT_RANGE*self.COUNT_RANGE)) % self.COUNT_RANGE
        if rint:
            self.instr.append('# loop C: {} _trains'.format(rint*self.COUNT_RANGE*self.COUNT_RANGE))
            self.instr.append('# loop (n_trains / self.COUNT_RANGE)')
            self.instr.append('startreq = len(instrset)')
            self._wait(intv-self._train([3]))  # can use counter 3 here because no delay can be greater than 4096 (actually 3554)
            self.instr.append('instrset.append( Branch.conditional(line=startreq, counter=2, value=self.COUNT_RANGE-1) )')
            self.instr.append('instrset.append( Branch.conditional(line=startreq, counter=1, value=self.COUNT_RANGE-1) )')
            if rint > 1:
                self.instr.append('instrset.append( Branch.conditional(line=startreq, counter=0, value={}) )'.format(rint-1))
            self.instr.append('# end loop C')
            nint = nint - rint*self.COUNT_RANGE*self.COUNT_RANGE


    def _fill_instr(self):
        #  How many times to repeat beam requests in "1 second"
        #  nint = TPGSEC/intv
        #  Global sync counts as 1 cycle
        intv = self.train_spacing
 #       nint = TPGSEC/intv

 #       if nint>0:
 #           sys.stderr.write('#Generating {} _trains with {} _train spacing\n'.
 #                 format(nint,intv))
 #       if self.bunches_per_train>1: 
 #           sys.stderr.write('#\tcontaining {} bunches with {} spacing\n'.
 #                 format(self.bunches_per_train,
 #                        self.bunch_spacing))

        #  Initial validation: train doesn't exceed one second
#        if ((nint-1)*intv+(self.bunches_per_train-1)*self.bunch_spacing) >= TPGSEC:
#            sys.stderr.write(f'*** train length exceeds one second: nint {nint}  intv {intv}  bu_per_tr {self.bunches_per_train}  bu_sp {self.bunch_spacing}')
#            raise ValueError

        if self.start_bucket>0:
            self.instr.append('# start at bucket {}'.format(self.start_bucket))
            self._wait(self.start_bucket)

        self.instr.append('first = len(instrset)')

        if self.repeat < 0:
            len = self._train([1,2])
            self._wait(intv-len)
            self.instr.append('instrset.append( Branch.unconditional(first) )')
        else:
            if self.repeat > 0:
                self._trains(intv,self.repeat)
                #  Conditional branch (opcode 2) to instruction 0 (1Hz sync)
            if self.notify:
                self.instr.append('instrset.append( CheckPoint() )')
            if self.rrepeat:
                self._wait(self.rpad)
                self.instr.append('instrset.append( Branch.unconditional(first) )')
            else:
                self.instr.append('last = len(instrset)')
                self.instr.append('instrset.append( FixedRateSync(marker="1H",occ=1) )')
                self.instr.append('instrset.append( Branch.unconditional(last) )')


def main():
    parser = argparse.ArgumentParser(description='simple validation printing')
    parser.add_argument("-t", "--train_spacing"     , required=True , type=int, help="buckets between start of each _train")
    parser.add_argument("-b", "--bunch_spacing"     , required=True , type=int, help="buckets between bunches within _train")
    parser.add_argument("-n", "--bunches_per_train" , required=True , type=int, help="number of bunches in each _train")
    parser.add_argument("-s", "--start_bucket"      , default=0     , type=int, help="starting bucket for first _train")
    parser.add_argument("-q", "--charge"            , default=None  , type=int, help="bunch charge, pC")
    parser.add_argument("-r", "--repeat"            , default=0     , type=int, help="number of times to repeat; -1=infinite")
    parser.add_argument("-N", "--notify"            , action='store_true',
                        help="assert SeqDone PV when repeats are finished")
    parser.add_argument("-d", "--description"       , required=True , type=str, help="description for event code")
    args = parser.parse_args()

    print('# traingenerator args {}'.format(args))
    gen = TrainGenerator(args.start_bucket, args.train_spacing,
                         args.bunch_spacing, args.bunches_per_train, args.charge, args.repeat, args.notify)
    if (len(gen.instr) > 1000):
        sys.stderr.write('*** Sequence has {} instructions.  May be too large to load. ***\n'.format(gen.ninstr))
    print('# {} instructions'.format(len(gen.instr)))

    seqcodes = {0:args.description}

    print('')
    print('seqcodes = {}'.format(seqcodes))
    print('')
    for i in gen.instr:
        print('{}'.format(i))

if __name__ == '__main__':
    main()
