#  Generates eventcodes for 2-integration cameras (fast,slow) and the bunch train (to simulate)
#  The trigger periods must align with the bunch spacing, else the DAQ will reject the camera triggers
from psdaq.seq.traingenerator import *
from psdaq.seq.periodicgenerator import *
from psdaq.seq.globals import *
from psdaq.seq.seq import *
from psdaq.seq.sub_rates import sub_rates
import argparse
import logging
import time

#  Beam offset from 1H marker
SXR_OFFSET = 7

#factors = {28:(500,65),  # 32500
#           35:(2000,13)} # 26000

def get_factor(product):

    if product <= 4096:
        return (product,1)

    for a in range(4096,1,-1):
        if product%a == 0:
            return (product//a,a)

    raise ValueError(f'{product} does not factor')

def factorize(product):
    result = []

    p = product
    while (p>1):
        f,p = get_factor(p)
        result.append(f)

    print(f'factors of {product} : {result}')
    return result

class LaserGenerator(object):
    def __init__(self,start_bucket,bunch_period,branch_counts,onoff):
        self.instr = []
        self.ninstr = 0
        self.instr.append('instrset = []')
        #  Offset the entire sequence from the 1H marker
        if start_bucket:
            self.instr.append(f'instrset.append( FixedRateSync(marker="910kH", occ={start_bucket}) )')
        #  Loop through on/off sequences alternating subroutines
        req = 0
        for i,n in enumerate(onoff):
            self.one_second(bunch_period,branch_counts,req)
            if n>1:
                self.instr.append(f'instrset.append( Branch.conditional(line=start,counter=3,value={n-1}) )')
                self.ninstr += 1
            req = 1-req
        self.instr.append(f'instrset.append( Branch.unconditional(line={1 if start_bucket else 0}) )')
        self.ninstr += 1

    def one_second(self,bunch_period,branch_counts,req):
        self.instr.append('start = len(instrset)')
        self.instr.append(f'instrset.append( ControlRequest([{req}]) )')
        self.instr.append(f'instrset.append( FixedRateSync(marker="910kH", occ={bunch_period}) )')
        self.ninstr += 2
        for i,f in enumerate(branch_counts):
            self.instr.append(f'instrset.append( Branch.conditional(line=start,counter={i},value={f-1}) )')
            self.ninstr += 1

    #  The call/return instruction doesn't seem to be working correctly
    #  Needs some vhdl simulation work
    def init(self,bunch_period,branch_counts,onoff):
        self.instr = []
        self.instr.append('instrset = []')
        self.instr.append('instrset.append( Branch.unconditional(line=0) )') # placeholder
        self.ninstr = 1
        #  laser on subroutine
        self.instr.append('subr_on = len(instrset)')
        self.instr.append(f'instrset.append( FixedRateSync(marker="910kH", occ={bunch_period}) )')
        self.instr.append('instrset.append( ControlRequest([0]) )')
        for i,f in enumerate(branch_counts):
            self.instr.append(f'instrset.append( Branch.conditional(line=subr_on,counter={i},value={f-1}) )')
        self.instr.append('instrset.append( Return() )')
        self.ninstr += 3 + len(branch_counts)
        # laser off subroutine
        self.instr.append('subr_off = len(instrset)')
        self.instr.append(f'instrset.append( FixedRateSync(marker="910kH", occ={bunch_period}) )')
        self.instr.append(f'instrset.append( ControlRequest([1]) )')
        for i,f in enumerate(branch_counts):
            self.instr.append(f'instrset.append( Branch.conditional(line=subr_off,counter={i},value={f-1}) )')
        self.instr.append('instrset.append( Return() )')
        self.ninstr += 3 + len(branch_counts)
        self.instr.append('subr_end = len(instrset)')
        self.instr.append('instrset[0] = Branch.unconditional(line=subr_end)')
        self.ninstr += 1
        #  Loop through on/off sequences alternating subroutines
        for i,n in enumerate(onoff):
            self.instr.append(f'i = len(instrset)')
            self.instr.append(f'instrset.append( Call({"subr_on" if i%2==0 else "subr_off"}) )')
            if n>1:
                self.instr.append(f'instrset.append( Branch.conditional(line=i,counter=3,value={n-1}) )')
                self.ninstr += 1
        self.instr.append('instrset.append( Branch.unconditional(line=subr_end) )')
        self.ninstr += len(onoff) + 1

def write_seq(gen,seqcodes,filename):

    if (len(gen.instr) > 1000):
        sys.stderr.write('*** Sequence has {} instructions.  May be too large to load. ***\n'.format(gen.ninstr))

    with open(filename,"w") as f:
        f.write('# {} instructions\n'.format(len(gen.instr)))

        f.write('\n')
        f.write('seqcodes = {}\n'.format(seqcodes))
        f.write('\n')
        for i in gen.instr:
            f.write('{}\n'.format(i))

    validate(filename)
    print(f'** Wrote {filename} **')

def one_camera_sequence(args):

    print(f'Generating one repeating bunch train for one camera')

    args_period = [int(TPGSEC*p) for p in args.periods]

    d = {'period' : args_period[0]//args.bunch_period,
         'readout': int(args.readout_time[0]*TPGSEC+args.bunch_period-1)//args.bunch_period}
    print(f'd {d}')

    #  The bunch trains
    gen = TrainGenerator(start_bucket     =(d['readout']+1)*args.bunch_period+SXR_OFFSET,
                         train_spacing    =d['period']*args.bunch_period,
                         bunch_spacing    =args.bunch_period,
                         bunches_per_train=d['period']-d['readout'],
                         repeat           =1,
                         charge           =None,
                         notify           =False,
                         rrepeat          =True)
    seqcodes = {0:'Bunch Train'}
    write_seq(gen,seqcodes,'beam.py')
    
    #  The laser triggers (on/off)
    factors = factorize(args_period[0]//args.bunch_period)
    gen = LaserGenerator(start_bucket=SXR_OFFSET,
                         bunch_period=args.bunch_period,
                         branch_counts=factors,
                         onoff=args.laser_onoff)
    seqcodes = {0:'Laser On',1:'Laser Off'}
    write_seq(gen,seqcodes,'laser.py')
    
    #  The Andor triggers
    gen = PeriodicGenerator(period=[d['period']*args.bunch_period],
                            start =[SXR_OFFSET],
                            charge=None,
                            repeat=-1,
                            notify=False)

    seqcodes = {0:'Slow Andor'}
    write_seq(gen,seqcodes,'codes.py')

def two_camera_sequence(args):

    args_period = [int(TPGSEC*p) for p in args.periods]

    #  period,readout times translated to units of bunch spacing
    d = {}
    if args_period[0] > args_period[1]:
        d['slow'] = {'period' : args_period[0]//args.bunch_period,
                     'readout': int(args.readout_time[0]*TPGSEC+args.bunch_period-1)//args.bunch_period}
        d['fast'] = {'period' : args_period[1]//args.bunch_period,
                     'readout': int(args.readout_time[1]*TPGSEC+args.bunch_period-1)//args.bunch_period}
    else:
        d['fast'] = {'period' : args_period[0]//args.bunch_period,
                     'readout': int(args.readout_time[0]*TPGSEC+args.bunch_period-1)//args.bunch_period}
        d['slow'] = {'period' : args_period[1]//args.bunch_period,
                     'readout': int(args.readout_time[1]*TPGSEC+args.bunch_period-1)//args.bunch_period}

    #  expand the slow gap to be a multiple of the fast period
    n = (d['slow']['readout'] - d['fast']['readout'] + d['fast']['period'] - 1) // d['fast']['period']
    d['slow']['readout'] = n*d['fast']['period']+d['fast']['readout']
    rpad = d['slow']['period']*args.bunch_period-(d['slow']['period']//d['fast']['period']-n)*d['fast']['period']*args.bunch_period

    print(f'd {d}  n {n}  rpad {rpad}')

    #  The bunch trains
    gen = TrainGenerator(start_bucket     =(d['slow']['readout']+1)*args.bunch_period+SXR_OFFSET,
                         train_spacing    =d['fast']['period']*args.bunch_period,
                         bunch_spacing    =args.bunch_period,
                         bunches_per_train=d['fast']['period']-d['fast']['readout'],
                         repeat           =d['slow']['period'] // d['fast']['period'] - n,
                         charge           =None,
                         notify           =False,
                         rrepeat          =True,
                         rpad             =rpad)
    seqcodes = {0:'Bunch Train'}
    write_seq(gen,seqcodes,'beam.py')

    #  The laser triggers (on/off)
    factors = factorize(args_period[0]//args.bunch_period)
    gen = LaserGenerator(start_bucket=SXR_OFFSET,
                         bunch_period=args.bunch_period,
                         branch_counts=factors,
                         onoff=args.laser_onoff)
    seqcodes = {0:'Laser On',1:'Laser Off'}
    write_seq(gen,seqcodes,'laser.py')
    
    #  The Andor triggers
    gen = PeriodicGenerator(period=[d['slow']['period']*args.bunch_period,
                                    d['fast']['period']*args.bunch_period],
                            start =[SXR_OFFSET,SXR_OFFSET],
                            charge=None,
                            repeat=-1,
                            notify=False)

    seqcodes = {0:'Slow Andor',1:'Fast Andor'}
    write_seq(gen,seqcodes,'codes.py')

    #  Since we don't actually have beam to include in the trigger logic,
    #  we need to gate the generation of the fast Andor eventcode to simulate it.
    gen = TrainGenerator(start_bucket     =(n+1)*d['fast']['period']*args.bunch_period+SXR_OFFSET,
                         train_spacing    =d['fast']['period']*args.bunch_period,
                         bunch_spacing    =args.bunch_period,
                         bunches_per_train=1,
                         repeat           =d['slow']['period'] // d['fast']['period'] - n,
                         charge           =None,
                         notify           =False,
                         rrepeat          =True,
                         rpad             =rpad)
    seqcodes = {0:'Fast Andor Gated'}
    write_seq(gen,seqcodes,'codes2.py')

def main():
    global args
    parser = argparse.ArgumentParser(description='rix 2-integrator mode',formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--periods", default=[1,0.01], type=float, nargs='+', help="integration periods (sec); default=[1,0.01]")
    parser.add_argument("--readout_time", default=[0.391,0.005], type=float, nargs='+', help="camera readout times (sec); default=[0.391,0.005]")
    parser.add_argument("--bunch_period", default=28, type=int, help="spacing between bunches in the train; default=28")
    parser.add_argument("--laser_onoff", type=int, nargs='+', 
                        default=[1,2,3,1],
                        help='''
                        slow cam periods that laser is on,off,on,off,...
                        example: 31,1,17,1,37,1,11,1  
                        on 31, off 1, on 17, off 1, on 37, off 1, on 11, off 1. 
                        (repeats every 31+1+17+1+37+1+11+1=100)''')
    parser.add_argument('--override', action='store_true', help='Do not correct readout periods')

    args = parser.parse_args()

    if len(args.periods) > 2:
        raise ValueError(f'Too many periods {args.periods}.  Limited to 2')

    args_period = [int(TPGSEC*p) for p in args.periods]

    #  validate integration periods with bunch_period
    for i,a in enumerate(args_period):
        l_bunch_period = a%args.bunch_period
        l_tpgsec = TPGSEC%a
        if l_bunch_period:
            logging.warning(f'period {a} ({a/TPGSEC} sec) is not a multiple of the bunch period {args.bunch_period}.')
        if l_tpgsec:
            logging.warning(f'period {a} ({a/TPGSEC} sec) is not an integer factor of the TPG 1Hz period')

        if l_bunch_period or l_tpgsec:
            if args.override:
                logging.warning('period {a} is not corrected')
                continue

            rates = sub_rates(args.bunch_period)
            newa = None
            for r in sorted(rates):
                if r[2]<a:
                    break
                newa = r[2]
            if newa is None:
                logging.error('No valid period replacement found')
                return
            logging.warning(f'Raising period {a/TPGSEC} sec to {newa/TPGSEC} sec.')
            args.periods[i] = newa/TPGSEC

    if len(args_period) == 1:
        one_camera_sequence(args)
    else:
        two_camera_sequence(args)

if __name__ == '__main__':
    main()

