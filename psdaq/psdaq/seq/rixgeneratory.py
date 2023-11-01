#  Generates eventcodes for 2-integration cameras (fast,slow) and the bunch train (to simulate)
#  The trigger periods must align with the bunch spacing, else the DAQ will reject the camera triggers
from psdaq.seq.traingenerator import *
from psdaq.seq.periodicgenerator import *
from psdaq.seq.globals import *
from psdaq.seq.seq import *
import argparse
import logging
import time

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

def one_camera_sequence(args):

    print(f'Generating one repeating bunch train for one camera')

    args_period = [int(TPGSEC*p) for p in args.periods]

    d = {'period' : args_period[0]//args.bunch_period,
         'readout': int(args.readout_time[0]*TPGSEC+args.bunch_period-1)//args.bunch_period}
    print(f'd {d}')

    #  The bunch trains
    gen = TrainGenerator(start_bucket     =(d['readout']+1)*args.bunch_period,
                         train_spacing    =d['period']*args.bunch_period,
                         bunch_spacing    =args.bunch_period,
                         bunches_per_train=d['period']-d['readout'],
                         repeat           =1,
                         charge           =None,
                         notify           =False,
                         rrepeat          =True)
    seqcodes = {0:'Bunch Train'}
    write_seq(gen,seqcodes,'beam.py')
    
    #  The Andor triggers
    gen = PeriodicGenerator(period=[d['period']*args.bunch_period],
                            start =[0],
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
    gen = TrainGenerator(start_bucket     =(d['slow']['readout']+1)*args.bunch_period,
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

    #  The Andor triggers
    gen = PeriodicGenerator(period=[d['slow']['period']*args.bunch_period,
                                    d['fast']['period']*args.bunch_period],
                            start =[0,0],
                            charge=None,
                            repeat=-1,
                            notify=False)

    seqcodes = {0:'Slow Andor',1:'Fast Andor'}
    write_seq(gen,seqcodes,'codes.py')

    #  Since we don't actually have beam to include in the trigger logic,
    #  we need to gate the generation of the fast Andor eventcode to simulate it.
    gen = TrainGenerator(start_bucket     =(n+1)*d['fast']['period']*args.bunch_period,
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
    parser.add_argument("--readout_time", default=[0.391,0.005], type=float, help="camera readout times (sec); default=[0.391,0.005]")
    parser.add_argument("--bunch_period", default=28, type=int, help="spacing between bunches in the train; default=28")
    parser.add_argument("--laser_onoff", type=int, nargs='+', 
                        help='''
                        slow cam periods that laser is on,off,on,off,...
                        example: 31,1,17,1,37,1,11,1  
                        on 31, off 1, on 17, off 1, on 37, off 1, on 11, off 1. 
                        (repeats every 100)''')
    args = parser.parse_args()

    if len(args.periods) > 2:
        raise ValueError(f'Too many periods {args.periods}.  Limited to 2')

    args_period = [int(TPGSEC*p) for p in args.periods]

    #  validate integration periods with bunch_period
    for a in args_period:
        if (a%args.bunch_period):
            raise ValueError(f'period {a} ({a/TPGSEC} sec) is not a multiple of the bunch period {args.bunch_period}')

    if len(args_period) == 1:
        one_camera_sequence(args)
    else:
        two_camera_sequence(args)

if __name__ == '__main__':
    main()

