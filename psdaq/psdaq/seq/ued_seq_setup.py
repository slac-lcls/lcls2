"""  Sequence the xpm to generate two eventcodes  
     eventcode 1 is the andor exposure trigger
     eventcode 0 is the readout trigger for timestamping
     The readout trigger should precede andor image readout completion
     The next exposure trigger should be timed after the andor is again ready
"""  
import argparse
import os
import tempfile
import subprocess
import epics
import os
from psdaq.seq.seq import Instruction
from psdaq.cas.pvedit import Pv

READOUT_TRIGGER_DELTA  = -4/360. # Time before/after exposure setting

#  Dictionary of PV prefixes for the camera, exposure trigger, and timestamp trigger
d = {1:{'cam':'UED:ANDOR:CAM:01:',
        'exp':'UED:CAM:TPR:01:CH02_',
        'tim':'UED:CAM:TPR:01:CH01_'},
     3:{'cam':'UED:ANDOR:CAM:03:',
        'exp':'UED:CAM:TPR:01:CH00_',
        'tim':'UED:CAM:TPR:01:CH04_'},
     0:{'cam': None,
        'exp': None,
        'tim': None}}

def main():
    parser = argparse.ArgumentParser(description='ued sequencer programming')
    parser.add_argument('--pv', type=str, default='DAQ:UED:XPM:0', help="sequence engine pv; default DAQ:UED:XPM:0")
    parser.add_argument('--cam', type=int, default=1, help="Andor camera number (1,3)")
    parser.add_argument("--eng", type=int, default=0, help="sequence engine; default=0")
    parser.add_argument("--period", type=float, default=None, help="camera exposure setting (sec)")
    parser.add_argument("--a60" , action='store_true', help="60Hz continuous mode")
    parser.add_argument("--test", action='store_true', help="test only")
    parser.add_argument("--ac"  , action='store_true', help="ac time base")
    parser.add_argument("--margin", type=float, default=None, help="addition to trigger interval")
    parser.add_argument("--setup", type=float, default=0.002, help="subtraction for timestamp trigger")
    parser.add_argument("--verbose", action='store_true', help="verbose printout")
    args = parser.parse_args()

    if args.margin is None:
        args.margin = 0.010 if args.ac else 0.002

    #
    #  Setup the TPR and CAMERA via EPICS
    #
    if args.cam not in d.keys():
        raise ValueError(f"--cam argument must be one of {d.keys()}")

    campv   = d[args.cam]['cam']
    exptrpv = d[args.cam]['exp']
    timtrpv = d[args.cam]['tim']

    # default
    exp_period = args.period

    if not args.test and campv is not None:
        os.environ["EPICS_CA_AUTO_ADDR_LIST"]="NO"
        os.environ["EPICS_CA_ADDR_LIST"]="172.27.99.255:5058"

        #  Setup the camera
        epics.caput(campv+'Acquire',0)                # stop
        epics.caput(campv+'AcquireTime',args.period)
        epics.caput(campv+'TriggerMode',1)            # external
        epics.caput(campv+'ImageMode',2)              # continuous
        epics.caput(campv+'AndorFastExtTrigger',0)    # disable
        epics.caput(campv+'TSS:TsPolicy',0)           # LAST_EC
        exp_period = epics.caget(campv+'AcquirePeriod_RBV')

        #  Setup the triggers
        epics.caput(exptrpv+'RATEMODE',2)  # event code
        epics.caput(exptrpv+'SEQCODE',257+4*args.eng) 
        epics.caput(timtrpv+'RATEMODE',2)  # event code
        epics.caput(timtrpv+'SEQCODE',256+4*args.eng) 
            
    if args.period is None:
        #  Program the sequence
        fname = 'ued_60Hz.py' if args.a60 else 'ued_360Hz.py'
        path = f'{os.path.dirname(os.path.realpath(__file__))}/{fname}'
        cmd = ["seqprogram","--pv", args.pv,"--seq", f'{args.eng}:{path}', '--start','--reset']
        print(cmd)
        if not args.test and campv is not None:
            result = subprocess.run(cmd)
            #  Remove DAQ control over seqence enable/disable
            pvSeqMask = Pv(f'{args.pv}:PART:0:SeqMask')
            pvSeqMask.put(0)
    else:
        if args.ac:
            #  Use the power line markers
            clkrate = 360
            def marker(intv):
                return f'ACRateSync( 63, \"60H\", occ={intv} )'
        else:
            #  Use the 100kH rate markers
            clkrate = 100e3
            def marker(intv):
                return f'FixedRateSync( \"100kH\", occ={intv} )'

        #rot = int(clkrate*(args.period+READOUT_TRIGGER_DELTA) +0.5)
        rot = int(clkrate*(args.period-args.setup) +0.5)
        exp = int(clkrate*(exp_period+args.margin) +0.5)

        if exp==0:
            raise ValueError(f'Input period {args.period} sec is less than 1/360Hz')

        print(f'** Exposure interval is {exp} at {clkrate}Hz = {exp/clkrate} sec **')
        print(f'** Readout delay     is {rot} at {clkrate}Hz = {rot/clkrate} sec **')

        fd, path = tempfile.mkstemp()
        with open(fd,'w') as f:

            def wait(intv):
                if intv <= Instruction.maxocc:
                    f.write(f'instrset.append( {marker(intv)} )\n')
                else:
                    n   = int(intv/Instruction.maxocc)
                    if n > Instruction.maxocc:
                        raise ValueError(f'Period {args.period} sec is too large.')
                    rem = intv%Instruction.maxocc
                    f.write(f'ln = len(instrset)\n')
                    f.write(f'instrset.append( {marker(Instruction.maxocc)} )\n')
                    if n>1:
                        f.write(f'instrset.append( Branch.conditional( line=ln, counter=0, value={n-1} ) )\n')
                    f.write(f'instrset.append( {marker(rem)} )\n')

            seqcodes = {0: f'Readout {rot/clkrate} sec', 1:'Exposure {exp/clkrate} sec'}
            f.write(f'seqcodes = {seqcodes}\n')
            f.write(f'instrset = []\n')
            f.write(f'instrset.append( ACRateSync( 63, \"60H\", occ=1 ) )\n')
            f.write(f'instrset.append( ControlRequest([1]) )\n')
            wait(rot)
            f.write(f'instrset.append( ControlRequest([0]) )\n')
            wait(exp-rot)
            f.write(f'instrset.append( Branch.unconditional(1) )\n')
            f.close()

        #  Program the sequence
        cmd = ["seqprogram","--pv", args.pv,"--seq", f'{args.eng}:{path}', '--reset']
        if args.verbose:
            cmd.append("--verbose")

        if not args.test:
            result = subprocess.run(cmd)
            os.remove(path)
            #  Disable the sequence now
            pvSeqEn = Pv(f'{args.pv}:SEQENG:{args.eng}:ENABLE')
            pvSeqEn.put(0)
            #  Let the DAQ control the seqence enable/disable
            pvSeqMask = Pv(f'{args.pv}:PART:0:SeqMask')
            pvSeqMask.put(1<<args.eng)

    if not args.test and campv is not None:
        #  Enable the camera now that all triggers and settings are config'd
        epics.caput(campv+'Acquire',1)                # start

if __name__ == '__main__':
    main()
