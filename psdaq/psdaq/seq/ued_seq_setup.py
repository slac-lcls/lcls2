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

EXPOSURE_TRIGGER_DELTA = 26 # Excess over camera exposure setting (360Hz)
                            # Read this from the camera
READOUT_TRIGGER_DELTA  = -4 # Time before/after exposure setting (360Hz)

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
    parser.add_argument("--test", action='store_true', help="test only")
    parser.add_argument("--verbose", action='store_true', help="verbose printout")
    args = parser.parse_args()

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
        path = f'{os.path.dirname(os.path.realpath(__file__))}/ued_360Hz.py'
        cmd = ["seqprogram","--pv", args.pv,"--seq", f'{args.eng}:{path}', '--start','--reset']
        print(cmd)
        if not args.test:
            result = subprocess.run(cmd)
            #  Remove DAQ control over seqence enable/disable
            pvSeqMask = Pv(f'{args.pv}:PART:0:SeqMask')
            pvSeqMask.put(0)
    else:
#        exp = int(360*args.period+EXPOSURE_TRIGGER_DELTA+0.5)
        rot = int(360*args.period+READOUT_TRIGGER_DELTA +0.5)
        exp = int(360*exp_period)+1

        if exp==0:
            raise ValueError(f'Input period {args.period} sec is less than 1/360Hz')

        print(f'** Exposure interval is {exp} at 360Hz = {exp/360.} sec **')
        print(f'** Readout delay     is {rot} at 360Hz = {rot/360.} sec **')

        fd, path = tempfile.mkstemp()
        with open(fd,'w') as f:

            def wait(intv):
                if intv <= Instruction.maxocc:
                    f.write(f'instrset.append( ACRateSync( 63, \"60H\", occ={intv} ) )\n')
                else:
                    n   = int(intv/Instruction.maxocc)
                    if n > Instruction.maxocc:
                        raise ValueError(f'Period {args.period} sec is too large.')
                    rem = intv%Instruction.maxocc
                    f.write(f'instrset.append( ACRateSync( 63, \"60H\", occ={Instruction.maxocc} ) )\n')
                    if n>1:
                        f.write(f'instrset.append( Branch.conditional( 2, 0, {n-1} ) )\n')
                    f.write(f'instrset.append( ACRateSync( 63, \"60H\", occ={rem} ) )\n')

            seqcodes = {0: f'Readout', 1:'Exposure Start'}
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
