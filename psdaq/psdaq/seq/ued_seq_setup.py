import argparse
import os
import tempfile
import subprocess
from psdaq.seq.seq import Instruction

def main():
    parser = argparse.ArgumentParser(description='ued sequencer programming')
    parser.add_argument('--pv', type=str, default='DAQ:UED:XPM:0', help="sequence engine pv; default DAQ:UED:XPM:0")
    parser.add_argument("--eng", type=int, default=0, help="sequence engine; default=0")
    parser.add_argument("--period", type=float, default=None, help="period (sec)")
    parser.add_argument("--verbose", action='store_true', help="verbose printout")
    args = parser.parse_args()

    if args.period is None:
        #  Program the sequence
        path = f'{os.path.dirname(os.path.realpath(__file__))}/ued_360Hz.py'
        cmd = ["seqprogram","--pv", args.pv,"--seq", f'{args.eng}:{path}', '--start']
        print(cmd)
        result = subprocess.run(cmd)

    else:
        per = int(360*args.period+0.5)

        if per==0:
            raise ValueError(f'Input period {args.period} sec is less than 1/360Hz')
        else:
            print(f'** Interval is {per} at 360Hz = {per/360.} sec **')

        fd, path = tempfile.mkstemp()
        with open(fd,'w') as f:
            seqcodes = {0: f'{args.period} sec', 1:'first'}
            f.write(f'seqcodes = {seqcodes}\n')
            f.write(f'instrset = []\n')
            f.write(f'instrset.append( ACRateSync( 63, \"60H\", occ=1 ) )\n')
            f.write(f'instrset.append( ControlRequest([0,1]) )\n')
            if per <= Instruction.maxocc:
                f.write(f'instrset.append( ACRateSync( 63, \"60H\", occ={per} ) )\n')
            else:
                n   = int(per/Instruction.maxocc)
                if n > Instruction.maxocc:
                    raise ValueError(f'Period {args.period} sec is too large.')
                rem = per%Instruction.maxocc
                f.write(f'instrset.append( ACRateSync( 63, \"60H\", occ={Instruction.maxocc} ) )\n')
                if n>1:
                    f.write(f'instrset.append( Branch.conditional( 2, 0, {n-1} ) )\n')
                f.write(f'instrset.append( ACRateSync( 63, \"60H\", occ={rem} ) )\n')
            f.write(f'instrset.append( ControlRequest([0]) )\n')
            f.write(f'instrset.append( Branch.unconditional(2) )\n')
            f.close()

        #  Program the sequence
        cmd = ["seqprogram","--pv", args.pv,"--seq", f'{args.eng}:{path}', '--start']
        if args.verbose:
            cmd.append("--verbose")

        result = subprocess.run(cmd)

        os.remove(path)

if __name__ == '__main__':
    main()
