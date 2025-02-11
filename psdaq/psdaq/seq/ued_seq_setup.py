from psdaq.seq.periodicgenerator import PeriodicGenerator
import argparse
import os
import tempfile
import subprocess

def main():
    parser = argparse.ArgumentParser(description='ued sequencer programming')
    parser.add_argument('--pv', type=str, default='DAQ:UED:XPM:0', help="sequence engine pv; default DAQ:UED:XPM:0")
    parser.add_argument("--eng", type=int, default=0, help="sequence engine; default=0")
    parser.add_argument("--period", type=float, default=None, help="period (sec); defaults to 360Hz virtual TS1")
    args = parser.parse_args()

    if args.period is None:
        #  Program the sequence
        path = f'{os.path.dirname(os.path.realpath(__file__))}/ued_360Hz.py'
        cmd = ["seqprogram","--pv", args.pv,"--seq", f'{args.eng}:{path}', '--start']
        print(cmd)
        result = subprocess.run(cmd)

    else:
        #  Use the periodic generator to create a sequence with two event codes and the specified interval
        per = int(360*args.period)
        gen = PeriodicGenerator(period=[per,per],start=[1,1], marker='a60Ht123456')

        fd, path = tempfile.mkstemp()
        with open(fd,'w') as f:
            seqcodes = {0: f'{args.period} sec', 1:'first'}
            f.write(f'seqcodes = {seqcodes}\n')
            #  Modify the sequence to only generate the first eventcode after the 1st iteration
            for i in gen.instr[:-1]:
                f.write(f'{i}\n')
            f.write('instrset.append( ControlRequest([0]) )\n')
            f.write('instrset.append( Branch.unconditional(2) )\n')
            f.close()

        #  Program the sequence
        cmd = ["seqprogram","--pv", args.pv,"--seq", f'{args.eng}:{path}', '--start']
        print(cmd)
        result = subprocess.run(cmd)

        os.remove(path)

if __name__ == '__main__':
    main()
