"""

"""
from psdaq.seq.globals import *
from psdaq.seq.seq import *
import argparse
import logging

args = None

#
#  Write the output file and check its validity
#
def write_seq(instr, seqcodes, filename):

    if (len(instr) > 2000):
        sys.stderr.write(f'*** Sequence has {len(instr)} instructions.  May be too large to load. ***\n')

    with open(filename,"w") as f:
        f.write('# {} instructions\n'.format(len(instr)))

        f.write('\n')
        f.write('seqcodes = {}\n'.format(seqcodes))
        f.write('\n')
        for i in instr:
            f.write('{}\n'.format(i))

    validate(filename)
    print(f'** Wrote {filename} **')

#
#  Generate a sequence engine with the appropriate structure
#
def generate_engine(engine):
    global_sync = 'FixedRateSync("1H", 1)'
#    motion_step = f'FixedRateSync("910kH",{args.period})'
    motion_step = f'Wait("910kH",{args.period})'

    instr = ['instrset = []']
    instr.append(f'instrset.append( {global_sync} )')

    instr.append(f'anchor = len(instrset)')

    for i in range(args.count):
        v = (i*2+1)>>(engine*4)
        instr.append(f'instrset.append( ControlRequest({v}) )')
        instr.append(f'instrset.append( {motion_step} )')

    instr.append(f'instrset.append( Branch.unconditional(anchor) )')

    #  Write it
    seqcodes = {i:'Base' if engine==0 and i==0 else f'C{4*engine+i-1}' for i in range(4)}
    write_seq(instr, seqcodes, f'{args.path}/counter{engine}.py')

def main():
    global args
    parser = argparse.ArgumentParser(description="counter sequencer",formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--period", default=1, type=int, help='Period in 929kHz steps')
    parser.add_argument("--count", default=31, type=int, help='Maximum count')
    parser.add_argument("--path", default='.', help='Output file path')
    args = parser.parse_args()

    generate_engine(0)
    count = args.count >> 3
    i = 1
    while count:
        generate_engine(i)
        i += 1
        count >>= 4

if __name__ == '__main__':
    main()
