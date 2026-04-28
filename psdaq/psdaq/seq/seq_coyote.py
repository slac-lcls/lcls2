"""
   This script creates a model set of sequencing instructions for the MFX Coyote experiment
   The sequencer instructions used are:
     ACRateSync( <timeslot mask>, <rate marker>, <num occurrences> )
     ControlRequest( <list of codes> )
     Branch.conditional( <instruction number>, <conditional counter>, <num iterations> )

   The instructions must obey the hardware constraints on <num occurrences>, <num_iterations>, and <conditional counter>.
   The output of this script is a python script itself for each sequence engine that can be 
   compiled by either the prorammer (seqprogram) or the display (seqplot).

   The motion scan looks something like this for
  --start_steps=4
  --stop_steps=5
  --samples_per_row=7
  --steps_btw_rows=6
  --steps_btw_windows=3
  --rows_per_window=4
  --windows_per_line=2
  --lines_of_windows=2
  -- steps_btw_lines=8
  -- codes=98 126 6 9

s = start position, S = stop position, [./\]= motion checkpoint, x = exposure

 s . . . . x x x x x x x . . . x x x x x x x . . .  (left to right)
     . . . x x x x x x x . . . x x x x x x x . . /  (right to left)
     \ . . x x x x x x x . . . x x x x x x x . . .  (left to right)
     . . . x x x x x x x . . . x x x x x x x . . / 
    .
    .
     . . . x x x x x x x . . . x x x x x x x . . .  (left to right)
     . . . x x x x x x x . . . x x x x x x x . . /  (right to left)
     \ . . x x x x x x x . . . x x x x x x x . . .  (left to right)
     . . . x x x x x x x . . . x x x x x x x . . / 
    .
  S
"""
from psdaq.seq.globals import *
from psdaq.seq.seq import *
import argparse
import logging

args = None

#
#  A single loop is limited to Instruction.maxocc iterations
#  Check how many levels of looping are required
#
def loop_levels(cycles, level_limit=4):
    nlevels = 0
    lcycles = cycles
    while lcycles:
        nlevels  += 1
        lcycles //= (Instruction.maxocc+1)
    if nlevels==0:  # Nothing to do
        return
    if nlevels>level_limit:
        raise RuntimeError(f'Loop of {cycles} exceeds hardware capability')
    return nlevels

#
#  Make a generic loop of event code requests and time steps
#
def loop(instr, codes_list, step, cycles, level_limit=4):
    nlevels = loop_levels(cycles, level_limit)
    instr.append(f'line = len(instrset)')                             # Set the anchor for the loop
    instr.append(f'instrset.append( ControlRequest({codes_list}) )')  # Generate the event codes
    instr.append(f'instrset.append( {step} )')                        # Make the time step
    if nlevels==1:
        instr.append(f'instrset.append( Branch.conditional(line, 0, {cycles-1}) )')  # Loop back to the anchor
    else:
        niters = 1
        for cc in range(nlevels-1):
            instr.append(f'instrset.append( Branch.conditional(line, {cc}, {Instruction.maxocc}) )') # Loop back to the anchor maximum times
            niters *= (Instruction.maxocc+1)

        #  Handle the remaining cycles
        rem = cycles - niters
        loop(instr, codes_list, step, rem)

#
#  Write the output file and check its validity
#
def write_seq(instr, seqcodes, filename):

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
    #
    #  Assumptions:
    #  1.  120 Hz motion steps
    #  2.  Trigger pulse picker shutter setup is one 120 Hz period

    #  Eventcodes:
    #    Engine 0:
    #       0:  Motion step (always 120 Hz between)
    #       1:  Pulse picker trigger
    #       2:  DAQ readout
    #    Engine 1:
    #       0..3:  WindowIndex[0..3]
    #    Engine 2:
    #       0..3:  WindowIndex[4..7]
    #

    global_sync = 'ACRateSync(0x1, "10H", 1)'    #  Start all engines on the same marker, so they are in-sync
    motion_step = 'ACRateSync(0x9, "60H", 1)'    #  Always 120H period between motion triggers (60Hz timeslots 1,4)

    if args.fixed_test:
        global_sync = 'FixedRateSync("1H", 1)'
        motion_step = 'FixedRateSync("100H",1)'

    motion_codes      = [0]   if engine==0 else []
    motion_codes_last = [0,1] if engine==0 else []

    instr = ['instrset = []']
    instr.append(f'instrset.append( {global_sync} )')

    #  Startup
    loop(instr, motion_codes, motion_step, args.start_steps-1) 

    #  Loop over windows - could use a Branch.conditional for this, but it's not large, and conditional counters are scarce (4).
    for line in range(args.lines_of_windows):
        instr.append(f'# Line {line}')                  # Insert a comment for human readers

        #  Prepare for loop over rows
        nlevels = loop_levels(args.rows_per_window-1)

        def generate_rows(cycles):
            if cycles==0:
                return
            instr.append(f'rows_anchor = len(instrset)')   #  Start a loop over rows

            def one_row(forward=True):
                #  Add pulse picker trigger to last motion_step before starting exposure
                instr.append(f'instrset.append( ControlRequest({motion_codes_last}) )')
                instr.append(f'instrset.append( {motion_step} )')

                for wl in range(args.windows_per_line):
                    w = line*args.windows_per_line + (wl if forward else args.windows_per_line-wl-1)
                    expo_codes      = [0,2]   if engine==0 else (args.codes[w]&0xf) if engine==1 else (args.codes[w]>>4)&0xf
                    expo_codes_last = [0,1,2] if engine==0 else expo_codes


                    #  Exposures (minus one)
                    loop(instr, expo_codes, motion_step, args.samples_per_row-1, 4-nlevels)  # the loop over rows uses <nlevels> conditional counters
                    #  Add pulse picker trigger to last motion_step before ending exposure
                    instr.append(f'instrset.append( ControlRequest({expo_codes_last}) )')
                    instr.append(f'instrset.append( {motion_step} )')
                    #  Steps between windows
                    if wl < args.windows_per_line-1:
                        loop(instr, motion_codes, motion_step, args.steps_btw_windows-1, 4-nlevels)
                        #  Add pulse picker trigger to last motion_step before starting exposure
                        instr.append(f'instrset.append( ControlRequest({motion_codes_last}) )')
                        instr.append(f'instrset.append( {motion_step} )')

                #  Steps between rows
                loop(instr, motion_codes, motion_step, args.steps_btw_rows-1, 4-nlevels)

            one_row(True)
            one_row(False)

            #  Loop back for next row
            if nlevels==1:
                instr.append(f'instrset.append( Branch.conditional(rows_anchor, 3, {cycles-1}) )')  # Loop back to the anchor
            else:
                niters = 1
                for cc in range(3,3-nlevels,-1):
                    instr.append(f'instrset.append( Branch.conditional(rows_anchor, {cc}, {Instruction.maxocc}) )') # Loop back to the anchor maximum times
                    niters *= (Instruction.maxocc+1)

                #  Handle the remaining cycles
                #  This effectively doubles (or more) the number of instructions when 
                #    the number of rows exceeds 4095.  It could be 
                #    reduced using the call/return instruction.
                rem = args.rows_per_window - niters
                generate_rows(rem)
                
        generate_rows(args.rows_per_window//2)

        if line < args.lines_of_windows-1:
            #  Move to the next line of windows
            loop(instr, motion_codes, motion_step, args.steps_btw_lines-1) 

    #  Stopping
    loop(instr, motion_codes, motion_step, args.stop_steps)

    #  Terminate
    instr.append('instrset.append( CheckPoint() )')
    instr.append('line = len(instrset)')
    instr.append('instrset.append( Branch.unconditional(line) )')

    #  Write it
    seqcodes = {0:'Motion', 1:'PulsePicker', 2:'DAQ'} if engine==0 else {i:f'Window{i}' for i in range(4)} if engine==1 else {i:f'Window{i+4}' for i in range(4)}
    write_seq(instr, seqcodes, f'{args.path}/engine{engine}.py')

def main():
    global args
    parser = argparse.ArgumentParser(description='coyote experiment sequencer',formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--start_steps", default=7, type=int, help='Motion steps before the first exposure')
    parser.add_argument("--stop_steps", default=7, type=int, help='Motion steps after the last exposure of last window')
    parser.add_argument("--samples_per_row", default=1000, type=int, help='Samples to step and expose for each row')
    parser.add_argument("--steps_btw_windows", default=10, type=int, help='Motion steps between exposures on adjacent windows')
    parser.add_argument("--steps_btw_rows", default=4, type=int, help='Motion steps between exposures on adjacent rows')
    parser.add_argument("--rows_per_window", default=100, type=int, help='Rows in each sample window')
    parser.add_argument("--steps_btw_lines", default=10, type=int, help='Motion steps between exposures on adjacent windows')
    parser.add_argument("--windows_per_line", default=3, type=int, help='Number of windows in a line')
    parser.add_argument("--lines_of_windows", default=2, type=int, help='Number of lines of windows')
    parser.add_argument("--codes", nargs='+', default=[i for i in range(1,5)], type=int, help='Event codes by window (left to right, top to bottom)')
    parser.add_argument("--fixed_test", action='store_true', help='Test with fixed rates')
    parser.add_argument("--path", default='.', help='Output file path')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    #  Enforcement
    if args.rows_per_window%2:
        raise ValueError(f'rows_per_window ({args.rows_per_window}) must be even')

    if len(args.codes) != args.windows_per_line * args.lines_of_windows:
        raise ValueError(f'length of codes list ({len(args.codes)}) does not match product of windows_per_line ({args.windows_per_line}) and lines_of_windows ({args.lines_of_windows})')

    generate_engine(0)
    generate_engine(1)
    generate_engine(2)

if __name__ == '__main__':
    main()
