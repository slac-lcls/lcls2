#!/usr/bin/env python

usage = './xtcavDark.py amox23616 104'
print(usage)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, default='amox23616', help="psana experiment string (e.g. 'amox23616')")
parser.add_argument('--run', type=int, default=104, help="run number") # 104
parser.add_argument('-f', '--fname', type=str, default='/reg/g/psdm/detector/data2_test/xtc/data-amox23616-r0104-e000400-xtcav.xtc2', help='xtc2 file')
parser.add_argument('--max_shots', nargs='?', const=400, type=int, default=400)
args = parser.parse_args()
print('parser.parse_args()', args)

from psana.xtcav.DarkBackgroundReference import DarkBackgroundReference

dark_background = DarkBackgroundReference(
    experiment=args.experiment, 
    run_number=args.run, 
    max_shots=args.max_shots,
    fname=args.fname)
