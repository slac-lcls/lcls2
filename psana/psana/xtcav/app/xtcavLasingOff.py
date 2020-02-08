#!/usr/bin/env python

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("experiment", help="psana experiment string (e.g. 'xppd7114')")
parser.add_argument("run", type=int, help="run number")
parser.add_argument('--max_shots', nargs='?', const=400, type=int, default=400)
parser.add_argument('--num_bunches', nargs='?', const=1, type=int, default=1)
#parser.add_argument('--validity_range', nargs='?', const=None, type=tuple, default=None)
parser.add_argument('--num_groups', nargs='?', const=12, type=int, default=12)
parser.add_argument('--snr_filter', nargs='?', const=10, type=int, default=10)
parser.add_argument('--roi_expand', nargs='?', const=1.0, type=float, default=1.0)
args = parser.parse_args()

from xtcav2.LasingOffReference import LasingOffReference

lor = LasingOffReference(
    experiment=args.experiment, 
    run_number=args.run, 
    max_shots=args.max_shots,
    num_bunches=args.num_bunches,
    num_groups=args.num_groups,        
    snr_filter=args.snr_filter,           
    roi_expand=args.roi_expand)
