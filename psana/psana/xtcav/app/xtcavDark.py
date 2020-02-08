#!/usr/bin/env python
#print("XXXXXXXXXXXX In xtcavDark")
if True :
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", help="psana experiment string (e.g. 'xppd7114')")
    parser.add_argument("run", type=int, help="run number")
    parser.add_argument('--max_shots', nargs='?', const=400, type=int, default=400)
    #parser.add_argument('--validity_range', nargs='?', const=None, type=tuple, default=None)
    args = parser.parse_args()

    from psana.xtcav.DarkBackgroundReference import *

    dark_background = DarkBackgroundReference(
        experiment=args.experiment, 
        run_number=args.run, 
        max_shots=args.max_shots)
