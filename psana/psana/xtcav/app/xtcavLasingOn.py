#!/usr/bin/env python

import sys
import argparse

scrname = sys.argv[0].rsplit('/')[-1]

usage = '\nE.g. : %s amox23616 137' % scrname\
      + '\n  or : %s amox23616 137 --max_shots 200 -f fname.xtc2\n' % scrname
print(usage)

parser = argparse.ArgumentParser()
parser.add_argument('experiment', help='psana experiment string (e.g. "amox23616")')
parser.add_argument('run', type=int, help='run number')
parser.add_argument('--max_shots', nargs='?', const=200, type=int, default=200)
parser.add_argument('--num_bunches', nargs='?', const=1, type=int, default=1)
parser.add_argument('-f', '--fname', type=str, default=\
  '/reg/g/psdm/detector/data2_test/xtc/data-amox23616-r0137-e000100-xtcav-v2.xtc2', help='xtc2 file')
#parser.add_argument('--mode', nargs='?', const='idx', default='idx')
#parser.add_argument('--validity_range', nargs='?', const=None, type=tuple, default=None)
#parser.add_argument('--snr_filter', nargs='?', const=10, type=int, default=10)
#parser.add_argument('--roi_expand', nargs='?', const=1.0, type=float, default=1.0)

args = parser.parse_args()
print('parser.parse_args()', args)

from psana.xtcav.LasingOnCharacterization import LasingOnCharacterization

XTCAVRetrieval=LasingOnCharacterization(args) 

sys.exit('END OF %s' % scrname)

#----------
