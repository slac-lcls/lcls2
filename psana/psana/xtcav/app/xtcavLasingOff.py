#!/usr/bin/env python

import sys
import argparse
import logging
logger = logging.getLogger(__name__)
from psana.pyalgos.generic.Utils import init_logger, STR_LEVEL_NAMES

scrname = sys.argv[0].rsplit('/')[-1]
usage = '\nE.g. : %s amox23616 131' % scrname\
      + '\n  or : %s amox23616 131 -l DEBUG --max_shots 200 --num_bunches 1\n' % scrname
print(usage)

d_fname = '/reg/g/psdm/detector/data2_test/xtc/data-amox23616-r0131-e000200-xtcav-v2.xtc2'

parser = argparse.ArgumentParser()
parser.add_argument('experiment', help='psana experiment string (e.g. "amox23616")')
parser.add_argument('run', type=int, help='run number')
parser.add_argument('--max_shots',   nargs='?', const=200, type=int,   default=200)
parser.add_argument('--num_bunches', nargs='?', const=1,   type=int,   default=1)
parser.add_argument('--num_groups',  nargs='?', const=12,  type=int,   default=12)
parser.add_argument('--snr_filter',  nargs='?', const=10,  type=int,   default=10)
parser.add_argument('--roi_expand',  nargs='?', const=1.0, type=float, default=1.0)
parser.add_argument('-f', '--fname', type=str, default=d_fname, help='xtc2 file')
parser.add_argument('-p', '--plot_image', type=bool, default=False, help='plot events')
parser.add_argument('-l', '--loglev', default='INFO', type=str, help='logging level name, one of %s' % STR_LEVEL_NAMES)

args = parser.parse_args()
print('Arguments: %s\n' % str(args))
for k,v in vars(args).items() : print('  %12s : %s' % (k, str(v)))

init_logger(args.loglev, fmt='[%(levelname).1s] L%(lineno)04d : %(message)s', datefmt='%Y-%m-%dT%H:%M:%S')

from psana.xtcav.LasingOffReference import LasingOffReference
LasingOffReference(args)

sys.exit('END OF %s' % scrname)

#----------
