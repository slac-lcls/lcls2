#!/usr/bin/env python

import sys

import logging
logger = logging.getLogger(__name__)
from psana.pyalgos.generic.Utils import init_logger, STR_LEVEL_NAMES

scrname = sys.argv[0].rsplit('/')[-1]
usage = 'E.g.: %s amox23616 104' % scrname\
      + '\n  or: %s amox23616 104 -f fname.xtc2' % scrname
print(usage)

d_fname = '/reg/g/psdm/detector/data2_test/xtc/data-amox23616-r0104-e000400-xtcav-v2.xtc2'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('experiment',    type=str, help="experiment name (e.g. 'amox23616')")
parser.add_argument('run',           type=int, help="run number") # 104
parser.add_argument('--max_shots',   type=int, default=400, help='number of events')
parser.add_argument('-f', '--fname', type=str, default=d_fname, help='xtc2 file')
parser.add_argument('-l', '--loglev', default='DEBUG', type=str, help='logging level name, one of %s' % STR_LEVEL_NAMES)

args = parser.parse_args()
print('Arguments: %s\n' % str(args))
for k,v in vars(args).items() : print('  %12s : %s' % (k, str(v)))

init_logger(args.loglev, fmt='[%(levelname).1s] L%(lineno)04d : %(message)s', datefmt='%Y-%m-%dT%H:%M:%S')

from psana.xtcav.DarkBackgroundReference import DarkBackgroundReference
DarkBackgroundReference(args)

sys.exit('END OF %s' % scrname)

#----------
