#!/usr/bin/env python

import sys
import argparse

usage = '\nE.g. : %s amox23616 137' % sys.argv[0].rsplit('/')[-1]\
      + '\n  or : %s amox23616 137 --max_shots 200 --num_bunches 1\n' % sys.argv[0].rsplit('/')[-1]
print(usage)

parser = argparse.ArgumentParser()
parser.add_argument('experiment', help='psana experiment string (e.g. "amox23616")')
parser.add_argument('run', type=int, help='run number')
parser.add_argument('--mode', nargs='?', const='idx', default='idx')
parser.add_argument('--max_shots', nargs='?', const=200, type=int, default=200)
parser.add_argument('--num_bunches', nargs='?', const=1, type=int, default=1)
#parser.add_argument('--validity_range', nargs='?', const=None, type=tuple, default=None)
parser.add_argument('--snr_filter', nargs='?', const=10, type=int, default=10)
parser.add_argument('--roi_expand', nargs='?', const=1.0, type=float, default=1.0)
parser.add_argument('-f', '--fname', type=str, default='/reg/g/psdm/detector/data2_test/xtc/data-amox23616-r0137-e000100-xtcav-v2.xtc2', help='xtc2 file')
args = parser.parse_args()

#================================
#sys.exit('TEST EXIT')
#================================

import numpy as np

from psana import DataSource
from psana.xtcav.LasingOnCharacterization import *

XTCAVRetrieval=LasingOnCharacterization() 

def processImage():
    t, power = XTCAVRetrieval.xRayPower()  
    agreement = XTCAVRetrieval.reconstructionAgreement()
    pulse = XTCAVRetrieval.pulseDelay()
    print('Agreement: %g%%; Maximum power: %g; GW Pulse Delay: %g '%\
          (agreement*100,np.amax(power), pulse[0]))

ds = DataSource(files=args.fname)
run = next(ds.runs())

nimgs=0

for nev,evt in enumerate(run.events()):
    logger.info('Event %03d'%nev)
    #img = camraw(evt)
    #if img is None: continue

    #=======================
    continue 
    #=======================    

    if not XTCAVRetrieval.processEvent(evt):
        continue
    processImage()
    nimgs += 1
    if nimgs>=args.max_shots: 
        break

#----------
sys.exit('END OF xtcavLasingOn')
#----------
