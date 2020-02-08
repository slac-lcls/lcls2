#!/usr/bin/env python

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("experiment", help="psana experiment string (e.g. 'xppd7114')")
parser.add_argument("run", type=int, help="run number")
parser.add_argument('--mode', nargs='?', const='idx', default='idx')
parser.add_argument('--max_shots', nargs='?', const=200, type=int, default=200)
parser.add_argument('--num_bunches', nargs='?', const=1, type=int, default=1)
#parser.add_argument('--validity_range', nargs='?', const=None, type=tuple, default=None)
parser.add_argument('--snr_filter', nargs='?', const=10, type=int, default=10)
parser.add_argument('--roi_expand', nargs='?', const=1.0, type=float, default=1.0)
args = parser.parse_args()

import psana
from xtcav2.LasingOnCharacterization import *
import numpy as np

data_source = psana.DataSource("exp=%s:run=%s:%s" % (args.experiment, str(args.run), args.mode))
XTCAVRetrieval=LasingOnCharacterization() 

n_r=0  #Counter for the total number of xtcav images processed within the run

def processImage():
    t, power = XTCAVRetrieval.xRayPower()  
    agreement = XTCAVRetrieval.reconstructionAgreement()
    pulse = XTCAVRetrieval.pulseDelay()
    print 'Agreement: %g%%; Maximum power: %g; GW Pulse Delay: %g ' %(agreement*100,np.amax(power), pulse[0])
    

if args.mode == 'idx':
    run = data_source.runs().next()
    times = run.times()
    for t in times: 
        evt = run.event(t)
        if not XTCAVRetrieval.processEvent(evt):
            continue
        processImage()
        n_r += 1
        if n_r>=args.max_shots: 
            break

elif args.mode == 'smd':
    for evt in data_source.events():
        if not XTCAVRetrieval.processEvent(evt):
            continue
        processImage()
        n_r += 1
        if n_r>=args.max_shots: 
            break

else:
    print "Mode not supported"


