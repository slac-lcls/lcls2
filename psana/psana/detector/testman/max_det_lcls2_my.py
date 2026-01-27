#!/usr/bin/env python

"""copy of /sdf/home/c/cpo/ipsana/max_det_lcls2.py"""
import sys
SCRNAME = sys.argv[0].rsplit('/')[-1]
CMD = """mpirun -n 5 python %s -e mfx101332224 -r 7 -d epix100 -n 100""" % SCRNAME

from psana import DataSource
import numpy as np
import os

os.environ['PS_SRV_NODES']='1'

from psana.detector.NDArrUtils import info_ndarr
import psutil
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
cpu_num = psutil.Process().cpu_num()
s_rsc = 'rank:%03d/%03d cpu:%03d' % (rank, size, cpu_num)

class Stats:
    def __init__(self,detarr):
        self.sum=detarr.astype(np.float64)
        self.sumsq=detarr.astype(np.float64)*detarr.astype(np.float64)
        self.maximum=detarr.astype(np.float64)
        self.nevent=1
    def update(self,detarr):
        self.sum+=detarr
        self.sumsq+=detarr.astype(np.float64)*detarr.astype(np.float64)
        self.maximum=np.maximum(self.maximum,detarr)
        self.nevent+=1

import argparse
parser = argparse.ArgumentParser(description='test lcls2 psana with MPI', usage=CMD)
parser.add_argument("-e", "--exp",  default='mfx101332224', type=str,  help="psana experiment (e.g. mfx101332224)")
parser.add_argument("-r", "--run",  default=7,              type=int,  help="psana run number (e.g. 7)")
parser.add_argument("-d", "--det",  default='epix100',      type=str,  help="detector name (e.g. epix100)")
parser.add_argument("-n", "--nevt", default=100,            type=int,  help="number of events")
args = parser.parse_args()

# this batch_size parameter may need to be tweaked
ds = DataSource(exp=args.exp, run=args.run, max_events=args.nevt, batch_size=2)
smd = ds.smalldata()
myrun = next(ds.runs())
det = myrun.Detector(args.det)

for nevt,evt in enumerate(myrun.events()):
    raw = det.raw.raw(evt)
    print(info_ndarr(raw, '%s evt:%03d raw:' % (s_rsc, nevt)))#, first=1000, last=1005))
    if raw is None: continue
    if nevt==0: stats = Stats(raw)
    else: stats.update(raw)

if smd.summary:
    totevent = smd.sum(stats.nevent)
    sum = smd.sum(stats.sum)
    sumsq = smd.sum(stats.sumsq)
    maximum = smd.max(stats.maximum)
    # this "if" statement picks out the mpi rank on which the sums are stored
    if totevent is not None: print(totevent,info_ndarr(sum,'\nsum  :'),info_ndarr(sumsq,'\nsumsq:', last=4))
smd.done()
