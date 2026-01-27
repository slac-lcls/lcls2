from psana import DataSource
import numpy as np
import os

os.environ['PS_SRV_NODES']='1'

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
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--exp", help="psana experiment (e.g. xppd7114)")
parser.add_argument("-r", "--run", help="psana run number (e.g. 43)", type=int)
parser.add_argument("-d", "--det", help="detector name (e.g. epixquad)")
parser.add_argument("-n", "--nevt", help="number of events", default=1e8, type=int)
args = parser.parse_args()

# this batch_size parameter may need to be tweaked
ds = DataSource(exp=args.exp, run=args.run, max_events=args.nevt, batch_size=2)
smd = ds.smalldata()
myrun = next(ds.runs())
det = myrun.Detector(args.det)

for nevt,evt in enumerate(myrun.events()):
    raw = det.raw.raw(evt)
    if raw is None: continue
    if nevt==0: stats = Stats(raw)
    else: stats.update(raw)

if smd.summary:
    totevent = smd.sum(stats.nevent)
    sum = smd.sum(stats.sum)
    sumsq = smd.sum(stats.sumsq)
    maximum = smd.max(stats.maximum)
    # this "if" statement picks out the mpi rank on which the sums are stored
    if totevent is not None: print(totevent,sum,sumsq)
smd.done()
