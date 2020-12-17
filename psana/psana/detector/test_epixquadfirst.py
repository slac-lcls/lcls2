from psana import DataSource
from cfg_utils import *
import numpy as np
import scipy as sp
import argparse
import time
import math

parser = argparse.ArgumentParser()
parser.add_argument('--expt', help='experiment name', default='ueddaq02')
parser.add_argument("--detname", type=str, default='epixquad', help='detector name; like tmohsd')
parser.add_argument("--nevents", type=int, default=1024, help='nevents')
parser.add_argument("--run", type=int, default=127, help='run')
args = parser.parse_args()

#plotPeriod = 0.5
plotPeriod = 10
#ds = DataSource(shmem='tst')
#ds = DataSource(shmem='tstP0')
#ds = DataSource(exp='tmoc00118',run=186,dir='/cds/data/psdm/tmo/tmoc00118/xtc')
#if args.expt=='tst':
#    ds = DataSource(exp='tstx00117',run=args.run,dir='/u2/lcls2/tst/tstx00117/xtc#')
#else:
#    ds = DataSource(exp='ueddaq02',run=args.run,dir='/u2/pcds/pds/ued/ueddaq02/xt#c')
ds = DataSource(exp=args.expt, run=args.run, dir=f'/cds/data/psdm/{args.expt[:3]}/{args.expt}/xtc')

from psmon import publish
import psmon.plots as plots
from psmon.plotting import Histogram,LinePlot,Image
publish.local=True
publish.plot_opts.aspect = 1

myrun = next(ds.runs())

det = myrun.Detector(args.detname)
#dethw = myrun.Detector(args.detname+'hw')

def dump1(arr,title,nx,start):
    print(f'{title} [{nx}]')
    s = ''
    for i in range(start,start+nx):
        s += ' {:04x}'.format(arr[i])
        if i%16==15:
            s += '\n'
    print(s)
    return start+nx

def dump2(arr):
    for i in range(arr.shape[0]):
        s = ''
        for j in range(arr.shape[1]):
            s += ' {:04x}'.format(arr[i][j])
            if j%16==15:
                s += '\n'
        print(s)

image_mgr = {}

print('--dethw--')
#dump_det_config(det,args.detname+'hw')

for nstep,step in enumerate(myrun.steps()):

    print('--step {}--'.format(nstep))
    dump_det_config(det,args.detname)

    avgimg = None
    for nevt,evt in enumerate(step.events()):
        image = det.raw.array(evt).astype(np.float)
        if avgimg is None:
            avgimg = image.copy()
        else:
            avgimg += image

    print(f'nevt {nevt}')

    avgimg = avgimg/(nevt+1)

    pedestal = np.savetxt( f'avgimg_r{args.run}_{nstep}.txt',avgimg)

    dimg = avgimg.copy()
    image_mgr[nstep] = Image('epixquad step %d'%nstep, xlabel='x', ylabel='y')
    image_mgr[nstep].image(dimg)
    image_mgr[nstep].publish()
