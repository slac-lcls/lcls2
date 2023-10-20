from psana import DataSource
from psmon import publish
from psmon.plots import Image,XYPlot
import os, sys, time
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
 

os.environ['PS_SRV_NODES']='1'
os.environ['PS_SMD_N_EVENTS']='1'


# passing exp and runnum
exp=sys.argv[1]
runnum=int(sys.argv[2])


mount_dir = '/sdf/data/lcls/drpsrcf/ffb'
#mount_dir = '/cds/data/drpsrcf'
xtc_dir = os.path.join(mount_dir, exp[:3], exp, 'xtc')
ds = DataSource(exp=exp,run=runnum,dir=xtc_dir,intg_det='andor_vls',batch_size=1, 
        psmon_publish=publish)


# we will remove this for batch processing and use "psplot" instead
# publish.local = True


def my_smalldata(data_dict):
    if 'unaligned_andor_norm' in data_dict:
        andor_norm = data_dict['unaligned_andor_norm'][0]
        myplot = XYPlot(0,"Andor (normalized)",range(len(andor_norm)),andor_norm)
        publish.send('ANDOR',myplot)
 
for myrun in ds.runs():
    andor = myrun.Detector('andor_vls')
    timing = myrun.Detector('timing')
    smd = ds.smalldata(filename='mysmallh5.h5',batch_size=1000, callbacks=[my_smalldata])
    norm = 0
    ndrop_inhibit = 0
    for nstep,step in enumerate(myrun.steps()):
        print('step:',nstep)
        for nevt,evt in enumerate(step.events()):
            andor_img = andor.raw.value(evt)
            # also need to check for events missing due to damage
            # (or compare against expected number of events)
            ndrop_inhibit += timing.raw.inhibitCounts(evt)
            smd.event(evt, mydata=nevt) # high rate data saved to h5
            # need to check Matt's new timing-system data on every
            # event to make sure we haven't missed normalization
            # data due to deadtime
            norm+=nevt # fake normalization
            if andor_img is not None:
                print('andor data on evt:',nevt,'ndrop_inhibit:',ndrop_inhibit)
                # check that the high-read readout group (2) didn't
                # miss any events due to deadtime
                if ndrop_inhibit[2]!=0: print('*** data lost due to deadtime')
                # need to prefix the name with "unaligned_" so
                # the low-rate andor dataset doesn't get padded
                # to align with the high rate datasets
                smd.event(evt, mydata=nevt,
                          unaligned_andor_norm=(andor_img/norm))
                norm=0
                ndrop_inhibit=0
