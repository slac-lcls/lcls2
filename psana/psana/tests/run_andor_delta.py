
from psana import DataSource
import os, sys, time
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

os.environ['PS_SMD_N_EVENTS']='1'


# passing exp and runnum
exp=sys.argv[1]
runnum=int(sys.argv[2])


# Unit-test: Run mpirun -n 4 python run_andor_delta.py rixc00221 49
# Known answer for manual unit test for rixc00221 r49
# The key is the andor timestamp + delay and the tuple values
# are no. of andor events and no. of integrating events
known_answers = {4565831569331579254: (109, 35077),
        4565831582156491780: (197, 64002),
        4565831594981407590: (197, 64002),
        4565831607806321605: (197, 64002),
        4565831620631237217: (197, 64002),}

def check_answer(ts, cn_atm_events, cn_intg_events):
    if exp == 'rixc00221' and runnum == 49:
        assert (cn_atm_events, cn_intg_events) == known_answers[ts]



# set intg_delta_t
intg_delta_t = 107800000 

mount_dir = '/sdf/data/lcls/drpsrcf/ffb'
#mount_dir = '/cds/data/drpsrcf'
xtc_dir = os.path.join(mount_dir, exp[:3], exp, 'xtc')
ds = DataSource(exp=exp,run=runnum,dir=xtc_dir,intg_det='andor_vls',intg_delta_t=intg_delta_t,
        detectors=['timing','andor_vls','atmopal'],
        max_events=0,
        live=False)

cn_intg_events = 0
cn_atm_events = 0
cn_andor_events = 0
andor_current_ts = 0
for myrun in ds.runs():
    andor = myrun.Detector('andor_vls')
    atmopal = myrun.Detector('atmopal')
    for nstep, mystep in enumerate(myrun.steps()):
        #print(f'BD:{rank} step: {nstep}')
        for nevt,evt in enumerate(mystep.events()):
            cn_intg_events += 1
            andor_img = andor.raw.value(evt)
            atmopal_img = atmopal.raw.image(evt)
            if atmopal_img is not None:
                cn_atm_events += 1
            
            if andor_img is not None:
                andor_current_ts = evt.timestamp
                cn_andor_events += 1
            
            # Check that no. of atm events are as expected with intg_delta_t
            if evt.EndOfBatch():
                delta_ns = evt.timestamp_diff(andor_current_ts)
                txt = f'BD:{rank} ts: {evt.timestamp} #atm: {cn_atm_events} #andor: {cn_andor_events} #intg: {cn_intg_events} delta_ns:{delta_ns}'
                check_answer(evt.timestamp, cn_atm_events, cn_intg_events)
                print(txt)
                cn_atm_events = 0
                cn_intg_events = 0
    print(f'BD:{rank} #atm: {cn_atm_events} #andor: {cn_andor_events} #intg: {cn_intg_events}')
        
