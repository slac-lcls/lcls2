from psana2 import DataSource
import numpy as np
import os
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def run_fakestep():
    # Set SMD0 and EB batch size
    os.environ['PS_SMD_N_EVENTS'] = '5'
    os.environ['PS_FAKESTEP_FLAG'] = '1'
    PS_EB_NODES = 2
    os.environ['PS_EB_NODES'] = str(PS_EB_NODES)
    batch_size = 4

    xtc_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_data', 'fakesteps')
    ds = DataSource(exp='xpptut15', run=14, dir=xtc_dir, batch_size=batch_size)
    cn_steps = 0
    cn_events = 0
    result = {'evt_per_step':[], 'n_steps': 0, 'n_events':0}
    for r, run in enumerate(ds.runs()):
        sdet = run.Detector('motor1')
        hsd = run.Detector('hsd')
        andor = run.Detector('andor')
        for i, step in enumerate(run.steps()):
            cn_evt_per_step = 0
            for j, evt in enumerate(step.events()):
                hsd_calib = hsd.raw.calib(evt)
                andor_calib = andor.raw.calib(evt)
                cn_evt_per_step += 1
                cn_events += 1
            cn_steps +=1
            result['evt_per_step'].append(cn_evt_per_step)

    result['n_steps'] = cn_steps
    result['n_events'] = cn_events
    if rank > PS_EB_NODES or size == 1:
        print(f'rank:{rank} #steps:{cn_steps} #events:{cn_events} #event_per_step:{result}')
    return result

if __name__ == "__main__":
    result = run_fakestep()
    result = comm.gather(result, root=0)
    if rank == 0:
        sum_events_per_step = np.zeros(4, dtype=np.int32)
        sum_events = 0
        n_steps = 0
        for i in range(size):
            if result[i]['evt_per_step']:
                sum_events_per_step += np.asarray(result[i]['evt_per_step'], dtype=np.int32)
            sum_events += result[i]['n_events']
            n_steps = np.max([n_steps, result[i]['n_steps']])

        print(f'sum_events_per_step:{sum_events_per_step} sum_events:{sum_events} sum_steps:{n_steps}')
        assert all(sum_events_per_step == [3,2,4,1])
        assert sum_events == 10
        assert n_steps == 4
