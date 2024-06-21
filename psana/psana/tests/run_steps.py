"""
Test smd0 and eventbuilder for handling step dgrams.
See https://docs.google.com/spreadsheets/d/1VlVCwEVGahab3omAFJLaF8DJWFcz-faI9Q9aHa7QTUw/edit?usp=sharing for test setup.

"""
import os, time, glob, sys
from psana.dgram import Dgram
from setup_input_files import setup_input_files
from psana import DataSource
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import logging
#logging.basicConfig(level=logging.INFO, format='(%(threadName)-10s) %(message)s', )# filename="log.log", filemode="w")

test_xtc_dir = os.environ.get('TEST_XTC_DIR', '.')
xtc_dir = os.path.join(test_xtc_dir, '.tmp_smd0')

def my_filter(evt):
    return True

def run_serial_read(n_events, batch_size=1, filter_fn=0):
    exp_xtc_dir = os.path.join(xtc_dir, '.tmp')
    os.environ['PS_SMD_N_EVENTS'] = str(n_events)
    ds = DataSource(exp='xpptut13', run=1, dir=exp_xtc_dir, batch_size=batch_size, filter=filter_fn)
    cn_steps = 0
    cn_events = 0
    result = {'evt_per_step':[0,0,0], 'n_steps': 0, 'n_events':0}
    for r, run in enumerate(ds.runs()):
        edet = run.Detector('HX2:DVD:GCC:01:PMON')
        sdet = run.Detector('motor2')
        for i, step in enumerate(run.steps()):
            cn_evt_per_step = 0
            for j, evt in enumerate(step.events()):
                cn_evt_per_step += 1
                cn_events += 1
            cn_steps +=1
            result['evt_per_step'][i] = cn_evt_per_step
        
    result['n_steps'] = cn_steps
    result['n_events'] = cn_events
    print(f'rank={rank} result={result}')
    return result
    
def check_results(results, expected_result):
    for result in results:
        assert result == expected_result

def test_runsinglefile_steps():
    ds = DataSource(files=os.path.join(xtc_dir,'.tmp','data-r0001-s00.xtc2'))
    cn_steps = 0
    cn_events = 0
    result = {'evt_per_step':[0,0,0], 'n_steps': 0, 'n_events':0}
    for run in ds.runs():
        for i, step in enumerate(run.steps()):
            cn_evt_per_step = 0
            for j, evt in enumerate(step.events()):
                cn_evt_per_step += 1
                cn_events += 1
            cn_steps +=1
            result['evt_per_step'][i] = cn_evt_per_step
        
    result['n_steps'] = cn_steps
    result['n_events'] = cn_events
    return result


if __name__ == "__main__":
    import pathlib
    p = pathlib.Path(xtc_dir)
    if not p.exists():
        if rank == 0:
            p.mkdir()
            setup_input_files(p, n_files=2, slow_update_freq=4, n_motor_steps=3, n_events_per_step=10, gen_run2=False)
    
    comm.Barrier()
    
    # Test run.steps() 
    test_cases = [\
            (51, 1, 0), \
            (51, 1, my_filter), \
            (51, 5, 0), \
            (51, 5, my_filter), \
            (20, 1, 0), \
            (19, 1, 0), \
            (1, 1, my_filter), \
            (1, 1, 0), \
            (3, 4, my_filter), 
            (3, 4, 0), \
             ]
    
    for test_case in test_cases:
        result = run_serial_read(test_case[0], batch_size=test_case[1], filter_fn=test_case[2])
        result = comm.gather(result, root=0)
        if rank == 0:
            sum_events_per_step = np.zeros(3, dtype=np.int32)
            sum_events = 0
            n_steps = 0
            for i in range(size):
                if result[i]['evt_per_step']:
                    sum_events_per_step += np.asarray(result[i]['evt_per_step'], dtype=np.int32)
                sum_events += result[i]['n_events']
                n_steps = np.max([n_steps, result[i]['n_steps']])
            
            assert all(sum_events_per_step == [10,10,10]) 
            assert sum_events == 30
            assert n_steps == 3
    
    # Test run.steps() for RunSingleFile
    if size == 1:
        result = test_runsinglefile_steps()
        assert result == {'evt_per_step': [10, 10, 10], 'n_steps': 3, 'n_events': 30}        

