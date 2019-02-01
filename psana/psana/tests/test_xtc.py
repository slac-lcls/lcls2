import os, shutil
import subprocess
import sys, os
sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path
from xtc import xtc
from det import det
from test_dgraminit import run as run_test_dgraminit

import hashlib
from psana import DataSource
import dgramCreate as dc

try:
    from pathlib import Path
except:
    from pathlib2 import Path # python 2 backport

class Test:
    @classmethod
    def setup_class(cls):
        subprocess.call(['xtcwriter'])

    def test_cydgram(self):
        fname = 'data_cydgram.xtc2'
        try:
            os.remove(fname)
        except:
            pass

        # read in an old xtc file
        ds = DataSource('data.xtc2')
        for run in ds.runs():
            pyxtc = dc.parse_xtc(run.configs[0])
            for evt in run.events():
                pyxtc.parse_event(evt)

        # put the dictionaries in a new xtc file
        cydgram = dc.CyDgram()
        pyxtc.write_events(fname, cydgram)

        # test that the values in the new file are correct
        xtc(fname)
 
    def test_xtc(self):
        xtc('data.xtc2')

    def setup_input_files(self):
        subprocess.call(['xtcwriter','-f','data-ts.xtc2', '-t']) # Mona FIXME: writing seq in xtcwriter broke dgramCreate
        subprocess.call(['smdwriter','-f','data-ts.xtc2'])
        tmp_dir = os.path.join('.tmp','smalldata')
        
        Path(tmp_dir).mkdir(exist_ok=True) # prevents race condition (python 2 compatible).

        shutil.copy('data-ts.xtc2',os.path.join('.tmp','data-r0001-s00.xtc2')) # Ex. of run 1 with two detectors s00 and s01
        shutil.copy('data-ts.xtc2',os.path.join('.tmp','data-r0001-s01.xtc2'))
        shutil.copy('smd.xtc2',os.path.join(tmp_dir,'data-r0001-s00.smd.xtc2'))
        shutil.copy('smd.xtc2',os.path.join(tmp_dir,'data-r0001-s01.smd.xtc2'))

        shutil.copy('smd.xtc2', os.path.join(tmp_dir, 'data.smd.xtc2')) # FIXME: chuck's hack to fix nosetests
        shutil.copy('smd.xtc2',os.path.join(tmp_dir,'data_1.smd.xtc2')) # FIXME: chuck's hack to fix nosetests

    def test_serial(self):
        self.setup_input_files()

        loop_based = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'user_loops.py')
        subprocess.check_call(['python',loop_based])

        callback_based = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'user_callbacks.py')
        subprocess.check_call(['python',callback_based])
        
        loop_exhaustive_based = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ds.py')
        subprocess.check_call(['python',loop_exhaustive_based])

    def test_mpi(self):
        self.setup_input_files()

        loop_based = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'user_loops.py')
        subprocess.check_call(['mpirun','-n','3','python',loop_based])

        callback_based = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'user_callbacks.py')
        subprocess.check_call(['mpirun','-n','3','python',callback_based])
        
        loop_exhaustive_based = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ds.py')
        subprocess.check_call(['mpirun','-n','3','python',loop_exhaustive_based])


    def test_legion(self):
        self.setup_input_files()

        python_path = os.environ.get('PYTHONPATH', '').split(':')
        python_path.append(os.path.dirname(os.path.realpath(__file__)))
        env = dict(list(os.environ.items()) + [
            ('PYTHONPATH', ':'.join(python_path)),
            ('PS_PARALLEL', 'legion'),
        ])
        subprocess.check_call(['legion_python', 'user_callbacks', '-ll:py', '1'], env=env)

    def test_run_pickle(self):
        # Test that run is pickleable
        self.setup_input_files()

        import run_pickle
        run_pickle.test_run_pickle()

    def test_legion(self):
        # Again, in Legion
        self.setup_input_files()

        python_path = os.environ.get('PYTHONPATH', '').split(':')
        python_path.append(os.path.dirname(os.path.realpath(__file__)))
        env = dict(list(os.environ.items()) + [
            ('PYTHONPATH', ':'.join(python_path)),
            ('PS_PARALLEL', 'legion'),
        ])
        subprocess.check_call(['legion_python', 'run_pickle', '-ll:py', '1'], env=env)

    def test_det(self):
        det()

    def test_dgram(self):
        run_test_dgraminit()
