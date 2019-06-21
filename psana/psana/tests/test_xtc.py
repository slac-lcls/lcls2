import os, shutil
import subprocess
import sys, os
import pytest
sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path
from xtc import xtc
from det import det

import hashlib
from psana import DataSource
import dgramCreate as dc


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
        ds = DataSource(files='data.xtc2')
        for run in ds.runs():
            pyxtc = dc.parse_xtc(run.configs[0])
            for evt in run.events():
                pyxtc.parse_event(evt)

        # put the dictionaries in a new xtc file
        cydgram = dc.CyDgram()
        pyxtc.write_events(fname, cydgram)

        # test that the values in the new file are correct
        xtc(fname, nsegments=1, cydgram=True)

    def test_xtcdata(self):
        xtc('data.xtc2', nsegments=2)

    def setup_input_files(self):
        xtc_dir = '.tmp'
        smd_dir = os.path.join(xtc_dir,'smalldata')

        if os.path.exists(smd_dir):
            shutil.rmtree(smd_dir,ignore_errors=True)
        if not os.path.exists(smd_dir):
            os.makedirs(smd_dir)

        # segments 0,1 and "counting" timestamps for event-building
        s01file = os.path.join(xtc_dir,'data-r0001-s00.xtc2')
        subprocess.call(['xtcwriter','-f',s01file,'-t'])
        subprocess.call(['smdwriter','-f',s01file,'-o',os.path.join(smd_dir,'data-r0001-s00.smd.xtc2')])

        # segments 2,3
        s23file = os.path.join(xtc_dir,'data-r0001-s01.xtc2')
        subprocess.call(['xtcwriter','-f',s23file,'-t','-s','2',])
        subprocess.call(['smdwriter','-f',s23file,'-o',os.path.join(smd_dir,'data-r0001-s01.smd.xtc2')])

        subprocess.call(['epicswriter','-f',s01file,'-o',os.path.join(xtc_dir,'data-r0001-s02.xtc2')])
        # Epics data with streamId 2
        subprocess.call(['epicswriter','-f',s01file,'-o',os.path.join(xtc_dir,'data-r0001-s03.xtc2'), '-s', '2'])
        
        shutil.copy(s01file,os.path.join(smd_dir,'data-r0002-s00.smd.xtc2'))
        shutil.copy(s23file,os.path.join(smd_dir,'data-r0002-s01.smd.xtc2'))

    def test_serial(self):
        self.setup_input_files()

        loop_based = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'user_loops.py')
        subprocess.check_call(['python',loop_based])

        callback_based = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'user_callbacks.py')
        subprocess.check_call(['python',callback_based])
        
        callback_based = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'parallelreader.py')
        subprocess.check_call(['python',callback_based])
        
        callback_based = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ds.py')
        subprocess.check_call(['python',callback_based])

    @pytest.mark.skipif(sys.platform == 'darwin', reason="psana with legion not supported on mac")
    def test_legion(self):
        self.setup_input_files()

        # Legion script mode.
        env = dict(list(os.environ.items()) + [
            ('PS_PARALLEL', 'legion'),
        ])
        callback_based = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'user_callbacks.py')
        subprocess.check_call(['legion_python', callback_based, '-ll:py', '1'], env=env)

        # Legion module mode.
        python_path = os.environ.get('PYTHONPATH', '').split(':')
        python_path.append(os.path.dirname(os.path.realpath(__file__)))
        env.update({
            'PYTHONPATH': ':'.join(python_path),
        })
        subprocess.check_call(['legion_python', 'user_callbacks', '-ll:py', '1'], env=env)

    def test_run_pickle(self):
        # Test that run is pickleable
        self.setup_input_files()

        import run_pickle
        run_pickle.test_run_pickle()

    @pytest.mark.skipif(sys.platform == 'darwin', reason="psana with legion not supported on mac")
    def test_legion_pickle(self):
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

