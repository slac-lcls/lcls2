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
    @staticmethod
    @pytest.fixture(scope='function')
    def xtc_file(tmp_path):
        fname = str(tmp_path / 'data.xtc2')
        subprocess.call(['xtcwriter', '-f', fname])
        return fname

    def test_cydgram(self, xtc_file, tmp_path):
        fname = str(tmp_path / 'data_cydgram.xtc2')

        # read in an old xtc file
        ds = DataSource(files=xtc_file)
        for run in ds.runs():
            pyxtc = dc.parse_xtc(run.configs[0])
            for evt in run.events():
                pyxtc.parse_event(evt)

        # put the dictionaries in a new xtc file
        cydgram = dc.CyDgram()
        pyxtc.write_events(fname, cydgram)

        # test that the values in the new file are correct
        xtc(fname, nsegments=1, cydgram=True)

    def test_xtcdata(self, xtc_file):
        xtc(xtc_file, nsegments=2)

    def setup_input_files(self, tmp_path):
        xtc_dir = tmp_path / '.tmp'
        xtc_dir.mkdir()
        smd_dir = xtc_dir / 'smalldata'
        smd_dir.mkdir()

        # segments 0,1 and "counting" timestamps for event-building
        s01file = str(xtc_dir / 'data-r0001-s00.xtc2')
        subprocess.call(['xtcwriter','-f',s01file,'-t'])
        subprocess.call(['smdwriter','-f',s01file,'-o',str(smd_dir / 'data-r0001-s00.smd.xtc2')])

        # segments 2,3
        s23file = str(xtc_dir / 'data-r0001-s01.xtc2')
        subprocess.call(['xtcwriter','-f',s23file,'-t','-s','2',])
        subprocess.call(['smdwriter','-f',s23file,'-o',str(smd_dir / 'data-r0001-s01.smd.xtc2')])

        subprocess.call(['epicswriter','-f',s01file,'-o',str(xtc_dir / 'data-r0001-s02.xtc2')])
        # Epics data with streamId 2
        subprocess.call(['epicswriter','-f',s01file,'-o',str(xtc_dir / 'data-r0001-s03.xtc2'), '-s', '2'])
        
        shutil.copy(s01file,str(smd_dir / 'data-r0002-s00.smd.xtc2'))
        shutil.copy(s23file,str(smd_dir / 'data-r0002-s01.smd.xtc2'))

    def test_serial(self, tmp_path):
        self.setup_input_files(tmp_path)

        env = dict(list(os.environ.items()) + [
            ('TEST_XTC_DIR', str(tmp_path)),
        ])

        loop_based = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'user_loops.py')
        subprocess.check_call(['python',loop_based], env=env)

        callback_based = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'user_callbacks.py')
        subprocess.check_call(['python',callback_based], env=env)
        
        callback_based = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'parallelreader.py')
        subprocess.check_call(['python',callback_based], env=env)
        
        callback_based = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ds.py')
        subprocess.check_call(['python',callback_based], env=env)

    @pytest.mark.skipif(sys.platform == 'darwin', reason="psana with legion not supported on mac")
    def test_legion(self, tmp_path):
        self.setup_input_files(tmp_path)

        # Legion script mode.
        env = dict(list(os.environ.items()) + [
            ('PS_PARALLEL', 'legion'),
            ('TEST_XTC_DIR', str(tmp_path)),
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

    def test_run_pickle(self, tmp_path):
        # Test that run is pickleable
        self.setup_input_files(tmp_path)

        import run_pickle
        run_pickle.test_run_pickle(tmp_path)

    @pytest.mark.skipif(sys.platform == 'darwin', reason="psana with legion not supported on mac")
    def test_legion_pickle(self, tmp_path):
        # Again, in Legion
        self.setup_input_files(tmp_path)

        python_path = os.environ.get('PYTHONPATH', '').split(':')
        python_path.append(os.path.dirname(os.path.realpath(__file__)))
        env = dict(list(os.environ.items()) + [
            ('PYTHONPATH', ':'.join(python_path)),
            ('PS_PARALLEL', 'legion'),
            ('TEST_XTC_DIR', str(tmp_path)),
        ])
        subprocess.check_call(['legion_python', 'run_pickle', '-ll:py', '1'], env=env)

    def test_det(self, xtc_file):
        det(xtc_file)

