import os, shutil
import subprocess
import sys
import pytest
sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path
from xtc import xtc
from det import det

import hashlib
from psana import DataSource
import dgramCreate as dc
from setup_input_files import setup_input_files


class Test:
    @staticmethod
    @pytest.fixture(scope='function')
    def xtc_file(tmp_path):
        fname = str(tmp_path / 'data.xtc2')
        subprocess.call(['xtcwriter', '-f', fname])
        return fname

    # cpo: remove this test because cydgram doesn't support charstr/enum
    # datatypes which are produced by xtcwriter.  also, cydgram is tested
    # in the test_py2xtc.py test (without these datatypes).
    """
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
    """

    def test_xtcdata(self, xtc_file):
        xtc(xtc_file, nsegments=2)

    def test_serial(self, tmp_path):
        setup_input_files(tmp_path)

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
        setup_input_files(tmp_path)

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
        setup_input_files(tmp_path)

        import run_pickle
        run_pickle.test_run_pickle(tmp_path)

    @pytest.mark.skipif(sys.platform == 'darwin', reason="psana with legion not supported on mac")
    def test_legion_pickle(self, tmp_path):
        # Again, in Legion
        setup_input_files(tmp_path)

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

