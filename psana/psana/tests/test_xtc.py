import os
import subprocess
import sys

import pytest
from psana import DataSource

from det import det, det_container, detnames
from run_chunking import run_test_chunking
from run_loop_callback import run_test_loop_callback
from setup_input_files import setup_input_files
from xtc import xtc

sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path


class Test:
    # Use pytest fixtures for creating test folders.
    # Test data are in /tmp/pytest-of-username
    @staticmethod
    @pytest.fixture(scope='function')
    def xtc_file(tmp_path):
        fname = str(tmp_path / 'data.xtc2')
        subprocess.call(['xtcwriter', '-f', fname])
        return fname

    # cpo: remove this test because cydgram doesn't support charstr/enum
    # datatypes which are produced by xtcwriter.  also, cydgram is tested
    # in the test_py2xtc.py test (without these datatypes).
    #def test_cydgram(self, xtc_file, tmp_path):
    #    fname = str(tmp_path / 'data_cydgram.xtc2')

    #    # read in an old xtc file
    #    ds = DataSource(files=xtc_file)
    #    for run in ds.runs():
    #        pyxtc = dc.parse_xtc(run.configs[0])
    #        for evt in run.events():
    #            pyxtc.parse_event(evt)

    #    # put the dictionaries in a new xtc file
    #    cydgram = dc.CyDgram()
    #    pyxtc.write_events(fname, cydgram)

    #    # test that the values in the new file are correct
    #    xtc(fname, nsegments=1, cydgram=True)

    def test_xtcdata(self, xtc_file):
        xtc(xtc_file, nsegments=2)

    def test_serial(self, tmp_path):
        setup_input_files(tmp_path)

        env = dict(list(os.environ.items()) + [
            ('TEST_XTC_DIR', str(tmp_path)),
        ])

        loop_based = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'user_loops.py')
        subprocess.check_call(['python',loop_based], env=env)

        loop_based_exhausted = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ds.py')
        subprocess.check_call(['python',loop_based_exhausted], env=env)

        run_early_termination = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'run_early_termination.py')
        subprocess.check_call(['python',run_early_termination], env=env)

    @pytest.mark.skip(reason="Skipping this test for now – needs fix.")
    def test_serial_loop_callback(self, tmp_path):
        setup_input_files(tmp_path, n_files=2, slow_update_freq=4, n_motor_steps=3, n_events_per_step=10, gen_run2=False)
        xtc_dir = os.path.join(str(tmp_path), ".tmp")
        run_test_loop_callback(xtc_dir)
        run_test_loop_callback(xtc_dir, withstep=True)

    @pytest.mark.skip(reason="Skipping this test for now – needs fix.")
    def test_serial_steps_w_ts_filter(self, tmp_path):
        env = dict(list(os.environ.items()) + [
            ('TEST_XTC_DIR', str(tmp_path)),
        ])
        step_w_ts_filter = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'run_steps_w_ts_filter.py')
        subprocess.check_call(['python', step_w_ts_filter], env=env)

    def test_detnames(self, xtc_file):
        # for now just check that the various detnames don't crash
        for flag in ['-r','-e','-s','-i']:
            subprocess.check_call(['detnames',flag,xtc_file])
        subprocess.check_call(['detnames',xtc_file])

    def test_det(self, xtc_file):
        det(xtc_file)
        detnames(xtc_file)
        det_container(xtc_file)

    def test_step_det(self):
        xtc_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data-w-step.xtc2')
        ds = DataSource(files=xtc_file)
        myrun = next(ds.runs())
        det = myrun.scaninfo
        expected_det = {('step_value', 'raw'): 'raw', ('step_docstring', 'raw'): 'raw'}
        assert det == expected_det
        step_v = myrun.Detector('step_value')
        step_s = myrun.Detector('step_docstring')
        for nstep,step in enumerate(myrun.steps()):
            if nstep == 0:
                assert step_v(step) == 0
                assert step_s(step) == '{"detname": "epixquad_0", "scantype": "pedestal", "step": 0}'
            elif nstep == 1:
                assert step_v(step) == 1
                assert step_s(step) == '{"detname": "epixquad_0", "scantype": "pedestal", "step": 1}'
            for nevt,evt in enumerate(step.events()):
                pass

    def test_chunking(self):
        run_test_chunking()
