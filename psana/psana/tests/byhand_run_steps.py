import os
import subprocess
from setup_input_files import setup_input_files

# mona split off this test since it's taking over 10s
# looking for a solution....

class Test:
    def test_mpi(self, tmp_path):
        setup_input_files(tmp_path)
        env = dict(list(os.environ.items()) + [
            ('TEST_XTC_DIR', str(tmp_path)),
            ('PS_SRV_NODES', '0'),
            ('PS_EB_NODES', '1')
        ])

        run_steps = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'run_steps.py')

        subprocess.check_call(['python',run_steps], env=env)

        # Test one EventBuilder with multiple Bigata cores
        subprocess.check_call(['mpirun','-n','5','python',run_steps], env=env)

        # Test multiple EventBuilder with multiple Bigata cores
        env['PS_EB_NODES'] = '2'
        subprocess.check_call(['mpirun','-n','7','python',run_steps], env=env)

        # Test fakestep insert (note that the script overrides some of PS_ env. variables
        #run_fakestep = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'run_fakestep.py')
        #subprocess.check_call(['mpirun','-n','5','python', run_fakestep], env=env)

    def test_steps_w_ts_filter(self, tmp_path):
        # This running with steps and timestamp filter.
        # Note that the test generates its own test files in .tmp_smd0/.tmp
        env = dict(list(os.environ.items()) + [
            ('TEST_XTC_DIR', str(tmp_path)),
        ])
        run_steps_w_ts_filter = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'run_steps_w_ts_filter.py')
        subprocess.check_call(['mpirun','-n','7','python', run_steps_w_ts_filter], env=env)
