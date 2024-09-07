import os, shutil
import subprocess
from setup_input_files import setup_input_files

# cpo and weninc split off this test because of an issue with openmpi where
# a python file that does "import mpi4py" cannot fork an "mpirun" command.
# see: https://bitbucket.org/mpi4py/mpi4py/issues/95/mpi4py-openmpi-300-breaks-subprocess

class Test:
    def test_mpi(self, tmp_path):
        setup_input_files(tmp_path)

        env = dict(list(os.environ.items()) + [
            #('TEST_XTC_DIR', str(tmp_path)),
            ('TEST_XTC_DIR', "/sdf/home/m/monarin/tmp"),
            ('PS_SRV_NODES', '0'),
            ('PS_EB_NODES', '1')
        ])

        loop_based = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'user_loops.py')
        subprocess.check_call(['mpirun','-n','3','python',loop_based], env=env)

        #callback_based = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'user_callbacks.py')
        #subprocess.check_call(['mpirun','-n','3','python',callback_based], env=env)

        run_early_termination = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'run_early_termination.py')
        subprocess.check_call(['mpirun','-n','3','python',run_early_termination], env=env)
        
        # Test more than 1 bigdata node
        loop_exhaustive_based = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ds.py')
        subprocess.check_call(['mpirun','-n','5','python',loop_exhaustive_based], env=env)
        
        run_mixed_rate = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'run_mixed_rate.py')
        subprocess.check_call(['mpirun','-n','5','python',run_mixed_rate], env=env)
        
        run_chunking = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'run_chunking.py')
        subprocess.check_call(['mpirun','-n','5','python',run_chunking], env=env)
        
        run_intg_det = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'run_intg_det.py')
        subprocess.check_call(['mpirun','-n','4','python',run_intg_det], env=env)
        
        # Test more than 1 eb node
        env['PS_EB_NODES'] = '2'
        subprocess.check_call(['mpirun','-n','7','python',run_mixed_rate], env=env)
        subprocess.check_call(['mpirun','-n','7','python',run_chunking], env=env)
        
        env['PS_EB_NODES'] = '1' # reset no. of eventbuilder cores
        env['PS_SRV_NODES'] = '2'
        run_smalldata = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'run_smalldata.py')
        subprocess.check_call(['mpirun','-n','6','python',run_smalldata], env=env)
