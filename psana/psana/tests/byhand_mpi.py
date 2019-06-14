import os, shutil
import subprocess

# cpo and weninc split off this test because of an issue with openmpi where
# a python file that does "import mpi4py" cannot fork an "mpirun" command.
# see: https://bitbucket.org/mpi4py/mpi4py/issues/95/mpi4py-openmpi-300-breaks-subprocess

class Test:
    # as a result of the problem described above, this test assumes all
    # input files have been setup by test_xtc.py
    def test_mpi(self):
        loop_based = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'user_loops.py')
        subprocess.check_call(['mpirun','-n','3','python',loop_based])

        callback_based = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'user_callbacks.py')
        subprocess.check_call(['mpirun','-n','3','python',callback_based])
        
        loop_exhaustive_based = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ds.py')
        subprocess.check_call(['mpirun','-n','3','python',loop_exhaustive_based])
