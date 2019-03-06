import os, shutil
import subprocess

# cpo and weninc split off this test because of an issue with openmpi where
# a python file that does "import mpi4py" cannot fork an "mpirun" command.
# see: https://bitbucket.org/mpi4py/mpi4py/issues/95/mpi4py-openmpi-300-breaks-subprocess

class Test:
    @classmethod
    def setup_class(cls):
        subprocess.call(['xtcwriter'])

    def setup_input_files(self):
        subprocess.call(['xtcwriter','-f','data-ts.xtc2', '-t']) # Mona FIXME: writing seq in xtcwriter broke dgramCreate
        subprocess.call(['smdwriter','-f','data-ts.xtc2'])
        tmp_dir = os.path.join('.tmp','smalldata')
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir,ignore_errors=True)
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        shutil.copy('data-ts.xtc2',os.path.join('.tmp','data-r0001-s00.xtc2')) # Ex. of run 1 with two detectors s00 and s01
        shutil.copy('data-ts.xtc2',os.path.join('.tmp','data-r0001-s01.xtc2'))
        shutil.copy('smd.xtc2',os.path.join(tmp_dir,'data-r0001-s00.smd.xtc2'))
        shutil.copy('smd.xtc2',os.path.join(tmp_dir,'data-r0001-s01.smd.xtc2'))
        shutil.copy('smd.xtc2',os.path.join('.tmp','data-r0001-epc.xtc2'))

        shutil.copy('data-ts.xtc2',os.path.join(tmp_dir,'data-r0002-s00.smd.xtc2'))
        shutil.copy('data-ts.xtc2',os.path.join(tmp_dir,'data-r0002-s01.smd.xtc2'))

        shutil.copy('smd.xtc2', os.path.join(tmp_dir, 'data.smd.xtc2')) # FIXME: chuck's hack to fix nosetests
        shutil.copy('smd.xtc2',os.path.join(tmp_dir,'data_1.smd.xtc2')) # FIXME: chuck's hack to fix nosetests

    def test_mpi(self):
        self.setup_input_files()

        loop_based = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'user_loops.py')
        subprocess.check_call(['mpirun','-n','3','python',loop_based])

        callback_based = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'user_callbacks.py')
        subprocess.check_call(['mpirun','-n','3','python',callback_based])
        
        loop_exhaustive_based = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ds.py')
        subprocess.check_call(['mpirun','-n','3','python',loop_exhaustive_based])
