import os
import subprocess
from .xtc import xtc
from .det import det

class Test:
    @classmethod
    def setup_class(cls):
        subprocess.call(['xtcwriter'])

    def test_xtc(self):
        xtc()

    def test_parallel(self):
        subprocess.call(['smdwriter','-f','data.xtc'])
        import shutil
        shutil.copy('data.xtc','data_1.xtc')
        shutil.copy('smd.xtc', 'smd_1.xtc')
        parallel = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'user.py')
        subprocess.call(['mpirun','-n','2','python',parallel])
    
    def test_det(self):
        det()


