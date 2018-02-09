import os
import subprocess
from .xtc import xtc
from .det import det

class Test:
    @classmethod
    def setup_class(cls):
        instdir = os.environ['INSTDIR']
        subprocess.call([os.path.join(instdir, 'bin/xtcwriter')])

    def test_xtc(self):
        xtc()

    def test_det(self):
        det()
