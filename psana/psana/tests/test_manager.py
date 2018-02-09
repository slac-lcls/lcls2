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

    def test_det(self):
        det()
