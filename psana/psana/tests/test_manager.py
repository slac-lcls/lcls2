import os
import subprocess
from .xtc import xtc
from .det import det

import hashlib
from psana.dgrammanager import DgramManager
import dgramCreate as dc


class Test:
    @classmethod
    def setup_class(cls):
        subprocess.call(['xtcwriter'])

    def hash_xtc(self, filename):
        with open(filename, "rb") as f:
            data=f.read()
        data = bytes(sorted(data))
        md5 = hashlib.md5()
        md5.update(data)

        return md5.hexdigest()

    def test_copy(self):
        try:
            os.remove('data_copy.xtc')
        except:
            pass

        input = 'data.xtc'
        input2 = 'data_copy.xtc'

        ds = DgramManager(input)
        pyxtc = dc.parse_xtc(ds)

        for evt in ds:
            pyxtc.parse_event(evt)

        cydgram = dc.CyDgram()
        pyxtc.write_events(input2, cydgram)

        h1 = self.hash_xtc(input)
        h2 = self.hash_xtc(input2)
        assert(h1==h2)

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


