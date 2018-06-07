import os, shutil
import subprocess
import sys, os
sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path
from xtc import xtc
from det import det
from test_dgraminit import run as run_test_dgraminit

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
        tmp_dir = os.path.join('.tmp','smalldata')
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir)
        shutil.copy('data.xtc',os.path.join('.tmp','data-0.xtc'))
        shutil.copy('data.xtc',os.path.join('.tmp','data-1.xtc'))
        shutil.copy('smd.xtc',os.path.join(tmp_dir,'data-0.smd.xtc'))
        shutil.copy('smd.xtc',os.path.join(tmp_dir,'data-1.smd.xtc'))
        parallel = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'user.py')
        subprocess.call(['mpirun','-n','2','python',parallel])
    
    def test_det(self):
        det()

    def test_dgram(self):
        run_test_dgraminit()

