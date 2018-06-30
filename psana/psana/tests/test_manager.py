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

    def test_cydgram(self):
        fname = 'data_cydgram.xtc'
        try:
            os.remove(fname)
        except:
            pass

        # read in an old xtc file
        ds = DgramManager('data.xtc')
        pyxtc = dc.parse_xtc(ds)

        for evt in ds:
            pyxtc.parse_event(evt)

        # put the dictionaries in a new xtc file
        cydgram = dc.CyDgram()
        pyxtc.write_events(fname, cydgram)

        # test that the values in the new file are correct
        xtc(fname)

    def test_xtc(self):
        xtc('data.xtc')

    def test_parallel(self):
        subprocess.call(['xtcwriter','-f','data-ts.xtc', '-t']) # Mona FIXME: writing seq in xtcwriter broke dgramCreate
        subprocess.call(['smdwriter','-f','data-ts.xtc'])
        tmp_dir = os.path.join('.tmp','smalldata')
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir)
        shutil.copy('data-ts.xtc',os.path.join('.tmp','data-0.xtc'))
        shutil.copy('data-ts.xtc',os.path.join('.tmp','data-1.xtc'))
        shutil.copy('smd.xtc',os.path.join(tmp_dir,'data-0.smd.xtc'))
        shutil.copy('smd.xtc',os.path.join(tmp_dir,'data-1.smd.xtc'))

        shutil.copy('smd.xtc', os.path.join(tmp_dir, 'data.smd.xtc')) # FIXME: chuck's hack to fix nosetests
        shutil.copy('smd.xtc',os.path.join(tmp_dir,'data_1.smd.xtc')) # FIXME: chuck's hack to fix nosetests

        parallel = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'user.py')
        subprocess.call(['mpirun','-n','3','python',parallel])
    
    def test_det(self):
        det()

    def test_dgram(self):
        run_test_dgraminit()

