import os, shutil
import subprocess
import sys, os
sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path
from xtc import xtc
from det import det
from test_dgraminit import run as run_test_dgraminit

import hashlib
from psana.dgrammanager import DgramManager
from psana import DataSource
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

    def setup_input_files(self):
        subprocess.call(['xtcwriter','-f','data-ts.xtc', '-t']) # Mona FIXME: writing seq in xtcwriter broke dgramCreate
        subprocess.call(['smdwriter','-f','data-ts.xtc'])
        tmp_dir = os.path.join('.tmp','smalldata')
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir)
        shutil.copy('data-ts.xtc',os.path.join('.tmp','data-r0001-s00.xtc')) # Ex. of run 1 with two detectors s00 and s01
        shutil.copy('data-ts.xtc',os.path.join('.tmp','data-r0001-s01.xtc'))
        shutil.copy('smd.xtc',os.path.join(tmp_dir,'data-r0001-s00.smd.xtc'))
        shutil.copy('smd.xtc',os.path.join(tmp_dir,'data-r0001-s01.smd.xtc'))

        shutil.copy('smd.xtc', os.path.join(tmp_dir, 'data.smd.xtc')) # FIXME: chuck's hack to fix nosetests
        shutil.copy('smd.xtc',os.path.join(tmp_dir,'data_1.smd.xtc')) # FIXME: chuck's hack to fix nosetests

    def test_mpi(self):
        self.setup_input_files()

        parallel = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'user.py')
        subprocess.check_call(['mpirun','-n','3','python',parallel])
    
    def test_legion(self):
        self.setup_input_files()

        python_path = os.environ.get('PYTHONPATH', '').split(':')
        python_path.append(os.path.dirname(os.path.realpath(__file__)))
        env = dict(list(os.environ.items()) + [
            ('PYTHONPATH', ':'.join(python_path)),
            ('PS_PARALLEL', 'legion'),
        ])
        subprocess.check_call(['legion_python', 'user', '-ll:py', '1'], env=env)

    def test_ds_pickle(self):
        self.setup_input_files()

        import pickle
        xtc_dir = os.path.join(os.getcwd(),'.tmp')
        ds = DataSource('exp=xpptut13:dir=%s'%(xtc_dir), filter=filter)
        ds_new = pickle.loads(pickle.dumps(ds))
        assert ds == ds_new

    def test_det(self):
        det()

    def test_dgram(self):
        run_test_dgraminit()
