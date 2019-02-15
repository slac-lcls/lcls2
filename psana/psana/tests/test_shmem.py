import os, shutil
import subprocess
import sys, os
sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path
from psana import DataSource


class Test:
    @classmethod
    def setup_class(cls):
        subprocess.call(['xtcwriter'])
        subprocess.call(['shmemServer'])

    def launch_server(self):
        cmd_args = ['shmemServer','-n','8','-f','data_shmem.xtc2','-p','shmem_test','-s','0x80000','-c','4']
        return subprocess.Popen(cmd_args, stdout=subprocess.PIPE)

    def launch_client(self):
        ds = DataSource('shmem','shmem_test')
        run = next(ds.runs())
        for evt in run.events():
            print("event transitionId",evt._dgrams[0].seq.service())
                
    def setup_input_files(self):
        subprocess.call(['xtcwriter','-n','4','-f','data_shmem.xtc2', '-t'])
        
    def test_shmem(self):
        self.setup_input_files()
        srv = self.launch_server()
        self.launch_client()
        srv.kill()
        
