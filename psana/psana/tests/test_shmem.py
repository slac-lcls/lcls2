import os, shutil
import subprocess
import sys, os
from psana import DataSource

class Test:
    def launch_server(self):
        cmd_args = ['shmemServer','-n','8','-f','data_shmem.xtc2','-p','shmem_test','-s','0x80000','-c','4']
        return subprocess.Popen(cmd_args, stdout=subprocess.PIPE)

    def launch_client(self):
        dg_count = 0
        ds = DataSource('shmem','shmem_test')
        run = next(ds.runs())
        for evt in run.events():
            if not evt:
                break
            if not evt._dgrams:
                break
            if not len(evt._dgrams):
                break   
            if evt._dgrams[0].seq.service() == 12:
                dg_count += 1
        assert dg_count == 4
                
    def setup_input_files(self):
        subprocess.call(['xtcwriter','-n','4','-f','data_shmem.xtc2', '-t'])
        
    def test_shmem(self):
        self.setup_input_files()
        srv = self.launch_server()
        self.launch_client()
        srv.kill()
        
