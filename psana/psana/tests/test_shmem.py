import os, shutil
import subprocess
import sys, os
import pytest
from psana import DataSource

client_count = 4  # number of clients in test
dgram_count  = 64 # number of expected datagrams per client

@pytest.mark.skipif(sys.platform == 'darwin' or os.getenv('TRAVIS') is not None, reason="shmem not supported on mac and centos7 failing in travis for unknown reasons")
class Test:

    @staticmethod
    def launch_server(tmp_file,pid):
        cmd_args = ['shmemServer','-c',str(client_count),'-n','10','-f',tmp_file,'-p','shmem_test_'+pid,'-s','0x80000']
        return subprocess.Popen(cmd_args)

    def launch_client(self,pid):
        shmem_file = os.path.dirname(os.path.realpath(__file__))+'/shmem_client.py'  
        cmd_args = ['python',shmem_file,pid]
        return subprocess.Popen(cmd_args)
                
    @staticmethod
    def setup_input_files(tmp_path):
        tmp_dir = tmp_path / 'shmem'
        tmp_dir.mkdir()
        tmp_file = tmp_dir / 'data_shmem.xtc2'
        subprocess.call(['xtcwriter','-t','-n',str(dgram_count),'-f',str(tmp_file)])
        return tmp_file
        
    def test_shmem(self, tmp_path):
        cli = []
        pid = str(os.getpid())
        tmp_file = self.setup_input_files(tmp_path)
        srv = self.launch_server(tmp_file,pid)
        assert srv != None,"server launch failure"
        try:
            for i in range(client_count):
              cli.append(self.launch_client(pid))
              assert cli[i] != None,"client "+str(i)+ " launch failure"
        except:
            srv.kill()
            raise
        nevents = 0
        for i in range(client_count):
          cli[i].wait()
          nevents += cli[i].returncode
        assert nevents == dgram_count,'incorrect number of l1accepts. found/expected: '+str(nevents)+'/'+str(dgram_count)
        srv.wait()
