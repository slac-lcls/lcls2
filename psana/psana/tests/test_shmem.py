import os, shutil
import subprocess
import sys, os
import pytest
from psana import DataSource

client_count = 4  # number of clients in test
dgram_count  = 64 # number of expected datagrams per client

@pytest.mark.skipif(sys.platform == 'darwin', reason="shmem not supported on mac")
class Test:
    def launch_server(self,tmp_file,pid):
        cmd_args = ['shmemServer','-c',str(client_count),'-n','10','-f',tmp_file,'-p','shmem_test_'+pid,'-s','0x80000']
        return subprocess.Popen(cmd_args, stdout=subprocess.PIPE)

    def launch_client(self,pid):
        shmem_file = os.path.dirname(os.path.realpath(__file__))+'/shmem_client.py'  
        cmd_args = ['python',shmem_file,pid]
        return subprocess.Popen(cmd_args, stdout=subprocess.PIPE)
                
    def setup_input_files(self, tmp_path):
        tmp_dir = tmp_path / 'shmem'
        tmp_dir.mkdir()
        tmp_file = tmp_dir / 'data_shmem.xtc2'
        # cpo: disable epics in the stream for now with "-e 0" since
        # we need changes in the TransitionCache.
        subprocess.call(['xtcwriter','-n',str(dgram_count),'-f',str(tmp_file),'-e','0'])
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
        for i in range(client_count):
          cli[i].wait()
          assert cli[i].returncode == dgram_count,"client "+str(i)+" failure"
        srv.wait()
