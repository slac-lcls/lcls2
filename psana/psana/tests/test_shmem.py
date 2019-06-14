import os, shutil
import subprocess
import sys, os
from psana import DataSource

client_count = 4  # number of clients in test
dgram_count  = 64 # number of expected datagrams per client

class Test:
    def launch_server(self,tmp_file,pid):
        cmd_args = ['shmemServer','-c',str(client_count),'-n','10','-f',tmp_file,'-p','shmem_test_'+pid,'-s','0x80000','-c','4']
        return subprocess.Popen(cmd_args, stdout=subprocess.PIPE)

    def launch_client(self,pid):
        shmem_file = os.path.dirname(os.path.realpath(__file__))+'/shmem_client.py'  
        cmd_args = ['python',shmem_file,pid]
        return subprocess.Popen(cmd_args, stdout=subprocess.PIPE)
                
    def setup_input_files(self):
        tmp_dir = os.path.join('.tmp','shmem')
        tmp_file = tmp_dir+'/data_shmem.xtc2'        
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir,ignore_errors=True)
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        subprocess.call(['xtcwriter','-n',str(dgram_count),'-f',tmp_file])
        return tmp_file
        
    def test_shmem(self):
        cli = []
        pid = str(os.getpid())
        tmp_file = self.setup_input_files()
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
