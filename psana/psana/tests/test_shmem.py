import os, shutil
import subprocess
import sys, os
from psana import DataSource

class Test:
    def launch_server(self,tmp_file,pid):
        cmd_args = ['shmemServer','-n','8','-f',tmp_file,'-p','shmem_test_'+pid,'-s','0x80000','-c','4']
        return subprocess.Popen(cmd_args, stdout=subprocess.PIPE)

    def launch_client(self,pid):
        dg_count = 0
        ds = DataSource('shmem','shmem_test_'+pid)
        run = next(ds.runs())
        for evt in run.events():
            if not evt:
                break
            if not evt._dgrams:
                break
            if not len(evt._dgrams):
                break
            # check for L1 accept transition ID 12
            if evt._dgrams[0].seq.service() == 12:
                dg_count += 1
        if (sys.version_info.major>2): assert dg_count == 4,"invalid dgram count"
                
    def setup_input_files(self):
        tmp_dir = os.path.join('.tmp','shmem')
        tmp_file = tmp_dir+'/data_shmem.xtc2'        
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir,ignore_errors=True)
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        subprocess.call(['xtcwriter','-n','4','-f',tmp_file, '-t'])        
        return tmp_file
        
    def test_shmem(self):
        pid = str(os.getpid())
        tmp_file = self.setup_input_files()
        srv = self.launch_server(tmp_file,pid)
        try:
            self.launch_client(pid)
        except:
            srv.kill()
            raise
        srv.wait()
