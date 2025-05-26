# Test shmem datasource with pubsub broadcasting

import os
import socket
import subprocess
import sys

import pytest

client_count = 4  # number of clients in test (1 supervisor, 3 clients)
dgram_count  = 64 # number of expected datagrams per client

@pytest.mark.skipif(sys.platform == 'darwin' or os.getenv('LCLS_TRAVIS') is not None, reason="shmem not supported on mac and centos7 failing in travis for unknown reasons")
class Test:

    @staticmethod
    def launch_server(tmp_file,pid):
        cmd_args = ['shmemServer','-c',str(client_count),'-n','10','-f',tmp_file,'-p','shmem_test_'+pid,'-s','0x80000']
        return subprocess.Popen(cmd_args)

    def launch_supervisor(self,pid,supervisor_ip_addr):
        shmem_file = os.path.dirname(os.path.realpath(__file__))+'/shmem_client.py'
        cmd_args = ['python',shmem_file,pid,'1',supervisor_ip_addr]
        return subprocess.Popen(cmd_args)

    def launch_client(self,pid,supervisor_ip_addr):
        shmem_file = os.path.dirname(os.path.realpath(__file__))+'/shmem_client.py'
        cmd_args = ['python',shmem_file,pid,'0',supervisor_ip_addr]
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
        assert srv is not None,"server launch failure"

        # shmem_ds uses host addr and port determined externally for
        # calibration constant broadcasting. In this test, we simulate
        # such a situation by acquiring an available port using a socket
        # then closing it immediately so that this port can be used later.
        IPAddr = socket.gethostbyname(socket.gethostname())
        sock = socket.socket()
        sock.bind(('', 0))
        port_no = sock.getsockname()[1]
        sock.close()
        supervisor_ip_addr = f'{IPAddr}:{port_no}'

        try:
            for i in range(client_count):
              if i == 0:
                  cli.append(self.launch_supervisor(pid, supervisor_ip_addr))
              else:
                  cli.append(self.launch_client(pid, supervisor_ip_addr))
              assert cli[i] is not None,"client "+str(i)+ " launch failure"
        except:
            srv.kill()
            raise
        nevents = 0
        for i in range(client_count):
          cli[i].wait()
          nevents += cli[i].returncode
        # cpo thinks the precise number of events in this assert
        # is not guaranteed, given the flexible nature of shmem
        # should be 64 but hope for 2
        assert nevents >= 2,'incorrect number of l1accepts. found/expected: '+str(nevents)+'/'+str(dgram_count)
        srv.wait()

