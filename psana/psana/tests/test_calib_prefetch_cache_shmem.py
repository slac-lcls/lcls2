import os
import socket
import subprocess
import sys

import pytest

client_count = 3
calib_prefetch_count = 1
dgram_count = 64

# Disable shared Jungfrau derived cache and shared geometry cache for this test.
TEST_ENV_OVERRIDES = {
    "PS_JF_SHARE_DERIVED": "0",
    "PS_GEO_SHARE": "0",
}

# Test-specific xtc2 file for jungfrau detector relative to the test file
#tmp_file_path = "/sdf/data/lcls/drpsrcf/ffb/users/monarin/jungfrau/mfx101332224-r9999-small.xtc2"
# The test file above is no longer working (no calibration constants available),
# so using another jungfrau xtc2 file with available calibration constants.
# The new file is larger, so the test will go own for a long time...
# Use pytest -s and when noticed there are several events processed, interrupt the test (Ctrl-C).
tmp_file_path = "/sdf/data/lcls/ds/mfx/mfx101344525/xtc/mfx101344525-r0125-s009-c000.xtc2"


@pytest.mark.skipif(
    sys.platform == 'darwin' or
    os.getenv('LCLS_TRAVIS') is not None or
    not os.path.exists(tmp_file_path),
    reason="shmem not supported or required xtc2 file not found"
)

class TestCalibPrefetchCacheShmem:
    """
    Integration test for calibration constant prefetching and
    detector-side cache loading via shared memory DataSource.
    """

    @staticmethod
    def launch_server(pid, env):
        cmd_args = [
            'shmemServer', '-c', str(client_count + calib_prefetch_count),
            '-n', '10', '-f', tmp_file_path, '-p', f'shmem_test_{pid}',
            '-r', '1', '-L', '1'
        ]
        return subprocess.Popen(cmd_args, env=env)

    def launch_supervisor(self, pid, supervisor_ip_addr, env):
        shmem_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'shmem_client.py')
        cmd_args = ['python', shmem_file, pid, '1', supervisor_ip_addr, '--test-detector-cache']
        return subprocess.Popen(cmd_args, env=env)

    def launch_client(self, pid, supervisor_ip_addr, env):
        shmem_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'shmem_client.py')
        cmd_args = ['python', shmem_file, pid, '0', supervisor_ip_addr, '--test-detector-cache']
        return subprocess.Popen(cmd_args, env=env)

    def launch_calib_prefetch(self, pid, env):
        cmd_args = [
            'python', '-m', 'psana.pscalib.app.calib_prefetch',
            '--shmem', f'shmem_test_{pid}',
            '--log-level', 'INFO',
            '--detectors', 'jungfrau'
        ]
        return subprocess.Popen(cmd_args, env=env)

    @pytest.mark.slow
    def test_detector_cache_shmem(self):
        cli = []
        pid = str(os.getpid())
        env = os.environ.copy()
        env.update(TEST_ENV_OVERRIDES)
        srv = self.launch_server(pid, env)
        assert srv is not None, "server launch failure"

        self.launch_calib_prefetch(pid, env)

        IPAddr = socket.gethostbyname(socket.gethostname())
        sock = socket.socket()
        sock.bind(('', 0))
        port_no = sock.getsockname()[1]
        sock.close()
        supervisor_ip_addr = f'{IPAddr}:{port_no}'

        try:
            for i in range(client_count):
                if i == 0:
                    cli.append(self.launch_supervisor(pid, supervisor_ip_addr, env))
                else:
                    cli.append(self.launch_client(pid, supervisor_ip_addr, env))
                assert cli[i] is not None, f"client {i} launch failure"
        except:
            srv.kill()
            raise

        for i in range(client_count):
            cli[i].wait()

        srv.wait()
