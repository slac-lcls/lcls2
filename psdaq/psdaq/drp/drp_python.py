import sys
import posix_ipc
from psdaq.configdb.pub_server import pub_bind
from psdaq.configdb.sub_client import sub_connect

partition = int(sys.argv[1])
pebble_bufsize = int(sys.argv[2])
transition_bufsize = int(sys.argv[3])
shm_mem_size = int(sys.argv[4])
detector_name = sys.argv[5]
detector_segment = sys.argv[6]
worker_num = int(sys.argv[7])


class IPCInfo:
    def __init__(self, partition, detector_name, detector_segment, worker_num, shm_mem_size):

        keybase = f"p{partition}_{detector_name}_{detector_segment}"; 

        try:
            self.mq_inp = posix_ipc.MessageQueue(f"/mqinp_{keybase}_{worker_num}", read=True, write=False) 
            self.mq_res = posix_ipc.MessageQueue(f"/mqres_{keybase}_{worker_num}", read=False, write=True)
        except posix_ipc.Error as exp:
            assert(False)
        try:
            self.shm_inp = posix_ipc.SharedMemory(f"/shminp_{keybase}_{worker_num}")
            self.shm_res = posix_ipc.SharedMemory(f"/shmres_{keybase}_{worker_num}")
        except posix_ipc.Error as exp:
            assert(False)


class DrpInfo:
    def __init__(self, detector_name, detector_segment, worker_num, pebble_bufsize, transition_bufsize, ipc_info):
        self.det_name = detector_name
        self.det_segment = detector_segment
        self.worker_num = worker_num
        self.pebble_bufsize = pebble_bufsize
        self.transition_bufsize = transition_bufsize
        self.ipc_info = ipc_info

ipc_info = IPCInfo(partition, detector_name, detector_segment, worker_num, shm_mem_size)
drp_info = DrpInfo(detector_name, detector_segment, worker_num, pebble_bufsize, transition_bufsize, ipc_info)

# Setup socket for calibration constant broadcast
is_publisher = True if worker_num == 0 else False
socket_name = detector_name + "_" + detector_segment
if is_publisher:
    pub_socket = pub_bind(socket_name)
else:
    sub_socket = sub_connect(socket_name)
print(f"[Python - Thread {worker_num}] {is_publisher=} setup socket]")

while True:
    print(f"[Worker: {worker_num} - Python] Python process waiting for new script to run")
    message, priority = ipc_info.mq_inp.receive()
    if message == "stop":
        exit(0)
    with open(message, "r") as fh:
        code = compile(fh.read(), message, 'exec')
        exec(code, globals(), locals())
