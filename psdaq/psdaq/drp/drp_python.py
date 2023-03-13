import sys
import sysv_ipc
import importlib
from psdaq.configdb.pub_server import pub_bind
from psdaq.configdb.sub_client import sub_connect

key_base = int(sys.argv[1])
pebble_bufsize = int(sys.argv[2])
transition_bufsize = int(sys.argv[3])
shm_mem_size = int(sys.argv[4])
detector_name = sys.argv[5]
detector_segment = sys.argv[6]
worker_num = int(sys.argv[7])

print(f"[Worker {worker_num} - Python] DEBUG Keybase: {key_base}, Pebble bufsize: {pebble_bufsize}, "
      f"Transition bufsize: {transition_bufsize}, Shmem size: {shm_mem_size}, "
      f"Detecter name: {detector_name}, Detector segment: {detector_segment}")

class IPCInfo:
    def __init__(self, key_base, shm_mem_size):
        try:
            self.mq_inp = sysv_ipc.MessageQueue(key_base) # sysv_ipc.IPC_CREAT)
            self.mq_res = sysv_ipc.MessageQueue(key_base+1) # sysv_ipc.IPC_CREAT)
        except sysv_ipc.Error as exp:
            assert(False)
        try:
            self.shm_inp = sysv_ipc.SharedMemory(key_base+2, size=shm_mem_size) #, flags=sysv_ipc.IPC_CREAT)
            self.shm_res = sysv_ipc.SharedMemory(key_base+3, size=shm_mem_size) #, flags=sysv_ipc.IPC_CREAT)
        except sysv_ipc.Error as exp:
            assert(False)


class DrpInfo:
    def __init__(self, detector_name, detector_segment, worker_num, pebble_bufsize, transition_bufsize, ipc_info):
        self.det_name = detector_name
        self.det_segment = detector_segment
        self.worker_num = worker_num
        self.pebble_bufsize = pebble_bufsize
        self.transition_bufsize = transition_bufsize
        self.ipc_info = ipc_info

ipc_info = IPCInfo(key_base, shm_mem_size)
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
