import sys
import sysv_ipc
import importlib

worker_num = int(sys.argv[1])
key_base =int( sys.argv[2])
mem_size = int(sys.argv[3])

class IPC:

    def __init__(self, key_base, mem_size):
        try:
            self.mq_inp = sysv_ipc.MessageQueue(key_base, sysv_ipc.IPC_CREAT)
            self.mq_res = sysv_ipc.MessageQueue(key_base+1, sysv_ipc.IPC_CREAT)
        except sysv_ipc.Error as exp:
            assert(False)
        try:
            self.shm_inp = sysv_ipc.SharedMemory(key_base+2, size=mem_size, flags=sysv_ipc.IPC_CREAT)
            self.shm_res = sysv_ipc.SharedMemory(key_base+3, size=mem_size, flags=sysv_ipc.IPC_CREAT)
        except sysv_ipc.Error as exp:
            assert(False)


ipc = IPC(key_base, mem_size)

print(f"[Worker {worker_num}]: DRP Python started. Waiting for script to run")

while True:
    message, priority = ipc.mq_inp.receive()
    if message == "stop":
        exit(0)
    with open(message, "r") as fh:
       code = compile(fh.read(), message, 'exec')
       exec(code, globals(), locals())
