import sys
import posix_ipc
import logging
logger = logging.getLogger(__name__)

partition = int(sys.argv[1])
pebble_bufsize = int(sys.argv[2])
transition_bufsize = int(sys.argv[3])
shm_mem_size = int(sys.argv[4])
detector_name = sys.argv[5]
detector_type = sys.argv[6]
detector_id = sys.argv[7]
detector_segment = int(sys.argv[8])
worker_num = int(sys.argv[9])
verbose = int(sys.argv[10])

logging.basicConfig(format='%(filename)s L%(lineno)04d: <%(levelname).1s> %(message)s',
                    level=logging.INFO if verbose==0 else logging.DEBUG)


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
    def __init__(self, detector_name, detector_type, detector_id, detector_segment, worker_num, pebble_bufsize, transition_bufsize, ipc_info):
        self.det_name = detector_name
        self.det_type = detector_type
        self.det_id = detector_id
        self.det_segment = detector_segment
        self.worker_num = worker_num
        self.pebble_bufsize = pebble_bufsize
        self.transition_bufsize = transition_bufsize
        self.ipc_info = ipc_info
        self.is_supervisor = False
        self.tcp_socket_name = None
        self.ipc_socket_name = f"ipc:///tmp/{detector_name}_{detector_segment}.pipe"

ipc_info = IPCInfo(partition, detector_name, detector_segment, worker_num, shm_mem_size)
drp_info = DrpInfo(detector_name, detector_type, detector_id, detector_segment, worker_num, pebble_bufsize, transition_bufsize, ipc_info)

try:
    while True:
        logger.debug(f"[Python - Worker: {worker_num}] Python process waiting for new script to run")
        message, priority = ipc_info.mq_inp.receive()
        if message == b"s":
            ipc_info.mq_res.send(b"s\n")
            continue
        pythonScript, is_supervisor, supervisor_ip_port = message.decode().split(',')
        drp_info.is_supervisor = False if is_supervisor == '' else True
        drp_info.tcp_socket_name = f"tcp://{supervisor_ip_port}"
        logger.debug(f"[Python - Worker: {worker_num}] {supervisor_ip_port = }")
        with open(pythonScript, "r") as fh:
            code = compile(fh.read(), pythonScript, 'exec')
            exec(code, globals(), locals())
except KeyboardInterrupt:
    logger.info(f"[Python - Worker: {worker_num}] KeyboardInterrupt received - drp_python process exiting")
finally:
    ipc_info.mq_inp.close()
    ipc_info.mq_res.close()
    ipc_info.shm_inp.close_fd()
    ipc_info.shm_res.close_fd()
