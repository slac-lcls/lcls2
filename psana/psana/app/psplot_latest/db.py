######################################################################################
# Simple Database for storing psplot process detail created by psplotdb server.
# RunInstance:
#    PK 
#   {1: slurm_job_id1, rixc00221, 49, sdfmilan032, 12301, pid, DbHistoryStatus.PLOTTED,
#    2: slurm_job_id2, rixc00221, 50, sdfmilan032, 12301, pid, DbHistoryStatus.PLOTTED,
#    3: slurm_job_id3, rixc00221, 49, sdfmilan032, 12301, pid, DbHistoryStatus.RECEIVED,
#
######################################################################################

from psana.psexp.zmq_utils import PubSocket
import socket
import zmq

class DbHistoryStatus():
    RECEIVED = 0
    PLOTTED = 1

class DbHistoryColumns():
    SLURM_JOB_ID = 0
    EXP = 1
    RUNNUM = 2
    NODE = 3
    PORT = 4
    PID = 5
    STATUS = 6

class DbHelper():
    def __init__(self):
        # Keep a history of received request:
        # - Instance stores a list of instance ids for a exp, runnum, node, port key
        # - History stores a detail (pid, slurm jobid, status, etc) for an instance id
        self.instance = {} 

    def connect(self, socket_name):
        self.srv_socket = PubSocket(socket_name, socket_type=zmq.PULL)
        print(f'Run psana with this socket: {socket_name}', flush=True)

    @staticmethod
    def get_socket(port=None):
        # We acquire an available port using a socket then closing it immediately 
        # so that this port can be used later.
        IPAddr = socket.gethostbyname(socket.gethostname())
        if port is None:
            sock = socket.socket()
            sock.bind(('', 0))
            port = sock.getsockname()[1]
            sock.close()
        supervisor_ip_addr = f'{IPAddr}:{port}'
        socket_name = f"tcp://{supervisor_ip_addr}"
        return socket_name

    def get_db_info(self):
        # Fetch data 
        print(f'Waiting for client...')
        obj = self.srv_socket.recv()
        return obj 

    def set(self, instance_id, what, val):
        self.instance[instance_id][what] = val

    def save(self, obj):
        next_id = 1
        if self.instance:
            ids = list(self.instance.keys())
            next_id = max(ids) + 1
        self.instance[next_id] = [obj['slurm_job_id'],
                obj['exp'],
                obj['runnum'],
                obj['node'],
                obj['port'],
                None,
                DbHistoryStatus.RECEIVED]
        return next_id

    def get(self, instance_id):
        found_instance = None
        if instance_id in self.instance:
            found_instance = self.instance[instance_id]
        return found_instance

