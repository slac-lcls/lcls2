from psana.psexp.zmq_utils import PubSocket
import socket
import zmq

class DbHistoryStatus():
    RECEIVED = 0
    PLOTTED = 1

class DbHistoryColumns():
    STATUS = 0
    PID = 1
    SLURM_JOB_ID = 2

class DbHelper():
    def __init__(self):
        # Keep a history of received request:
        # - Instance stores a list of instance ids for a exp, runnum, node, port key
        # - History stores a detail (pid, slurm jobid, status, etc) for an instance id
        self.instance = {} 
        self.history = {}

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

    def set(self, key, instance_id, values):
        self.history[key, instance_id] = values

    def save(self, obj):
        key = (obj['exp'], obj['runnum'], obj['node'], obj['port'])
        next_id = 0
        if key in self.instance:
            ids = self.instance[key] 
            next_id = max(ids) + 1
        else:
            self.instance[key] = []
        self.instance[key] += [next_id]

        self.history[key, next_id] = [DbHistoryStatus.RECEIVED, None, None]
        return key, next_id

    def get(self, what, query_string):
        found_key, found_instance_id = None, None
        for key, val in self.history.items():
            if str(val[what]) == str(query_string):
                found_key, found_instance_id = key[0], key[1]
                break
        return found_key, found_instance_id

