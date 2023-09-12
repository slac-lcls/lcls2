from psana.psexp.zmq_utils import PubSocket
import socket
import zmq

class DbHelper():
    def __init__(self, port=None):
        IPAddr = socket.gethostbyname(socket.gethostname())
        # We acquire an available port using a socket then closing it immediately 
        # so that this port can be used later.
        if port is None:
            sock = socket.socket()
            sock.bind(('', 0))
            port = sock.getsockname()[1]
            sock.close()
        supervisor_ip_addr = f'{IPAddr}:{port}'
        socket_name = f"tcp://{supervisor_ip_addr}"
        print(f'Run psana with this socket: {socket_name}', flush=True)
        self.srv_socket = PubSocket(socket_name, socket_type=zmq.PULL)

    def get_db_info(self):
        # Fetch data 
        print(f'Waiting for client...')
        obj = self.srv_socket.recv()
        return obj 
