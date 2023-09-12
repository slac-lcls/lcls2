from psana.psexp.zmq_utils import PubSocket
import socket
import zmq

class DbHelper():
    def __init__(self):
        # We acquire an available port using a socket then closing it immediately 
        # so that this port can be used later.
        IPAddr = socket.gethostbyname(socket.gethostname())
        sock = socket.socket()
        sock.bind(('', 0))
        port_no = sock.getsockname()[1]
        sock.close()
        supervisor_ip_addr = f'{IPAddr}:{port_no}'
        socket_name = f"tcp://{supervisor_ip_addr}"
        print(f'Run psana with this socket: {socket_name}', flush=True)
        self.srv_socket = PubSocket(socket_name, socket_type=zmq.PULL)

    def get_db_info(self):
        # Fetch data 
        print(f'Getting data from client')
        obj = self.srv_socket.recv()
        return obj 
