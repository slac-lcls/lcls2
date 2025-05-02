
import zmq

client_socket = None

def zmq_send(**kwargs):
    global client_socket
    if client_socket is None:
        client_socket = SubSocket(kwargs["fake_dbase_server"], socket_type=zmq.PUSH)
    data = {}
    for key, val in kwargs.items():
        if key == "fake_dbase_server":
            continue
        data[key] = val
    client_socket.send(data)
    print(f"sent {data} to {kwargs['fake_dbase_server']}")


class ZMQSocket:
    def __init__(self, zmq_socket):
        self.socket = zmq_socket

    def send(self, data):
        self.socket.send_pyobj(data)

    def recv(self):
        return self.socket.recv_pyobj()


class PubSocket(ZMQSocket):
    """A helper for a Binder Zmq-Socket"""

    def __init__(self, socket_name, socket_type=zmq.PUB):
        context = zmq.Context()
        zmq_socket = context.socket(socket_type)
        zmq_socket.bind(socket_name)
        super(PubSocket, self).__init__(zmq_socket)


class SubSocket(ZMQSocket):
    """A helper for a Connector Zmq-Socket"""

    def __init__(self, socket_name, socket_type=zmq.SUB):
        context = zmq.Context()
        zmq_socket = context.socket(socket_type)
        zmq_socket.connect(socket_name)
        super(SubSocket, self).__init__(zmq_socket)

        # Subscribe to all
        if socket_type == zmq.SUB:
            topicfilter = ""
            self.socket.setsockopt_string(zmq.SUBSCRIBE, topicfilter)


class SrvSocket(ZMQSocket):
    def __init__(self, socket_name):
        context = zmq.Context()
        zmq_socket = context.socket(zmq.REP)
        zmq_socket.bind(socket_name)
        super(SrvSocket, self).__init__(zmq_socket)


class ClientSocket(ZMQSocket):
    def __init__(self, socket_name):
        context = zmq.Context()
        zmq_socket = context.socket(zmq.REQ)
        zmq_socket.connect(socket_name)
        super(ClientSocket, self).__init__(zmq_socket)
