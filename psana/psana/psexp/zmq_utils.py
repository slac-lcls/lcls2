import zmq
import zlib, pickle
import time


client_socket = None
def zmq_send(**kwargs):
    global client_socket
    if client_socket is None:
        client_socket = SubSocket(kwargs['fake_dbase_server'], socket_type=zmq.PUSH)
    data = {}
    for key, val in kwargs.items():
        if key == 'fake_dbase_server': continue
        data[key] = val
    client_socket.send(data)
    print(f"sent {data} to {kwargs['fake_dbase_server']}")


def send_zipped_pickle(zmq_socket, obj, flags=0, protocol=-1):
    """pickle an object, and zip the pickle before sending it"""
    p = pickle.dumps(obj, protocol)
    z = zlib.compress(p)
    zmq_socket.send(z, flags=flags)
    return z


def recv_zipped_pickle(zmq_socket, flags=0, protocol=-1):
    """unzip and unpickle received data"""
    z = zmq_socket.recv(flags)
    p = zlib.decompress(z)
    return z,pickle.loads(p)


class PubSocket:
    """ A helper for a Binder Zmq-Socket
    """
    def __init__(self, socket_name, socket_type=zmq.PUB):
        self._context = zmq.Context()
        self._zmq_socket = self._context.socket(socket_type)
        self._zmq_socket.bind(socket_name)

    def send(self, data):
        return send_zipped_pickle(self._zmq_socket, data)

    def sendz(self, zdata, flags=0):
        self._zmq_socket.send(zdata, flags=flags)

    def recv(self):
        return self.recvz()[1]

    def recvz(self):
        z,data = recv_zipped_pickle(self._zmq_socket)
        return z,data


class SubSocket:
    """ A helper for a Connector Zmq-Socket
    """
    def __init__(self, socket_name, socket_type=zmq.SUB):
        self._context = zmq.Context()
        self._zmq_socket = self._context.socket(socket_type)
        self._zmq_socket.connect(socket_name)

        # Subscribe to all
        if socket_type == zmq.SUB:
            topicfilter = ""
            self._zmq_socket.setsockopt_string(zmq.SUBSCRIBE, topicfilter)
    
    def send(self, data):
        return send_zipped_pickle(self._zmq_socket, data)

    def sendz(self, zdata, flags=0):
        self._zmq_socket.send(zdata, flags=flags)

    def recv(self):
        return self.recvz()[1]

    def recvz(self):
        z,data = recv_zipped_pickle(self._zmq_socket)
        return z,data

