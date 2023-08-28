import zmq
import zlib, pickle
import time

class PubSocket:
    def __init__(self, socket_name):
        self._context = zmq.Context()
        self._zmq_socket = self._context.socket(zmq.PUB)
        self._zmq_socket.bind(socket_name)

    def send(self, data):
        return self.send_zipped_pickle(data)

    def sendz(self, zdata, flags=0):
        self._zmq_socket.send(zdata, flags=flags)

    def send_zipped_pickle(self, obj, flags=0, protocol=-1):
        """pickle an object, and zip the pickle before sending it"""
        p = pickle.dumps(obj, protocol)
        z = zlib.compress(p)
        self._zmq_socket.send(z, flags=flags)
        return z


class SubSocket:
    def __init__(self, socket_name):
        self._context = zmq.Context()
        self._zmq_socket = self._context.socket(zmq.SUB)
        self._zmq_socket.connect(socket_name)

        # Subscribe to all
        topicfilter = ""
        self._zmq_socket.setsockopt_string(zmq.SUBSCRIBE, topicfilter)

    def recv(self):
        return self.recvz()[1]

    def recvz(self):
        st = time.time()
        z,data = self.recv_zipped_pickle()
        en = time.time()
        print(f"Subscriber recv took:{en-st:.2f}s.")
        return z,data

    def recv_zipped_pickle(self, flags=0, protocol=-1):
        """inverse of send_zipped_pickle"""
        z = self._zmq_socket.recv(flags)
        p = zlib.decompress(z)
        return z,pickle.loads(p)
