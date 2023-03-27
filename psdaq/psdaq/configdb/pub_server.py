import zmq
import random
import sys
import time
import zlib, pickle

# Global socket
socket = None

def pub_bind(socket_name):
    context = zmq.Context()
    global socket
    socket = context.socket(zmq.PUB)
    #socket.bind(f"ipc:///tmp/{socket_name}.pipe")
    socket.bind(socket_name)
    return socket

def pub_send(calib_const):
    send_zipped_pickle(socket, calib_const)

def send_zipped_pickle(zmq_socket, obj, flags=0, protocol=-1):
    """pickle an object, and zip the pickle before sending it"""
    p = pickle.dumps(obj, protocol)
    z = zlib.compress(p)
    return zmq_socket.send(z, flags=flags)

if __name__ == "__main__":
    port = "5556"
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    socket = pub_bind(port)
    pub_send(socket)
