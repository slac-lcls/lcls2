import sys
import zmq
import time
import zlib, pickle

socket = None

def sub_connect(socket_name):
    # Socket to talk to server
    context = zmq.Context()
    global socket
    socket = context.socket(zmq.SUB)
    #socket.connect(f"ipc:///tmp/{socket_name}.pipe")
    socket.connect(socket_name)

    # Subscribe to all
    topicfilter = ""
    socket.setsockopt_string(zmq.SUBSCRIBE, topicfilter)
    return socket

def sub_recv():
    st = time.time()
    calib_const = recv_zipped_pickle(socket)
    en = time.time()
    print(f"Subscriber recv took:{en-st:.2f}s.")
    return calib_const

def recv_zipped_pickle(zmq_socket, flags=0, protocol=-1):
    """inverse of send_zipped_pickle"""
    z = zmq_socket.recv(flags)
    p = zlib.decompress(z)
    return pickle.loads(p)

if __name__ == "__main__":
    ipaddr = "localhost"
    port = "5556"
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    socket = sub_connect(ipaddr, port)
    sub_recv(socket)

