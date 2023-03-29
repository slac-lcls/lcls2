import zmq
import zlib, pickle
import time
calibconst_socket = None

def pub_bind(socket_name):
    context = zmq.Context()
    global calibconst_socket
    calibconst_socket = context.socket(zmq.PUB)
    #calibconst_socket.bind(f"ipc:///tmp/{socket_name}.pipe")
    calibconst_socket.bind(socket_name)
    return calibconst_socket

def pub_send(calib_const):
    send_zipped_pickle(calibconst_socket, calib_const)

def send_zipped_pickle(zmq_socket, obj, flags=0, protocol=-1):
    """pickle an object, and zip the pickle before sending it"""
    p = pickle.dumps(obj, protocol)
    z = zlib.compress(p)
    zmq_socket.send(z, flags=flags)

def sub_connect(socket_name):
    # Socket to talk to server
    context = zmq.Context()
    global calibconst_socket
    calibconst_socket = context.socket(zmq.SUB)
    #calibconst_socket.connect(f"ipc:///tmp/{socket_name}.pipe")
    calibconst_socket.connect(socket_name)

    # Subscribe to all
    topicfilter = ""
    calibconst_socket.setsockopt_string(zmq.SUBSCRIBE, topicfilter)
    return calibconst_socket

def sub_recv():
    st = time.time()
    calib_const = recv_zipped_pickle(calibconst_socket)
    en = time.time()
    print(f"Subscriber recv took:{en-st:.2f}s.")
    return calib_const

def recv_zipped_pickle(zmq_socket, flags=0, protocol=-1):
    """inverse of send_zipped_pickle"""
    z = zmq_socket.recv(flags)
    p = zlib.decompress(z)
    return pickle.loads(p)
