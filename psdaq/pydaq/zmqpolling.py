import zmq
import time
import sys
import random, os
# from  multiprocessing import Process
from threading import Thread
import random
import queue

from psana.dgrammanager import DgramManager
from psana import dgram

def server_pull(port, recv_queue):
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind("tcp://*:%s" % port)

    poller = zmq.Poller()
    messages = []
    poller.register(socket, zmq.POLLIN)
    while True:
        socks = dict(poller.poll(1000))
        if socket in socks and socks[socket] == zmq.POLLIN:
            message = socket.recv()
            recv_queue.put(message)
        if socks == {}:
            print("Poller timed out")
            return 

def client(port_push, dgq=[]):
    context = zmq.Context()
    socket_push = context.socket(zmq.PUSH)
    socket_push.connect("tcp://localhost:%s" % port_push)
    print("Connected to server with port %s" % port_push)

    ct = 0
    while True:
        if dgq.empty():
            return True
        evt_data = dgq.get()
        socket_push.send(evt_data) 
def unpack_dgrams(recv_queue):

    lq = list(recv_queue.queue)
    # find the configure
    for ct, dgrm in enumerate(lq):
        if int(dgrm[0]) == 2:
            config_ind = ct
            break

    # with open('config.xtc', 'wb') as f:
    #     f.write(lq[config_ind][1:])

    # fd = os.open('config.xtc', os.O_RDONLY)
    # config = dgram.Dgram(file_descriptor=fd)

    config = dgram.Dgram(view = lq[config_ind][1:])

    dgrams = [config]
    for ct,dgrm in enumerate(lq):
        if ct == config_ind:
            continue
        dgr = dgram.Dgram(view = dgrm[1:], config = config)
        dgrams.append(dgr)


    timestamps = [dgr.seq.timestamp() for dgr in dgrams]
    _, dgrams = zip(*sorted(zip(timestamps, dgrams)))

    return dgrams


if __name__ == "__main__":
    input = 'data-00.xtc'

    ds = DgramManager(input)
    dgram_queue = queue.Queue()
    recv_queue = queue.Queue()

    for ct, evt in enumerate(ds):
        if ct == 0:
            dgrm = ds.configs[0]
            serv = 2
        elif ct>10:
            break
        else:
            dgrm = evt.dgrams[0]
            serv = 0
        evt_bytes = memoryview(dgrm).tobytes()
        msg = chr(serv).encode() + evt_bytes
        dgram_queue.put(msg)

    # Now we can run a few servers
    server_push_port = '5555'
    server = Thread(target=server_pull, args=(server_push_port, recv_queue))

    clients = []
    for i in range(4):
        clients.append(Thread(target=client, args=(server_push_port,\
                                                   dgram_queue)))

    server.start()
    [t.start() for t in clients]

    server.join()
    [t.join() for t in clients]

    dgrams = unpack_dgrams(recv_queue)

