import zmq
import logging
import time
from socket import gethostname
from os import getpid
from CollectMsg import CollectMsg

class Collection:
    def __init__(self, ctx, cm_host, platform):
        self.cm_host = cm_host
        self.platform = platform
        # collection mgr sockets (well known ports)
        self.dealer_socket = ctx.socket(zmq.DEALER)
        self.sub_socket = ctx.socket(zmq.SUB)
        self.dealer_socket.linger = 0
        self.sub_socket.linger = 0
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, '')
        self.dealer_socket.connect("tcp://%s:%d" % (cm_host, CollectMsg.router_port(platform)))
        self.sub_socket.connect("tcp://%s:%d" % (cm_host, CollectMsg.pub_port(platform)))
        self.dealer_port = Collection.parse_port(self.dealer_socket.getsockopt(zmq.LAST_ENDPOINT))
        self.sub_port = Collection.parse_port(self.sub_socket.getsockopt(zmq.LAST_ENDPOINT))
        logging.debug('collect_dealer_port = %d' % self.dealer_port)
        logging.debug('collect_sub_port = %d' % self.sub_port)

        self.poller = zmq.Poller()
        self.poller.register(self.dealer_socket, zmq.POLLIN)
        self.poller.register(self.sub_socket, zmq.POLLIN)

    @staticmethod
    def parse_port(inbytestring):
        try:
            retval = int(inbytestring.split(b':')[-1])
        except:
            print("Failed to parse", inbytestring)
            retval = 0
        return retval

    def partitionInfo(self, coll_msg):
        self.getMsg(CollectMsg.PLAT)
        coll_msg.send(self.dealer_socket)
        return self.getMsg(CollectMsg.ALLOC)

    def connectionInfo(self, coll_msg):
        coll_msg.send(self.dealer_socket)
        return self.getMsg(CollectMsg.CONNECT)

    def getMsg(self, msg_type):
        while True:
            items = dict(self.poller.poll(1000))

            # Collection Mgr client: DEALER socket
            if self.dealer_socket in items:
                cmmsg = CollectMsg.recv(self.dealer_socket)
                if cmmsg.key == msg_type:
                    return cmmsg

            # Collection Mgr client: SUB socket
            if self.sub_socket in items:
                cmmsg = CollectMsg.recv(self.sub_socket)
                if cmmsg.key == msg_type:
                    return cmmsg
