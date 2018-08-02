import os
import time
import copy
import socket
from uuid import uuid4
import zmq

PORT_BASE = 29980


def create_msg(key, msg_id=None, sender_id=None, body={}):
    if msg_id is None:
        msg_id = str(uuid4())
    msg = {'header': {
               'key': key,
               'msg_id': msg_id,
               'sender_id': sender_id},
           'body': body}
    return msg


def pull_port(platform):
    return PORT_BASE + platform


def pub_port(platform):
    return PORT_BASE + platform + 10


def rep_port(platform):
    return PORT_BASE + platform + 20


def wait_for_answers(socket, wait_time, msg_id):
    """
    Wait and return all messages from socket that match msg_id
    Parameters
    ----------
    socket: zmq socket
    wait_time: int, wait time in milliseconds
    msg_id: int, expected msg_id of received messages
    """
    remaining = wait_time
    start = time.time()
    while socket.poll(remaining) == zmq.POLLIN:
        msg = socket.recv_json()
        if msg['header']['msg_id'] == msg_id:
            yield msg
        else:
            print('unexpected msg_id: got %s but expected %s' %
                  (msg['header']['msg_id'], msg_id))
        remaining = max(0, int(wait_time - 1000*(time.time() - start)))


def confirm_response(socket, wait_time, msg_id, ids):
    msgs = []
    for msg in wait_for_answers(socket, wait_time, msg_id):
        msgs.append(msg)
        ids.remove(msg['header']['sender_id'])
        if len(ids) == 0:
            break
    return len(ids), msgs


class CollectionManager():
    def __init__(self, platform):
        self.context = zmq.Context(1)
        self.pull = self.context.socket(zmq.PULL)
        self.pub = self.context.socket(zmq.PUB)
        self.rep = self.context.socket(zmq.REP)
        self.pull.bind('tcp://*:%d' % pull_port(platform))
        self.pub.bind('tcp://*:%d' % pub_port(platform))
        self.rep.bind('tcp://*:%d' % rep_port(platform))
        self.cmstate = {}
        self.ids = set()
        self.handle_request = {
            'plat': self.handle_plat,
            'alloc': self.handle_alloc,
            'connect': self.handle_connect
        }
        # start main loop
        self.run()

    def run(self):
        while True:
            msg = self.rep.recv_json()
            key = msg['header']['key']
            answer = self.handle_request[key]()
            self.rep.send_json(answer)

    def handle_plat(self):
        self.cmstate.clear()
        self.ids.clear()
        msg = create_msg('plat')
        self.pub.send_json(msg)
        for answer in wait_for_answers(self.pull, 1000, msg['header']['msg_id']):
            for level, item in answer['body'].items():
                if level not in self.cmstate:
                    self.cmstate[level] = {}
                id = answer['header']['sender_id']
                self.cmstate[level][id] = item
                self.ids.add(id)
        return create_msg('ok', body=self.cmstate)

    def handle_alloc(self):
        # FIXME select all procs for now
        ids = copy.copy(self.ids)
        msg = create_msg('alloc', body={'ids': list(ids)})
        self.pub.send_json(msg)

        # make sure all the clients respond to alloc message with their connection info
        ret, answers = confirm_response(self.pull, 1000, msg['header']['msg_id'], ids)

        if ret:
            return create_msg('error', body={'error': '%d client did not respond' % ret})
        for answer in answers:
            id = answer['header']['sender_id']
            for level, item in answer['body'].items():
                self.cmstate[level][id].update(item)
        print('cmstate after alloc:')
        print(self.cmstate)
        return create_msg('ok')

    def handle_connect(self):
        # FIXME select all procs for now
        ids = copy.copy(self.ids)
        msg = create_msg('connect', body=self.cmstate)
        self.pub.send_json(msg)

        ret, answers = confirm_response(self.pull, 5000, msg['header']['msg_id'], ids)
        if ret:
            return create_msg('error', body={'error': '%d client did not respond' % ret})
        else:
            return create_msg('ok')


class Client:
    def __init__(self, platform):
        self.context = zmq.Context(1)
        self.push = self.context.socket(zmq.PUSH)
        self.sub = self.context.socket(zmq.SUB)
        self.push.connect('tcp://localhost:%d' % pull_port(platform))
        self.sub.connect('tcp://localhost:%d' % pub_port(platform))
        self.sub.setsockopt(zmq.SUBSCRIBE, b'')
        self.state = ''
        handle_request = {
            'plat': self.handle_plat,
            'alloc': self.handle_alloc,
            'connect': self.handle_connect
        }
        while True:
            msg = self.sub.recv_json()
            key = msg['header']['key']
            handle_request[key](msg)
            if key == 'connect':
                break

    def handle_plat(self, msg):
        # time.sleep(1.5)
        hostname = socket.gethostname()
        pid = os.getpid()
        self.id = hash(hostname+str(pid))
        body = {'drp': {'proc_info': {
                        'host': hostname,
                        'pid': pid}}}
        reply = create_msg('plat', msg['header']['msg_id'], self.id, body=body)
        self.push.send_json(reply)

    def handle_alloc(self, msg):
        body = {'drp': {'connect_info': {'infiniband': '123.456.789'}}}
        reply = create_msg('alloc', msg['header']['msg_id'], self.id, body)
        self.push.send_json(reply)
        self.state = 'alloc'

    def handle_connect(self, msg):
        if self.state == 'alloc':
            reply = create_msg('ok', msg['header']['msg_id'], self.id)
            self.push.send_json(reply)


if __name__ == '__main__':
    from multiprocessing import Process

    def manager():
        manager = CollectionManager(0)

    def client(i):
        c = Client(0)

    procs = [Process(target=manager)]
    for i in range(2):
        pass
        procs.append(Process(target=client, args=(i,)))

    for p in procs:
        p.start()

    # Commands
    platform = 0
    context = zmq.Context(1)
    req = context.socket(zmq.REQ)
    req.connect('tcp://localhost:%d' % rep_port(platform))
    time.sleep(0.5)

    msg = create_msg('plat')
    req.send_json(msg)
    print('Answer:', req.recv_multipart())

    msg = create_msg('alloc')
    req.send_json(msg)
    print('Answer:', req.recv_multipart())

    msg = create_msg('connect')
    req.send_json(msg)
    print('Answer:', req.recv_multipart())

    for p in procs:
        p.join()
