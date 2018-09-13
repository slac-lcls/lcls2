import os
import time
import copy
import socket
from uuid import uuid4
import zmq
from transitions import Machine, MachineError, State
import argparse
import logging

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
            logging.error('unexpected msg_id: got %s but expected %s' %
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
            'getstate': self.handle_getstate,
        }
        self.triggers = ['plat', 'alloc', 'connect',
                         'configure', 'unconfigure',
                         'beginrun', 'endrun',
                         'enable', 'disable', 'reset']
        self.states = [
            'reset',
            'plat',
            'alloc',
            'connect',
            'configured',
            'running',
            'enabled',
            'error'
        ]

        self.collectMachine = Machine(self, self.states, initial='reset')

        self.collectMachine.add_transition('reset', '*', 'reset',
                                           conditions='condition_reset')
        self.collectMachine.add_transition('plat', ['reset', 'plat'], 'plat',
                                           conditions='condition_plat')
        self.collectMachine.add_transition('alloc', 'plat', 'alloc',
                                           conditions='condition_alloc')
        self.collectMachine.add_transition('connect', 'alloc', 'connect',
                                           conditions='condition_connect')
        self.collectMachine.add_transition('configure', 'connect', 'configured',
                                           conditions='condition_configure')
        self.collectMachine.add_transition('beginrun', 'configured', 'running',
                                           conditions='condition_beginrun')
        self.collectMachine.add_transition('endrun', 'running', 'configured',
                                           conditions='condition_endrun')
        self.collectMachine.add_transition('enable', 'running', 'enabled',
                                           conditions='condition_enable')
        self.collectMachine.add_transition('disable', 'enabled', 'running',
                                           conditions='condition_disable')
        self.collectMachine.add_transition('unconfigure', 'configured', 'connect',
                                           conditions='condition_unconfigure')

        logging.info('Initial state = %s' % self.state)

        # start main loop
        self.run()

    def run(self):
        try:
            while True:
                answer = None
                try:
                    msg = self.rep.recv_json()
                    key = msg['header']['key']
                    if key in self.triggers:
                        answer = self.handle_trigger(key)
                    else:
                        answer = self.handle_request[key]()
                except KeyError:
                    answer = create_msg('error')
                if answer is not None:
                    self.rep.send_json(answer)
        except KeyboardInterrupt:
            logging.info('KeyboardInterrupt')

    def handle_trigger(self, key):
        logging.debug('handle_trigger(\'%s\') in state %s' % (key, self.state))
        stateBefore = self.state
        trigError = None
        try:
            self.trigger(key)
        except MachineError as ex:
            logging.debug('MachineError: %s' % ex)
            trigError = str(ex)
        else:
            # check for error: trigger failed to change the state
            if (self.state == stateBefore):
                # two exceptions OK: reset->reset and plat->plat
                if ((self.state == key == 'reset') or
                    (self.state == key == 'plat')):
                    pass
                else:
                    # enter the error state
                    self.to_error()
                    trigError = '%s failed, entered error state' % key

        if trigError is None:
            answer = create_msg(self.state, body=self.cmstate)
        else:
            errMsg = trigError.replace("\"", "")
            logging.error(errMsg)
            answer = create_msg(self.state, body={'error': errMsg})

        return answer

    def condition_alloc(self):
        # FIXME select all procs for now
        ids = copy.copy(self.ids)
        msg = create_msg('alloc', body={'ids': list(ids)})
        self.pub.send_json(msg)

        # make sure all the clients respond to alloc message with their connection info
        ret, answers = confirm_response(self.pull, 1000, msg['header']['msg_id'], ids)
        if ret:
            logging.error('%d client did not respond to alloc' % ret)
            logging.debug('condition_alloc() returning False')
            return False
        for answer in answers:
            id = answer['header']['sender_id']
            for level, item in answer['body'].items():
                self.cmstate[level][id].update(item)

        # give number to drp nodes for the event builder
        if 'drp' in self.cmstate:
            for i, node in enumerate(self.cmstate['drp']):
                self.cmstate['drp'][node]['drp_id'] = i

        # give number to teb nodes for the event builder
        if 'teb' in self.cmstate:
            for i, node in enumerate(self.cmstate['teb']):
                self.cmstate['teb'][node]['teb_id'] = i

        # give number to meb nodes for the event builder
        if 'meb' in self.cmstate:
            for i, node in enumerate(self.cmstate['meb']):
                self.cmstate['meb'][node]['meb_id'] = i

        logging.debug('cmstate after alloc:\n%s' % self.cmstate)
        logging.debug('condition_alloc() returning True')
        return True

    def condition_connect(self):
        # FIXME select all procs for now
        ids = copy.copy(self.ids)
        msg = create_msg('connect', body=self.cmstate)
        self.pub.send_json(msg)

        ret, answers = confirm_response(self.pull, 5000, msg['header']['msg_id'], ids)
        if ret:
            logging.error('%d client did not respond to connect' % ret)
            logging.debug('condition_connect() returning False')
            return False
        else:
            logging.debug('condition_connect() returning True')
            return True

    def handle_getstate(self):
        return create_msg(self.state, body=self.cmstate)

    def on_enter_reset(self):
        self.cmstate.clear()
        self.ids.clear()
        return

    def condition_plat(self):
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
        # should a nonempty platform be required for successful transition?
        logging.debug('condition_plat() returning True')
        return True

    def condition_common(self, transition, timeout):
        retval = True
        ids = copy.copy(self.ids)
        msg = create_msg(transition)
        self.pub.send_json(msg)

        # make sure all the clients respond to transition before timeout
        ret, answers = confirm_response(self.pull, timeout, msg['header']['msg_id'], ids)
        if ret:
            # Error
            retval = False
            logging.error('%d client did not respond to %s' % (ret, transition))
        else:
            retval = True
            for answer in answers:
                try:
                    for node, err_msg in answer['body']['err_info'].items():
                        # Error
                        retval = False
                        logging.error('%s: %s' % (node, err_msg))
                except KeyError:
                    pass
            return retval

    def condition_configure(self):
        retval = self.condition_common('configure', 1000)
        logging.debug('condition_configure() returning %s' % retval)
        return retval

    def condition_unconfigure(self):
        retval = self.condition_common('unconfigure', 1000)
        logging.debug('condition_unconfigure() returning %s' % retval)
        return retval

    def condition_beginrun(self):
        retval = self.condition_common('beginrun', 1000)
        logging.debug('condition_beginrun() returning %s' % retval)
        return retval

    def condition_endrun(self):
        retval = self.condition_common('endrun', 1000)
        logging.debug('condition_endrun() returning %s' % retval)
        return retval

    def condition_enable(self):
        retval = self.condition_common('enable', 1000)
        logging.debug('condition_enable() returning %s' % retval)
        return retval

    def condition_disable(self):
        retval = self.condition_common('disable', 1000)
        logging.debug('condition_disable() returning %s' % retval)
        return retval

    def condition_reset(self):
        # is a reply to reset necessary?
        msg = create_msg('reset')
        self.pub.send_json(msg)
        logging.debug('condition_reset() returning True')
        return True

class Client:
    def __init__(self, platform):
        self.context = zmq.Context(1)
        self.push = self.context.socket(zmq.PUSH)
        self.sub = self.context.socket(zmq.SUB)
        self.push.connect('tcp://localhost:%d' % pull_port(platform))
        self.sub.connect('tcp://localhost:%d' % pub_port(platform))
        self.sub.setsockopt(zmq.SUBSCRIBE, b'')
        handle_request = {
            'plat': self.handle_plat,
            'alloc': self.handle_alloc,
            'connect': self.handle_connect
        }
        while True:
            try:
                msg = self.sub.recv_json()
                key = msg['header']['key']
                handle_request[key](msg)
            except KeyError as ex:
                logging.debug('KeyError: %s' % ex)

            if key == 'connect':
                break

    def handle_plat(self, msg):
        logging.debug('Client handle_plat()')
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
        logging.debug('Client handle_alloc()')
        body = {'drp': {'connect_info': {'infiniband': '123.456.789'}}}
        reply = create_msg('alloc', msg['header']['msg_id'], self.id, body)
        self.push.send_json(reply)
        self.state = 'alloc'

    def handle_connect(self, msg):
        logging.debug('Client handle_connect()')
        if self.state == 'alloc':
            reply = create_msg('ok', msg['header']['msg_id'], self.id)
            self.push.send_json(reply)


if __name__ == '__main__':
    from multiprocessing import Process

    # Process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=int, choices=range(0, 8), default=0, help='platform (default 0)')
    parser.add_argument('-a', action='store_true', help='autoconnect')
    parser.add_argument('-v', action='store_true', help='be verbose')
    args = parser.parse_args()
    platform = args.p

    if args.v:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

    def manager():
        manager = CollectionManager(platform)

    def client(i):
        c = Client(platform)

    procs = [Process(target=manager)]
    for i in range(2):
        # procs.append(Process(target=client, args=(i,)))
        pass

    for p in procs:
        p.start()
        pass

    if args.a:
        # Commands
        context = zmq.Context(1)
        req = context.socket(zmq.REQ)
        req.connect('tcp://localhost:%d' % rep_port(platform))
        time.sleep(0.5)

        msg = create_msg('plat')
        req.send_json(msg)
        print('Answer to plat:', req.recv_multipart())

        msg = create_msg('alloc')
        req.send_json(msg)
        print('Answer to alloc:', req.recv_multipart())

        msg = create_msg('connect')
        req.send_json(msg)
        print('Answer to connect:', req.recv_multipart())

    for p in procs:
        try:
            p.join()
        except KeyboardInterrupt:
            pass
