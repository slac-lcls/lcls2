#!/usr/bin/env python

"""
Test Client
"""

import os
import time
import copy
import socket
import zmq
from psdaq.control.control import back_pull_port, back_pub_port, create_msg
import argparse
import logging
from psdaq.control.syslog import SysLog
import zmq.utils.jsonapi as json

class Client:
    def __init__(self, platform, collectHost, alias):

        # initialize state
        self.state = 'reset'

        # establish id
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        self.id = hash(self.hostname+str(self.pid))
        self.alias = alias

        # configure zmq sockets
        self.context = zmq.Context(1)
        self.push = self.context.socket(zmq.PUSH)
        self.sub = self.context.socket(zmq.SUB)
        self.push.connect('tcp://%s:%d' % (collectHost, back_pull_port(platform)))
        self.sub.connect('tcp://%s:%d' % (collectHost, back_pub_port(platform)))
        self.sub.setsockopt(zmq.SUBSCRIBE, b'')

        # define commands
        handle_request = {
            'reset': self.handle_reset,
            'rollcall': self.handle_rollcall,
            'alloc': self.handle_alloc,
            'connect': self.handle_connect,
            'disconnect': self.handle_disconnect,
            'configure': self.handle_configure,
            'beginrun': self.handle_beginrun,
            'beginstep': self.handle_beginstep,
            'enable': self.handle_enable,
            'disable': self.handle_disable,
            'endstep': self.handle_endstep,
            'endrun': self.handle_endrun,
            'unconfigure': self.handle_unconfigure
        }

        # process messages
        while True:
            try:
                topic, rawmsg = self.sub.recv_multipart()
            except Exception as ex1:
                logging.error('recv_multipart() exception: %s' % ex1)
            else:
                logging.debug('topic=<%s>' % topic)
                try:
                    msg = json.loads(rawmsg)
                except Exception as ex2:
                    logging.error('json.loads() exception: %s' % ex2)
                else:
                    key = msg['header']['key']
                    handle_request[key](msg)

    def handle_rollcall(self, msg):
        logging.debug('Client handle_rollcall(msg_id=\'%s\')' % msg['header']['msg_id'])
        # time.sleep(1.5)
        body = {'drp': {'proc_info': {
                        'alias': self.alias,
                        'host': self.hostname,
                        'pid': self.pid}}}
        reply = create_msg('rollcall', msg['header']['msg_id'], self.id, body=body)
        self.push.send_json(reply)

    def handle_alloc(self, msg):
        logging.debug('Client handle_alloc(msg_id=\'%s\')' % msg['header']['msg_id'])
        body = {'drp': {'connect_info': {'infiniband': '123.456.789'}}}
        reply = create_msg('alloc', msg['header']['msg_id'], self.id, body)
        self.push.send_json(reply)
        self.state = 'allocated'

    def handle_connect(self, msg):
        logging.debug('Client handle_connect(msg_id=\'%s\')' % msg['header']['msg_id'])
        if self.state == 'allocated':
            self.state = 'connected'
            reply = create_msg('ok', msg['header']['msg_id'], self.id)
            self.push.send_json(reply)

    def handle_disconnect(self, msg):
        logging.debug('Client handle_disconnect(msg_id=\'%s\')' % msg['header']['msg_id'])
        if self.state == 'connected':
            self.state = 'allocated'
            body = {'err_info': 'This is only a test'}
            reply = create_msg('disconnect', msg['header']['msg_id'], self.id, body)
            self.push.send_json(reply)

    def handle_configure(self, msg):
        logging.debug('Client handle_configure(msg_id=\'%s\')' % msg['header']['msg_id'])
        if self.state == 'connected':
            self.state = 'paused'
            reply = create_msg('ok', msg['header']['msg_id'], self.id)
            self.push.send_json(reply)

    def handle_unconfigure(self, msg):
        logging.debug('Client handle_unconfigure(msg_id=\'%s\')' % msg['header']['msg_id'])
        if self.state == 'paused':
            self.state = 'connected'
            reply = create_msg('ok', msg['header']['msg_id'], self.id)
            self.push.send_json(reply)

    def handle_enable(self, msg):
        logging.debug('Client handle_enable(msg_id=\'%s\')' % msg['header']['msg_id'])
        if self.state == 'paused':
            self.state = 'running'
            reply = create_msg('ok', msg['header']['msg_id'], self.id)
            self.push.send_json(reply)

    def handle_disable(self, msg):
        logging.debug('Client handle_disable(msg_id=\'%s\')' % msg['header']['msg_id'])
        if self.state == 'running':
            self.state = 'paused'
            reply = create_msg('ok', msg['header']['msg_id'], self.id)
            self.push.send_json(reply)

    def handle_beginstep(self, msg):
        logging.debug('Client handle_beginstep(msg_id=\'%s\')' % msg['header']['msg_id'])
        if True:
            reply = create_msg('ok', msg['header']['msg_id'], self.id)
            self.push.send_json(reply)

    def handle_endstep(self, msg):
        logging.debug('Client handle_endstep(msg_id=\'%s\')' % msg['header']['msg_id'])
        if True:
            reply = create_msg('ok', msg['header']['msg_id'], self.id)
            self.push.send_json(reply)

    def handle_beginrun(self, msg):
        logging.debug('Client handle_beginrun(msg_id=\'%s\')' % msg['header']['msg_id'])
        if True:
            reply = create_msg('ok', msg['header']['msg_id'], self.id)
            self.push.send_json(reply)

    def handle_endrun(self, msg):
        logging.debug('Client handle_endrun(msg_id=\'%s\')' % msg['header']['msg_id'])
        if True:
            reply = create_msg('ok', msg['header']['msg_id'], self.id)
            self.push.send_json(reply)

    def handle_reset(self, msg):
        logging.debug('Client handle_reset(msg_id=\'%s\')' % msg['header']['msg_id'])
        self.state = 'reset'
        # is a reply to reset necessary?

def main():

    try:
        # process arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('-p', type=int, choices=range(0, 8), default=0, help='platform (default 0)')
        parser.add_argument('-C', metavar='COLLECT_HOST', default='localhost', help='collection host (default localhost)')
        parser.add_argument('-P', metavar='INSTRUMENT', default='tst', help='instrument (default tst)')
        parser.add_argument('-u', metavar='ALIAS', required=True, help='unique ID')
        parser.add_argument('-v', action='store_true', help='be verbose')
        args = parser.parse_args()

        # configure logging
        if args.v:
            level = logging.DEBUG
        else:
            level = logging.WARNING
        logger = SysLog(instrument=args.P, level=level)
        logging.info('logging configured')

        # start client
        client = Client(args.p, args.C, args.u)

    except KeyboardInterrupt:
        logging.info('KeyboardInterrupt')

if __name__ == '__main__':
    main()
