#!/usr/bin/env python

"""
DTI Proxy
"""

import os
import time
import copy
import socket
import zmq
from psdaq.control.collection import pull_port, pub_port, create_msg
from psdaq.control.collection import DaqControl
import argparse
import logging
from psp import PV
import pyca

class Client:

    def __init__(self, platform, pv_base, collectHost):

        # initialize state
        self.state = 'reset'

        # establish id
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        self.id = hash(self.hostname+str(self.pid))

        # configure zmq sockets
        self.context = zmq.Context(1)
        self.push = self.context.socket(zmq.PUSH)
        self.sub = self.context.socket(zmq.SUB)
        self.push.connect('tcp://%s:%d' % (collectHost, pull_port(platform)))
        self.sub.connect('tcp://%s:%d' % (collectHost, pub_port(platform)))
        self.sub.setsockopt(zmq.SUBSCRIBE, b'')

        # initialize PVs
        self.pvMsgClear = PV(pv_base+':MsgClear', initialize=True)
        logging.debug("Create PV: %s" % self.pvMsgClear.name)
        self.pvMsgHeader = PV(pv_base+':MsgHeader', initialize=True)
        logging.debug("Create PV: %s" % self.pvMsgHeader.name)
        self.pvMsgInsert = PV(pv_base+':MsgInsert', initialize=True)
        logging.debug("Create PV: %s" % self.pvMsgInsert.name)
        self.pvRun = PV(pv_base+':Run', initialize=True)
        logging.debug("Create PV: %s" % self.pvRun.name)

        # define commands
        handle_request = {
            'reset': self.handle_reset,
            'plat': self.handle_plat,
            'alloc': self.handle_alloc,
            'connect': self.handle_connect,
            'configure': self.handle_configure,
            'enable': self.handle_enable,
            'disable': self.handle_disable,
            'unconfigure': self.handle_unconfigure
        }

        # process messages
        while True:
            try:
                msg = self.sub.recv_json()
                key = msg['header']['key']
                handle_request[key](msg)
            except KeyError as ex:
                logging.debug('KeyError: %s' % ex)

    @staticmethod
    def pv_put(pv, val):
        retval = False
        if not pv.isinitialized:
            logging.error("PV not initialized: %s" % pv.name)
        elif not pv.isconnected:
            logging.error("PV not connected: %s" % pv.name)
        else:
            try:
                pv.put(val)
            except pyca.pyexc:
                logging.error("PV put(%d) timeout: %s" % (val, pv.name))
            else:
                retval = True
                logging.debug("PV put(%d): %s" % (val, pv.name))
        return retval

    def handle_plat(self, msg):
        logging.debug('Client handle_plat()')
        # time.sleep(1.5)
        body = {'drp-no-teb': {'proc_info': {
                        'host': self.hostname,
                        'pid': self.pid}}}
        reply = create_msg('plat', msg['header']['msg_id'], self.id, body=body)
        self.push.send_json(reply)

    def handle_alloc(self, msg):
        logging.debug('Client handle_alloc()')
        body = {'drp-no-teb': {'connect_info': {
                           'infiniband': '172.21.52.122'
                       }}}
        reply = create_msg('alloc', msg['header']['msg_id'], self.id, body)
        self.push.send_json(reply)
        self.state = 'allocated'

    def handle_connect(self, msg):
        logging.debug('Client handle_connect()')
        if self.state == 'allocated':
            self.pv_put(self.pvRun, 0)  # clear Run PV before configure
            self.state = 'connected'
            reply = create_msg('ok', msg['header']['msg_id'], self.id)
            self.push.send_json(reply)

    def handle_configure(self, msg):
        logging.debug('Client handle_configure()')
        if self.state == 'connected':
            # check for errors
            if (self.pv_put(self.pvMsgClear, 0) and
                self.pv_put(self.pvMsgClear, 1) and 
                self.pv_put(self.pvMsgClear, 0) and
                self.pv_put(self.pvMsgHeader, DaqControl.transitionId['Configure']) and
                self.pv_put(self.pvMsgInsert, 0) and
                self.pv_put(self.pvMsgInsert, 1) and 
                self.pv_put(self.pvMsgInsert, 0)):
                # success: change state and reply 'ok'
                self.state = 'paused'
                reply = create_msg('ok', msg['header']['msg_id'], self.id)
            else:
                # failure: reply 'error'
                err_msg = "configure: PV put failed"
                logging.error(err_msg)
                node = 'drp-no-teb/%s/%s' % (self.pid, self.hostname)
                body = {'err_info': { node : err_msg}}
                reply = create_msg('error', msg['header']['msg_id'], self.id,
                                   body=body)
            self.push.send_json(reply)
        else:
            logging.error('handle_configure() invalid state: %s' % self.state)

    def handle_unconfigure(self, msg):
        logging.debug('Client handle_unconfigure()')
        if self.state == 'paused':
            # check for errors
            if (self.pv_put(self.pvMsgHeader, DaqControl.transitionId['Unconfigure']) and
                self.pv_put(self.pvMsgInsert, 0) and
                self.pv_put(self.pvMsgInsert, 1) and 
                self.pv_put(self.pvMsgInsert, 0)):
                # success: change state and reply 'ok'
                self.state = 'connected'
                reply = create_msg('ok', msg['header']['msg_id'], self.id)
            else:
                # failure: reply 'error'
                err_msg = "unconfigure: PV put failed"
                logging.error(err_msg)
                node = 'drp-no-teb/%s/%s' % (self.pid, self.hostname)
                body = {'err_info': { node : err_msg}}
                reply = create_msg('error', msg['header']['msg_id'], self.id,
                                   body=body)
            self.push.send_json(reply)
        else:
            logging.error('handle_unconfigure() invalid state: %s' % self.state)

    def handle_enable(self, msg):
        logging.debug('Client handle_enable()')
        if self.state == 'paused':
            # check for errors
            if (self.pv_put(self.pvMsgHeader, DaqControl.transitionId['Enable']) and
                self.pv_put(self.pvMsgInsert, 0) and
                self.pv_put(self.pvMsgInsert, 1) and 
                self.pv_put(self.pvMsgInsert, 0) and
                self.pv_put(self.pvRun, 1)):
                # success: change state and reply 'ok'
                self.state = 'running'
                reply = create_msg('ok', msg['header']['msg_id'], self.id)
            else:
                # failure: reply 'error'
                err_msg = "enable: PV put failed"
                logging.error(err_msg)
                node = 'drp-no-teb/%s/%s' % (self.pid, self.hostname)
                body = {'err_info': { node : err_msg}}
                reply = create_msg('error', msg['header']['msg_id'], self.id,
                                   body=body)
            self.push.send_json(reply)
        else:
            logging.error('handle_enable() invalid state: %s' % self.state)

    def handle_disable(self, msg):
        logging.debug('Client handle_disable()')
        if self.state == 'running':
            # check for errors
            if (self.pv_put(self.pvMsgHeader, DaqControl.transitionId['Disable']) and
                self.pv_put(self.pvMsgInsert, 0) and
                self.pv_put(self.pvMsgInsert, 1) and 
                self.pv_put(self.pvMsgInsert, 0) and
                self.pv_put(self.pvRun, 0)):
                # success: change state and reply 'ok'
                self.state = 'paused'
                reply = create_msg('ok', msg['header']['msg_id'], self.id)
            else:
                # failure: reply 'error'
                err_msg = "disable: PV put failed"
                logging.error(err_msg)
                node = 'drp-no-teb/%s/%s' % (self.pid, self.hostname)
                body = {'err_info': { node : err_msg}}
                reply = create_msg('error', msg['header']['msg_id'], self.id,
                                   body=body)
            self.push.send_json(reply)
        else:
            logging.error('handle_disable() invalid state: %s' % self.state)

    def handle_reset(self, msg):
        logging.debug('Client handle_reset()')
        self.state = 'reset'
        # is a reply to reset necessary?


def main():

    try:
        # process arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('pvbase', help='EPICS PV base (e.g. DAQ:LAB2:PART:2)')
        parser.add_argument('-p', type=int, choices=range(0, 8), default=0, help='platform (default 0)')
        parser.add_argument('-C', metavar='COLLECT_HOST', default='localhost', help='collection host')
        parser.add_argument('-v', action='store_true', help='be verbose')
        args = parser.parse_args()

        # configure logging
        if args.v:
            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        else:
            logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

        # start client
        client = Client(args.p, args.pvbase, args.C)

    except KeyboardInterrupt:
        logging.info('KeyboardInterrupt')

if __name__ == '__main__':
    main()
