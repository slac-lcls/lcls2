#!/usr/bin/env python

"""
DTI Proxy
"""

import os
import time
import copy
import socket
from uuid import uuid4
import zmq
from collection import pull_port, pub_port, create_msg
import argparse
import logging
from psp import PV
import pyca

class Client:

    def __init__(self, platform, pv_base):

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
        self.push.connect('tcp://localhost:%d' % pull_port(platform))
        self.sub.connect('tcp://localhost:%d' % pub_port(platform))
        self.sub.setsockopt(zmq.SUBSCRIBE, b'')

        # initialize PVs
        self.pvMsgEnable = PV(pv_base+':MsgEnable', initialize=True)
        logging.debug("Create PV: %s" % self.pvMsgEnable.name)
        self.pvMsgDisable = PV(pv_base+':MsgDisable', initialize=True)
        logging.debug("Create PV: %s" % self.pvMsgDisable.name)
        self.pvMsgConfig = PV(pv_base+':MsgConfig', initialize=True)
        logging.debug("Create PV: %s" % self.pvMsgConfig.name)
        self.pvMsgUnconfig = PV(pv_base+':MsgUnconfig', initialize=True)
        logging.debug("Create PV: %s" % self.pvMsgUnconfig.name)
        self.pvRun = PV(pv_base+':Run', initialize=True)
        logging.debug("Create PV: %s" % self.pvRun.name)

        # define commands
        handle_request = {
            'reset': self.handle_reset,
            'plat': self.handle_plat,
            'alloc': self.handle_alloc,
            'connect': self.handle_connect,
            'configure': self.handle_configure,
            'beginrun': self.handle_beginrun,
            'enable': self.handle_enable,
            'disable': self.handle_disable,
            'endrun': self.handle_endrun,
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
        body = {'dti': {'proc_info': {
                        'host': self.hostname,
                        'pid': self.pid}}}
        reply = create_msg('plat', msg['header']['msg_id'], self.id, body=body)
        self.push.send_json(reply)

    def handle_alloc(self, msg):
        logging.debug('Client handle_alloc()')
        body = {'dti': {'connect_info': {'infiniband': '123.456.789'}}}
        reply = create_msg('alloc', msg['header']['msg_id'], self.id, body)
        self.push.send_json(reply)
        self.state = 'alloc'

    def handle_connect(self, msg):
        logging.debug('Client handle_connect()')
        if self.state == 'alloc':
            self.state = 'connect'
            reply = create_msg('ok', msg['header']['msg_id'], self.id)
            self.push.send_json(reply)

    def handle_configure(self, msg):
        logging.debug('Client handle_configure()')
        if self.state == 'connect':
            # check for errors
            if (self.pv_put(self.pvMsgConfig, 0) and
                self.pv_put(self.pvMsgConfig, 1) and 
                self.pv_put(self.pvMsgConfig, 0)):
                # success: change state and reply 'ok'
                self.state = 'configured'
                reply = create_msg('ok', msg['header']['msg_id'], self.id)
            else:
                # failure: reply 'error'
                err_msg = "PV put failed: %s" % self.pvMsgConfig.name
                logging.error(err_msg)
                node = 'dti/%s/%s' % (self.pid, self.hostname)
                body = {'err_info': { node : err_msg}}
                reply = create_msg('error', msg['header']['msg_id'], self.id,
                                   body=body)
            self.push.send_json(reply)
        else:
            logging.error('handle_configure() invalid state: %s' % self.state)

    def handle_unconfigure(self, msg):
        if self.state == 'configured':
            # check for errors
            if (self.pv_put(self.pvMsgUnconfig, 0) and
                self.pv_put(self.pvMsgUnconfig, 1) and 
                self.pv_put(self.pvMsgUnconfig, 0)):
                # success: change state and reply 'ok'
                self.state = 'connect'
                reply = create_msg('ok', msg['header']['msg_id'], self.id)
            else:
                # failure: reply 'error'
                err_msg = "PV put failed: %s" % self.pvMsgUnconfig.name
                logging.error(err_msg)
                node = 'dti/%s/%s' % (self.pid, self.hostname)
                body = {'err_info': { node : err_msg}}
                reply = create_msg('error', msg['header']['msg_id'], self.id,
                                   body=body)
            self.push.send_json(reply)
        else:
            logging.error('handle_unconfigure() invalid state: %s' % self.state)

    def handle_beginrun(self, msg):
        logging.debug('Client handle_beginrun()')
        if self.state == 'configured':
            # check for errors
            if (self.pv_put(self.pvRun, 1)):
                # success: change state and reply 'ok'
                self.state = 'running'
                reply = create_msg('ok', msg['header']['msg_id'], self.id)
            else:
                # failure: reply 'error'
                err_msg = "PV put failed: %s" % self.pvRun.name
                logging.error(err_msg)
                node = 'dti/%s/%s' % (self.pid, self.hostname)
                body = {'err_info': { node : err_msg}}
                reply = create_msg('error', msg['header']['msg_id'], self.id,
                                   body=body)
            self.push.send_json(reply)
        else:
            logging.error('handle_beginrun() invalid state: %s' % self.state)

    def handle_endrun(self, msg):
        logging.debug('Client handle_endrun()')
        if self.state == 'running':
            # check for errors
            if (self.pv_put(self.pvRun, 0)):
                # success: change state and reply 'ok'
                self.state = 'configured'
                reply = create_msg('ok', msg['header']['msg_id'], self.id)
            else:
                # failure: reply 'error'
                err_msg = "PV put failed: %s" % self.pvRun.name
                logging.error(err_msg)
                node = 'dti/%s/%s' % (self.pid, self.hostname)
                body = {'err_info': { node : err_msg}}
                reply = create_msg('error', msg['header']['msg_id'], self.id,
                                   body=body)
            self.push.send_json(reply)
        else:
            logging.error('handle_endrun() invalid state: %s' % self.state)

    def handle_enable(self, msg):
        logging.debug('Client handle_enable()')
        if self.state == 'running':
            # check for errors
            if (self.pv_put(self.pvMsgEnable, 0) and
                self.pv_put(self.pvMsgEnable, 1) and
                self.pv_put(self.pvMsgEnable, 0)):
                # success: change state and reply 'ok'
                self.state = 'enabled'
                reply = create_msg('ok', msg['header']['msg_id'], self.id)
            else:
                # failure: reply 'error'
                err_msg = "PV put failed: %s" % self.pvMsgEnable.name
                logging.error(err_msg)
                node = 'dti/%s/%s' % (self.pid, self.hostname)
                body = {'err_info': { node : err_msg}}
                reply = create_msg('error', msg['header']['msg_id'], self.id,
                                   body=body)
            self.push.send_json(reply)
        else:
            logging.error('handle_enable() invalid state: %s' % self.state)

    def handle_disable(self, msg):
        logging.debug('Client handle_disable()')
        if self.state == 'enabled':
            # check for errors
            if (self.pv_put(self.pvMsgDisable, 0) and
                self.pv_put(self.pvMsgDisable, 1) and
                self.pv_put(self.pvMsgDisable, 0)):
                # success: change state and reply 'ok'
                self.state = 'running'
                reply = create_msg('ok', msg['header']['msg_id'], self.id)
            else:
                # failure: reply 'error'
                err_msg = "PV put failed: %s" % self.pvMsgDisable.name
                logging.error(err_msg)
                node = 'dti/%s/%s' % (self.pid, self.hostname)
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


if __name__ == '__main__':

    try:
        # process arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('pvbase', help='EPICS PV base (e.g. DAQ:LAB2:PART:2)')
        parser.add_argument('-p', type=int, choices=range(0, 8), default=0, help='platform (default 0)')
        parser.add_argument('-v', action='store_true', help='be verbose')
        args = parser.parse_args()

        # configure logging
        if args.v:
            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        else:
            logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

        # start client
        client = Client(args.p, args.pvbase)

    except KeyboardInterrupt:
        logging.info('KeyboardInterrupt')
