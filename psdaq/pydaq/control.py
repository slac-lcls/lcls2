#!/usr/bin/env python

"""
control - Control level

Author: Chris Ford <caf@slac.stanford.edu>
"""

import zmq
from Collection import Collection
from ControlTransition import ControlTransition as Transition
from ControlMsg import ControlMsg
from CollectMsg import CollectMsg
from psp import PV
from os import getpid
from socket import gethostname
import pprint
import pyca
import logging
import argparse
import time
from psana.dgrammanager import DgramManager
from psana import dgram
from transitions import Machine, MachineError, State

class ControlState(object):

    def __init__(self, pv_base):

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

        return

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

    def unconfiguredEnter(self):
        self.pv_put(self.pvRun, 0)
        return

    def configuredEnter(self):
        self.pv_put(self.pvRun, 0)
        return

    def runningEnter(self):
        self.pv_put(self.pvRun, 1)
        return

    def enabledEnter(self):
        self.pv_put(self.pvRun, 1)
        return

    def configureBefore(self):
        self.pv_put(self.pvMsgConfig, 0)
        self.pv_put(self.pvMsgConfig, 1)
        self.pv_put(self.pvMsgConfig, 0)
        return

    def unconfigureBefore(self):
        self.pv_put(self.pvMsgUnconfig, 0)
        self.pv_put(self.pvMsgUnconfig, 1)
        self.pv_put(self.pvMsgUnconfig, 0)
        return

    def enableBefore(self):
        self.pv_put(self.pvMsgEnable, 0)
        self.pv_put(self.pvMsgEnable, 1)
        self.pv_put(self.pvMsgEnable, 0)
        return

    def disableBefore(self):
        self.pv_put(self.pvMsgDisable, 0)
        self.pv_put(self.pvMsgDisable, 1)
        self.pv_put(self.pvMsgDisable, 0)
        return

    # getTrigger - returns the trigger function for <request>, or else None
    def getTrigger(self, request):
        return getattr(self, request, None)

def main():

    # Process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('pvbase', help='EPICS PV base (e.g. DAQ:LAB2:PART:2)')
    parser.add_argument('-p', type=int, choices=range(0, 8), default=0, help='platform (default 0)')
    parser.add_argument('-C', metavar='CM_HOST', default='localhost', help='Collection Manager host')
    parser.add_argument('-u', metavar='UNIQUE_ID', default='control', help='Name')
    parser.add_argument('-v', action='store_true', help='be verbose')
    args = parser.parse_args()

    if args.v:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info('control level starting')

    ctx = zmq.Context()

    coll = Collection(ctx, args.C, args.p)

    pybody = {}
    pybody['host'] = gethostname()
    pybody['pid'] = getpid()
    idbody = {}
    idbody['procInfo']=pybody
    mainbody = {}
    mainbody['control']=idbody
    hellomsg = CollectMsg(key=CollectMsg.HELLO, body=mainbody)
    partition = coll.partitionInfo(hellomsg)
    pprint.pprint(partition.body)

    # set up our end of connections, potentially based on the information
    # about who is in the partition (e.g. number of eb/drp nodes)
    # control sockets (ephemeral ports)
    control_router_socket = ctx.socket(zmq.ROUTER)
    control_pull_socket = ctx.socket(zmq.PULL)
    control_router_socket.bind("tcp://*:*")
    control_pull_socket.bind("tcp://*:*")
    control_router_port = Collection.parse_port(control_router_socket.getsockopt(zmq.LAST_ENDPOINT))
    control_pull_port = Collection.parse_port(control_pull_socket.getsockopt(zmq.LAST_ENDPOINT))
    logging.debug('control_router_port = %d' % control_router_port)
    logging.debug('control_pull_port = %d' % control_pull_port)

    pybody = {}
    pybody['router_port'] = {'adrs': gethostname(), 'port': control_router_port}
    pybody['pull_port'] =   {'adrs': gethostname(), 'port': control_pull_port}
    connbody = {}
    connbody['connectInfo']=pybody
    mainbody = {}
    mainbody['control']=connbody

    portsmsg = CollectMsg(key=CollectMsg.CONNECTINFO, body=mainbody)
    connect_info = coll.connectionInfo(portsmsg)
    pprint.pprint(connect_info.body)

    # now make the connections and report to CM when done

    # Control state

    states = ['UNCONFIGURED', 'CONFIGURED', 'RUNNING', 'ENABLED']
    states = [
        State(name='UNCONFIGURED',  on_enter='unconfiguredEnter'),
        State(name='CONFIGURED',    on_enter='configuredEnter'),
        State(name='RUNNING',       on_enter='runningEnter'),
        State(name='ENABLED',       on_enter='enabledEnter')
    ]
    transitions = [
        { 'trigger': 'CONFIGURE',    'before': 'configureBefore',
          'source':  'UNCONFIGURED', 'dest':   'CONFIGURED'},
        { 'trigger': 'UNCONFIGURE',  'before': 'unconfigureBefore',
          'source':  'CONFIGURED',   'dest':   'UNCONFIGURED'},
        { 'trigger': 'BEGINRUN',
          'source':  'CONFIGURED',   'dest':   'RUNNING'},
        { 'trigger': 'ENDRUN',
          'source':  'RUNNING',      'dest':   'CONFIGURED'},
        { 'trigger': 'ENABLE',       'before': 'enableBefore',
          'source':  'RUNNING',      'dest':   'ENABLED'},
        { 'trigger': 'DISABLE',      'before': 'disableBefore',
          'source':  'ENABLED',      'dest':   'RUNNING'}
    ]
    controlState = ControlState(args.pvbase)
    controlMachine = Machine(controlState, states, transitions=transitions,
                             initial='UNCONFIGURED',
                             ignore_invalid_triggers=True)
    logging.debug("Initial controlState.state: %s" % controlState.state)

    poller = zmq.Poller()
    poller.register(control_router_socket, zmq.POLLIN)
    poller.register(control_pull_socket, zmq.POLLIN)
    try:
        while True:
            items = dict(poller.poll(1000))

            # Handle control_pull_socket socket
            if control_pull_socket in items:
                msg = control_pull_socket.recv()
                config = dgram.Dgram(view=msg)
                # now it's in dgram.Dgram object
                ttt = config.seq.timestamp()
                print('Timestamp:', ttt)    # FIXME

            # Execute state command request
            if control_router_socket in items:
                msg = control_router_socket.recv_multipart()
                identity = msg[0]
                request = msg[1]
                logging.debug('Received <%s> from control_router_socket' % request)

                if request in [ Transition.configure, Transition.beginrun,
                                Transition.enable, Transition.disable,
                                Transition.endrun, Transition.unconfigure,
                                ControlMsg.GETSTATE]:

                    if request != ControlMsg.GETSTATE:
                        # translate request to state machine trigger, then call it
                        try:
                            decoded = request.decode()
                        except UnicodeDecodeError as ex:
                            logging.error('decode(): %s' % ex)
                            controlTrigger = None
                        else:
                            controlTrigger = controlState.getTrigger(decoded)

                        if controlTrigger:
                            try:
                                controlTrigger()
                            except MachineError as ex:
                                logging.error('Transition failed: %s' % ex)
                        else:
                            logging.debug('No trigger found for request <%s>' %
                                          request)

                    # Send reply to client
                    controlmsg = ControlMsg(identity=identity,
                                            key=controlState.state.encode())
                    controlmsg.send(control_router_socket)
                    continue

                else:
                    logging.warning("Unknown msg <%s>" % request)
                    # Send reply to client
                    logging.debug("Sending <HUH?> reply")
                    controlmsg = ControlMsg(identity=identity,
                                            key=ControlMsg.HUH)
                    controlmsg.send(control_router_socket)
                    continue

    except KeyboardInterrupt:
        logging.debug("Interrupt received")

    # Clean up
    logging.debug("Clean up control level")

    # Close all sockets associated with this context, and then
    # terminate the context.
    ctx.destroy(0)

    logging.info('control level exiting')

if __name__ == '__main__':
    main()
