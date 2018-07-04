#!/usr/bin/env python

"""
control - Control level

Author: Chris Ford <caf@slac.stanford.edu>
"""

import zmq
#import Collection
from Collection import Collection
from ControlTransition import ControlTransition as Transition
from ControlMsg import ControlMsg
from CollectMsg import CollectMsg
from ControlState import ControlState, StateMachine
from psp import PV
from os import getpid
from socket import gethostname
import pprint
import pyca
import logging
import argparse
import time
import zmq.utils.jsonapi as json
from psana.dgrammanager import DgramManager
from psana import dgram

class ControlStateMachine(StateMachine):

    # Define states
    state_unconfigured = ControlState(b'UNCONFIGURED')
    state_configured = ControlState(b'CONFIGURED')
    state_running = ControlState(b'RUNNING')
    state_enabled = ControlState(b'ENABLED')

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

        # register callbacks for each valid state+transition combination

        unconfigured_dict = {
            Transition.configure: (self.configfunc, self.state_configured)
        }

        configured_dict = {
            Transition.unconfigure: (self.unconfigfunc,
                                          self.state_unconfigured),
            Transition.beginrun: (self.beginrunfunc, self.state_running)
        }

        running_dict = {
            Transition.endrun: (self.endrunfunc, self.state_configured),
            Transition.enable: (self.enablefunc, self.state_enabled)
        }

        enabled_dict = {
            Transition.disable: (self.disablefunc, self.state_running)
        }

        self.state_unconfigured.register(unconfigured_dict)
        self.state_configured.register(configured_dict)
        self.state_running.register(running_dict)
        self.state_enabled.register(enabled_dict)

        # Start with a default state.
        self._state = self.state_unconfigured

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

    def configfunc(self):
        logging.debug("configfunc()")
        # PV MsgConfig=0
        # PV MsgConfig=1
        # PV MsgConfig=0
        # PV Run=0
        return (self.pv_put(self.pvMsgConfig, 0) and
                self.pv_put(self.pvMsgConfig, 1) and
                self.pv_put(self.pvMsgConfig, 0) and
                self.pv_put(self.pvRun, 0))

    def unconfigfunc(self):
        logging.debug("unconfigfunc()")
        # PV MsgUnconfig=0
        # PV MsgUnconfig=1
        # PV MsgUnconfig=0
        # PV Run=0
        return (self.pv_put(self.pvMsgUnconfig, 0) and
                self.pv_put(self.pvMsgUnconfig, 1) and
                self.pv_put(self.pvMsgUnconfig, 0) and
                self.pv_put(self.pvRun, 0))

    def beginrunfunc(self):
        logging.debug("beginrunfunc()")
        # PV Run=1
        return self.pv_put(self.pvRun, 1)

    def endrunfunc(self):
        logging.debug("endrunfunc()")
        # PV Run=0
        return self.pv_put(self.pvRun, 0)

    def enablefunc(self):
        logging.debug("enablefunc()")
        # PV MsgEnable=0
        # PV MsgEnable=1
        # PV MsgEnable=0
        # PV Run=1
        return (self.pv_put(self.pvMsgEnable, 0) and
                self.pv_put(self.pvMsgEnable, 1) and
                self.pv_put(self.pvMsgEnable, 0) and
                self.pv_put(self.pvRun, 1))

    def disablefunc(self):
        logging.debug("disablefunc()")
        # PV MsgDisable=0
        # PV MsgDisable=1
        # PV MsgDisable=0
        # PV Run=1
        return (self.pv_put(self.pvMsgDisable, 0) and
                self.pv_put(self.pvMsgDisable, 1) and
                self.pv_put(self.pvMsgDisable, 0) and
                self.pv_put(self.pvRun, 1))

def test_ControlStateMachine():

    # ControlStateMachine tests

    yy = ControlStateMachine('DAQ:LAB2:PART:2')
    print("ControlStateMachine state:", yy.state())
    assert(yy.state() == yy.state_unconfigured)

    yy.on_transition(Transition.configure)
    print("ControlStateMachine state:", yy.state())
    assert(yy.state() == yy.state_configured)

    yy.on_transition(Transition.beginrun)
    print("ControlStateMachine state:", yy.state())
    assert(yy.state() == yy.state_running)

    yy.on_transition(Transition.enable)
    print("ControlStateMachine state:", yy.state())
    assert(yy.state() == yy.state_enabled)

    yy.on_transition(Transition.disable)
    print("ControlStateMachine state:", yy.state())
    assert(yy.state() == yy.state_running)

    yy.on_transition(Transition.endrun)
    print("ControlStateMachine state:", yy.state())
    assert(yy.state() == yy.state_configured)

    yy.on_transition(Transition.unconfigure)
    print("ControlStateMachine state:", yy.state())
    assert(yy.state() == yy.state_unconfigured)

    print("ControlStateMachine OK")

    return

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
    hellomsg = CollectMsg(key=CollectMsg.HELLO, body=json.dumps(mainbody))
    partition = coll.partitionInfo(hellomsg)
    pprint.pprint(json.loads(partition.body))

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

    portsmsg = CollectMsg(key=CollectMsg.PORTS, body=json.dumps(mainbody))
    connect_info = coll.connectionInfo(portsmsg)
    pprint.pprint(json.loads(connect_info.body))

    # now make the connections and report to CM when done

    # Control state
    yy = ControlStateMachine(args.pvbase)
    logging.debug("ControlStateMachine state: %s" % yy.state())


    sequence = 0

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
                logging.debug('Received <%s> from control_router_socket' % request.decode())

                if request == ControlMsg.PING:
                    # Send reply to client
                    logging.debug("Sending <PONG> reply")
                    control_router_socket.send(identity, zmq.SNDMORE)
                    cmmsg = ControlMsg(sequence, key=ControlMsg.PONG)
                    cmmsg.send(control_router_socket)
                    continue

                if request == ControlMsg.PONG:
                    continue

                if request in [ Transition.configure, Transition.beginrun,
                                Transition.enable, Transition.disable,
                                Transition.endrun, Transition.unconfigure,
                                ControlMsg.GETSTATE]:

                    if request != ControlMsg.GETSTATE:
                        oldstate = yy.state()
                        # Do transition
                        yy.on_transition(request)
                        newstate = yy.state()
                        if newstate != oldstate:
                            logging.debug("ControlStateMachine state: %s" % newstate)

                    # Send reply to client
                    control_router_socket.send(identity, zmq.SNDMORE)
                    cmmsg = ControlMsg(sequence, key=yy.state().key())
                    cmmsg.send(control_router_socket)
                    continue

                else:
                    logging.warning("Unknown msg <%s>" % request.decode())
                    # Send reply to client
                    logging.debug("Sending <HUH?> reply")
                    control_router_socket.send(identity, zmq.SNDMORE)
                    cmmsg = ControlMsg(sequence, key=ControlMsg.HUH)
                    cmmsg.send(control_router_socket)
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
