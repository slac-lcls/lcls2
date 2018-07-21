#!/usr/bin/env python

"""
collectMgr - Collection Manager

Author: Chris Ford <caf@slac.stanford.edu>
"""

import zmq
import sys
import time
import logging
import pprint
import argparse
import copy
from transitions import Machine, MachineError, State

from CollectMsg import CollectMsg

class CollectState(object):

    entries = {}
    identityDict = {}
    UNASSIGNED = "Unassigned"

    def __init__(self, platform):
        self._platform = platform
        self._partName = self.UNASSIGNED

    def noplatEnter(self, identity):
        # Reset the state
        self.reset()
        return

    def platEnter(self, identity):
        # Reset the state to avoid duplicates in case PLAT is repeated
        self.reset()
        # Send PLAT broadcast
        logging.debug("Sending PLAT broadcast")
        platmsg = CollectMsg(key=CollectMsg.PLAT)
        platmsg.send(self.publisher)
        return

    def allocEnter(self, identity):
        logging.debug("Sending ALLOC individually")
        for level,nodes in self.entries.items():
            for nodeid,node in enumerate(nodes):
                # map to the identity
                # could also create a reverse-mapping dict
                who = (level,nodeid)
                for identityKey,item in self.identityDict.items():
                    if item==who:
                        break
                    # should raise error if not found
                body = copy.copy(self.entries)
                body['id'] = nodeid
                allocMsg = CollectMsg(identity=identityKey,
                                      key=CollectMsg.ALLOC,
                                      body=body)
                allocMsg.send(self.cmd)
                logging.debug("...sent ALLOC")
        return

    def nonEmptyVerify(self, identity):
        retval = (len(self.entries) > 0)
        if not retval:
            logging.error("Platform is empty")
        return retval

    def connectInfoVerify(self, identity):
        errFlag = False
        for level,nodes in self.entries.items():
            for nodeid,node in enumerate(nodes):
                if 'connectInfo' not in node:
                    errFlag = True
                    logging.error("%s%d: Missing connectInfo" % (level, nodeid))
        return not errFlag

    def procInfoVerify(self, identity):
        errFlag = False
        for level,nodes in self.entries.items():
            for nodeid,node in enumerate(nodes):
                if 'procInfo' not in node:
                    errFlag = True
                    logging.error("%s%d: Missing procInfo" % (level, nodeid))
        return not errFlag

    def connectEnter(self, identity):
        # Send CONNECT individually
        logging.debug("Sending CONNECT individually")
        for identity in self.identityDict.keys():
            connectMsg = CollectMsg(identity=identity,
                                    key=CollectMsg.CONNECT,
                                    body=self.entries)
            connectMsg.send(self.cmd)
            logging.debug("...sent CONNECT")
        return

    def killBefore(self, identity):
        # Send KILL broadcast
        logging.debug("Sending KILL broadcast")
        cmmsg = CollectMsg(key=CollectMsg.KILL)
        cmmsg.send(self.publisher)
        return

    # getTrigger - returns the trigger function for <request>,
    #              or None is no trigger is found.
    def getTrigger(self, request):
        return getattr(self, request, None)

    def reset(self):
        # Unassign the partition name
        self._partName = self.UNASSIGNED

        # Remove all of the nodes
        self.entries = {}

        # Clear the identity dictionary
        self.identityDict = {}
        return

    def partName(self):
        return self._partName

    def platform(self):
        return self._platform

# Runs self test of CollectState class

def test_collectState (verbose=1):
    collectState0 = CollectState(0)
    collectState1 = CollectState(1)
    collectState2 = CollectState(2, 20)

def main():

    # Process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=int, choices=range(0, 8), default=0, help='platform (default 0)')
    parser.add_argument('-v', action='store_true', help='be verbose')
    args = parser.parse_args()

    if args.v:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info('collectMgr starting')

    # Collect state

    states = [
        State(name='NOPLAT',  on_enter='noplatEnter'),
        State(name='PLAT',    on_enter='platEnter'),
        State(name='ALLOC',   on_enter='allocEnter'),
        State(name='CONNECT', on_enter='connectEnter')
    ]
    transitions = [
        { 'trigger': 'PLAT',
          'source':  ['NOPLAT', 'PLAT'], 'dest':   'PLAT'},
        { 'trigger': 'ALLOC',
          'conditions': ['nonEmptyVerify', 'procInfoVerify'],
          'source':  'PLAT',             'dest':   'ALLOC'},
        { 'trigger': 'CONNECT',          'conditions': 'connectInfoVerify',
          'source':  'ALLOC',            'dest':   'CONNECT'},
        { 'trigger': 'KILL',             'before': 'killBefore',
          'source':  states,             'dest':   'NOPLAT'}
    ]
    collectState = CollectState(platform=args.p)
    collectMachine = Machine(collectState, states, transitions=transitions,
                             initial='NOPLAT')
    logging.debug("Initial collectState.state: %s" % collectState.state)

    # context and sockets
    ctx = zmq.Context()
    collectState.cmd = ctx.socket(zmq.ROUTER)
    collectState.cmd.bind("tcp://*:%d" % CollectMsg.router_port(collectState.platform()))
    collectState.publisher = ctx.socket(zmq.PUB)
    collectState.publisher.bind("tcp://*:%d" % CollectMsg.pub_port(collectState.platform()))

    poller = zmq.Poller()
    poller.register(collectState.cmd, zmq.POLLIN)
    try:
        while True:
            items = dict(poller.poll(1000))

            # Execute state cmd request
            if collectState.cmd in items:
                collectmsg = CollectMsg.recv(collectState.cmd)
                identity = collectmsg.identity
                request = collectmsg.key
                logging.debug('Received <%s> from cmd' % request)

                try:
                    decoded = request.decode()
                except UnicodeDecodeError as ex:
                    logging.error('decode(): %s' % ex)
                    collectTrigger = None
                else:
                    collectTrigger = collectState.getTrigger(decoded)

                if collectTrigger:
                    # Handle state transition by calling trigger
                    try:
                        collectTrigger(identity)
                    except MachineError as ex:
                        logging.error('Transition failed: %s' % ex)

                    # *fall through* to GETSTATE for reply after transition
                    request = CollectMsg.GETSTATE

                if request == CollectMsg.GETSTATE:
                    # Send state reply to client
                    logging.debug("Sending state reply")
                    statemsg = CollectMsg(identity=identity,
                                          key=collectState.state.encode(),
                                          body=collectState.entries)
                    statemsg.send(collectState.cmd)
                    continue

                elif request == CollectMsg.DIE:
                    # Send DIE broadcast
                    logging.debug("Sending DIE broadcast")
                    cmmsg = CollectMsg(key=CollectMsg.DIE)
                    cmmsg.send(collectState.publisher)

                    # Send DIESTARTED reply to client
                    logging.debug("Sending DIESTARTED reply")
                    cmmsg = CollectMsg(identity=identity,
                                       key=CollectMsg.DIESTARTED)
                    cmmsg.send(collectState.cmd)
                    continue

                elif request == CollectMsg.HELLO:
                    if collectState.state not in ['PLAT']:
                        logging.warning("Dropped HELLO message (state=%s)" %
                                        collectState.state)
                        continue

                    hellodict = collectmsg.body
                    logging.debug("HELLO body = %s" % hellodict)

                    # add new entry
                    #collectState[identity] = hellodict
                    for level,item in hellodict.items():
                        keys = collectState.entries.keys()
                        if level not in keys:
                            collectState.entries[level] = []
                        collectState.identityDict[identity]=(level,len(collectState.entries[level]))
                        collectState.entries[level].append(item)
                        break

                    continue

                elif request == CollectMsg.CONNECTINFO:
                    connectInfo = collectmsg.body
                    logging.debug("CONNECTINFO body = %s" % connectInfo)
                    try:
                        level, index = collectState.identityDict[identity]
                        collectState.entries[level][index].update(connectInfo[level])
                    except Exception as ex:
                        logging.error(ex)

                    continue

                elif request == CollectMsg.DUMP:
                    # Send reply to client
                    logging.debug("Sending DUMPSTARTED reply")
                    cmmsg = CollectMsg(identity=identity,
                                       key=CollectMsg.DUMPSTARTED)
                    cmmsg.send(collectState.cmd)

                    # Dump state to console
                    print("platform:", collectState.platform())
                    print("partName:", collectState.partName())
                    print("Nodes:")
                    pprint.pprint(collectState.entries)
                    continue

                else:
                    logging.warning("Unknown msg <%s>" % request)
                    # Send reply to client
                    logging.debug("Sending <HUH?> reply")
                    cmmsg = CollectMsg(identity=identity,
                                       key=CollectMsg.HUH)
                    cmmsg.send(collectState.cmd)
                    continue

    except KeyboardInterrupt:
        logging.debug("Interrupt received")

    # Clean up
    logging.debug("Clean up")

    # terminate zmq context
    ctx.destroy()

    logging.info('collectMgr exiting')

if __name__ == '__main__':
    main()
