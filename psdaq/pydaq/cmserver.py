#!/usr/bin/env python

"""
cmserver - Collection Manager server

Author: Chris Ford <caf@slac.stanford.edu>
"""

import zmq
import sys
import time
import logging
import pickle
import pprint
import argparse

from CMMsg import CMMsg
from ZTimer import ZTimer

class CMState(object):

    entries = {}
    UNASSIGNED = "Unassigned"

    def __init__(self, platform, heartbeatInterval=10):
        self._platform = platform
        self._heartbeatInterval = heartbeatInterval
        self._nodeTimeout = 3 * heartbeatInterval
        self._partName = self.UNASSIGNED

    # dictionary access maps to entries:
    def __getitem__(self, k):
        return self.entries[k]

    def __setitem__(self, k, v):
        self.entries[k] = v

    def timestamp(self, k):
        self.entries[k]['_lastContact'] = int(time.monotonic())

    def keys(self):
        return list(self.entries.keys())

    def nodes(self):
        return list(self.entries.values())

    def reset(self):
        # Unassign the partition name
        self._partName = self.UNASSIGNED

        # Remove all of the nodes
        self.entries = {}
        return

    def partName(self):
        return self._partName

    def platform(self):
        return self._platform

    def heartbeatInterval(self):
        return self._heartbeatInterval

    def nodeTimeout(self):
        return self._nodeTimeout

    def expired_keys(self, timeout):
        rv = []
        now = int(time.monotonic())
        for k in self.entries.keys():
            try:
                if now > (self.entries[k]['_lastContact'] + timeout):
                    rv.append(k)
            except:
                pass
        return rv

    def find_duplicates(self, prop):
        rv = []
        for k in self.entries.keys():
            try:
                # check for matching ip+pid
                if ((self.entries[k]['ip'] == prop['ip']) and (self.entries[k]['pid'] == prop['pid'])):
                    rv.append(k)
            except:
                pass
        return rv

    def remove_keys(self, keylist):
        rlist = []
        for k in keylist:
            if k in self.entries:
                rlist.append("%s/%s" % (self.entries[k]['ip'], self.entries[k]['pid']))
                del self.entries[k]
        return rlist

    def __repr__(self):
        mstr = "CMState:\npartName:{partName}\nplatform:{platform}\nheartbeatInterval:{heartbeatInterval}\nnodeTimeout:{nodeTimeout}\nentries:\n{entries}".format(
            partName=self.partName,
            platform=self.platform,
            heartbeatInterval=self.heartbeatInterval(),
            nodeTimeout=self.nodeTimeout(),
            entries=pprint.pformat(self.entries))
        return mstr

    def dump(self):
        print(self)

# Runs self test of CMState class

def test_cmstate (verbose=1):
    cmstate0 = CMState(0)
    cmstate1 = CMState(1)
    cmstate2 = CMState(2, 20)
    if verbose:
        cmstate0.dump()
        cmstate1.dump()
        cmstate2.dump()

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

    logging.info('cmserver starting')

    timer1started = False
    # CM state
    cmstate = CMState(platform=args.p)

    pongCount = 0
    # context and sockets
    ctx = zmq.Context()
    cmd = ctx.socket(zmq.ROUTER)
    cmd.bind("tcp://*:%d" % CMMsg.router_port(cmstate.platform()))
    publisher = ctx.socket(zmq.PUB)
    publisher.bind("tcp://*:%d" % CMMsg.pub_port(cmstate.platform()))
    timerReceive = ctx.socket(zmq.PAIR)
    timerEndpoint = "inproc://timer"
    timerReceive.sndtimeo = 0
    timerReceive.bind(timerEndpoint)

    if cmstate.heartbeatInterval() > 0:
        # create timer thread
        timers = [ { 'period' : cmstate.heartbeatInterval(), 'offset' :  0, 'msg' : CMMsg.STARTPING },
                   { 'period' : cmstate.heartbeatInterval(), 'offset' :  1, 'msg' : CMMsg.STARTEXP  } ]

        timer1 = ZTimer("Timer-1", ctx, timerEndpoint, timers)
        timer1.start()
        timer1started = True

    sequence = 0

    poller = zmq.Poller()
    poller.register(cmd, zmq.POLLIN)
    poller.register(timerReceive, zmq.POLLIN)
    try:
        while True:
            items = dict(poller.poll(1000))

            # Execute timer request
            if timerReceive in items:
                request = timerReceive.recv()
                logging.debug('Received <%s> from timer' % request.decode())

                if request == CMMsg.STARTPING:
                    # Send PING broadcast
                    cmmsg = CMMsg(key=CMMsg.PING)
                    cmmsg.send(publisher)
                    logging.debug("Published <PING>")
                    continue
                elif request == CMMsg.STARTEXP:
                    # Remove expired keys
                    exlist = cmstate.expired_keys(cmstate.nodeTimeout())
                    if len(exlist) > 0:
                        removed = cmstate.remove_keys(exlist)
                        for rr in removed:
                            logging.warning("Node timed out: %s" % rr)
                        logging.warning("Removed %d nodes after %ds timeout" % (len(removed), cmstate.nodeTimeout()))
                    continue

            # Execute state cmd request
            if cmd in items:
                msg = cmd.recv_multipart()
                identity = msg[0]
                request = msg[1]
                logging.debug('Received <%s> from cmd' % request.decode())

                if request == CMMsg.GETSTATE:

                    # Send STATE reply to client
                    logging.debug("Sending STATE reply")
                    cmd.send(identity, zmq.SNDMORE)
                    cmmsg = CMMsg(key=CMMsg.STATE)
                    cmmsg['platform'] = cmstate.platform()
                    cmmsg['partName'] = cmstate.partName()
                    cmmsg['nodes'] = pickle.dumps(cmstate.nodes())
                    cmmsg.send(cmd)
                    continue

                if request == CMMsg.STARTPH1:
                    # Assign partition name
                    try:
                        prop = CMMsg.decode_properties(msg[4])
                    except Exception as ex:
                        logging.error(ex)
                        prop = {}
                    if "partName" in prop:
                        cmstate._partName = prop["partName"]
                        logging.debug("Partition name: %s" % cmstate.partName())

                    # Send PH1 broadcast
                    logging.debug("Sending PH1 broadcast")
                    cmmsg = CMMsg(key=CMMsg.PH1)
                    cmmsg.send(publisher)

                    # Send PH1STARTED reply to client
                    logging.debug("Sending PH1STARTED reply")
                    cmd.send(identity, zmq.SNDMORE)
                    cmmsg = CMMsg(key=CMMsg.PH1STARTED)
                    cmmsg.send(cmd)
                    continue

                if request == CMMsg.STARTPH2:
                    # Send PH2 broadcast
                    logging.debug("Sending PH2 broadcast")
                    cmmsg = CMMsg(key=CMMsg.PH2)
                    cmmsg.send(publisher)

                    # Send PH2STARTED reply to client
                    logging.debug("Sending PH2STARTED reply")
                    cmd.send(identity, zmq.SNDMORE)
                    cmmsg = CMMsg(key=CMMsg.PH2STARTED)
                    cmmsg.send(cmd)
                    continue

                if request == CMMsg.STARTKILL:
                    # Send KILL broadcast
                    logging.debug("Sending KILL broadcast")
                    cmmsg = CMMsg(key=CMMsg.KILL)
                    cmmsg['platform'] = cmstate.platform()
                    cmmsg.send(publisher)

                    # reset the CM state
                    cmstate.reset()

                    # Send KILLSTARTED reply to client
                    logging.debug("Sending KILLSTARTED reply")
                    cmd.send(identity, zmq.SNDMORE)
                    cmmsg = CMMsg(key=CMMsg.KILLSTARTED)
                    cmmsg['platform'] = cmstate.platform()
                    cmmsg.send(cmd)
                    continue

                elif request == CMMsg.STARTDIE:
                    # Send DIE broadcast
                    logging.debug("Sending DIE broadcast")
                    cmmsg = CMMsg(key=CMMsg.DIE)
                    cmmsg.send(publisher)

                    # Send DIESTARTED reply to client
                    logging.debug("Sending DIESTARTED reply")
                    cmd.send(identity, zmq.SNDMORE)
                    cmmsg = CMMsg(key=CMMsg.DIESTARTED)
                    cmmsg.send(cmd)
                    continue

                elif request == CMMsg.HELLO:
                    try:
                        prop = CMMsg.decode_properties(msg[4])
                    except Exception as ex:
                        logging.error(ex)
                        prop = {}
                    logging.debug("properties = %s" % prop)

                    # remove any duplicates before adding new entry
                    exlist = cmstate.find_duplicates(prop)
                    if len(exlist) > 0:
                        removed = cmstate.remove_keys(exlist)
                        for rr in removed:
                            logging.warning("Node duplicated: %s" % rr)
                        logging.warning("Removed %d duplicate nodes" % len(removed))

                    # add new entry
                    cmstate[identity] = prop
                    try:
                        # update timestamp
                        cmstate.timestamp(identity)
                    except:
                        logging.debug("HELLO timestamp failed")

                    continue

                elif request == CMMsg.PORTS:
                    try:
                        prop = CMMsg.decode_properties(msg[4])
                    except:
                        prop = {}

                    if 'ports' in prop:
                        try:
                            cmstate[identity]['ports'] = prop['ports']
                        except:
                            logging.debug("Setting PORTS property failed")
                    continue

                elif request == CMMsg.PONG:
                    pongCount += 1
                    logging.debug("PONG #%d" % pongCount)
                    if identity in cmstate.keys():
                        try:
                            # update timestamp
                            cmstate.timestamp(identity)
                        except:
                            logging.debug("PONG timestamp failed")
                    continue

                elif request == CMMsg.STARTDUMP:
                    # Send reply to client
                    logging.debug("Sending DUMPSTARTED reply")
                    cmd.send(identity, zmq.SNDMORE)
                    cmmsg = CMMsg(key=CMMsg.DUMPSTARTED)
                    cmmsg.send(cmd)

                    # Dump state to console
                    print("platform:", cmstate.platform())
                    print("partName:", cmstate.partName())
                    print("heartbeatInterval:", cmstate.heartbeatInterval())
                    print("nodeTimeout:", cmstate.nodeTimeout())
                    print("Nodes:")
                    pprint.pprint(cmstate.entries)
                    continue

                else:
                    logging.warning("Unknown msg <%s>" % request.decode())
                    # Send reply to client
                    logging.debug("Sending <HUH?> reply")
                    cmd.send(identity, zmq.SNDMORE)
                    cmmsg = CMMsg(key=CMMsg.HUH)
                    cmmsg.send(cmd)
                    continue

    except KeyboardInterrupt:
        logging.debug("Interrupt received")

    # Clean up
    logging.debug("Clean up")
    try:
        timerReceive.send(b"")  # signal timer to exit
    except zmq.Again:
        pass

    time.sleep(.25)

    # close zmq sockets
    cmd.close()
    publisher.close()
    timerReceive.close()

    # terminate zmq context
    ctx.term()

    if timer1started:
        timer1.join()         # join timer thread

    logging.info('cmserver exiting')

if __name__ == '__main__':
    main()
