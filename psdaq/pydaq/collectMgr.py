#!/usr/bin/env python

"""
collectMgr - Collection Manager

Author: Chris Ford <caf@slac.stanford.edu>
"""

import zmq
import sys
import time
import logging
import zmq.utils.jsonapi as json
import pprint
import argparse

from CollectMsg import CollectMsg
from ZTimer import ZTimer

class CMState(object):

    entries = {}
    UNASSIGNED = "Unassigned"

    def __init__(self, platform, heartbeatInterval=0):
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
                # check for multiple control levels
                if (self.entries[k]['level'] == prop['level'] == 0):
                    rv.append(k)
            except Exception:
                pass
        return rv

    def remove_keys(self, keylist):
        rlist = []
        for k in keylist:
            if k in self.entries:
                try:
                    rlist.append(self.entries[k]['name'])
                except KeyError:
                    rlist.append("(no name)")
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

    logging.info('collectMgr starting')

    timer1started = False
    # CM state
    cmstate = CMState(platform=args.p)

    pongCount = 0
    # context and sockets
    ctx = zmq.Context()
    cmd = ctx.socket(zmq.ROUTER)
    cmd.bind("tcp://*:%d" % CollectMsg.router_port(cmstate.platform()))
    publisher = ctx.socket(zmq.PUB)
    publisher.bind("tcp://*:%d" % CollectMsg.pub_port(cmstate.platform()))
    timerReceive = ctx.socket(zmq.PAIR)
    timerEndpoint = "inproc://timer"
    timerReceive.sndtimeo = 0
    timerReceive.bind(timerEndpoint)

    if cmstate.heartbeatInterval() > 0:
        # create timer thread
        timers = [ { 'period' : cmstate.heartbeatInterval(), 'offset' :  0, 'msg' : CollectMsg.STARTPING },
                   { 'period' : cmstate.heartbeatInterval(), 'offset' :  1, 'msg' : CollectMsg.STARTEXP  } ]

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

                if request == CollectMsg.STARTPING:
                    # Send PING broadcast
                    #cmmsg = CollectMsg(sequence, key=CollectMsg.PING)
                    #cmmsg.send(publisher)
                    #logging.debug("Published <PING>")
                    continue
                elif request == CollectMsg.STARTEXP:
                    # Remove expired keys
                    exlist = cmstate.expired_keys(cmstate.nodeTimeout())
                    if len(exlist) > 0:
                        pass
                        #removed = cmstate.remove_keys(exlist)
                        #for rr in removed:
                        #    logging.warning("Node timed out: %s" % rr)
                        #logging.warning("Removed %d nodes after %ds timeout" % (len(removed), cmstate.nodeTimeout()))
                    continue

            # Execute state cmd request
            if cmd in items:
                msg = cmd.recv_multipart()
                identity = msg[0]
                request = msg[1]
                logging.debug('Received <%s> from cmd' % request.decode())

                if request == CollectMsg.GETSTATE:

                    # Send STATE reply to client
                    logging.debug("Sending STATE reply")
                    cmd.send(identity, zmq.SNDMORE)
                    print('cmstate.nodes():', cmstate.nodes())
                    try:
                        testbody = json.dumps(cmstate.nodes())
                    except Exception as ex:
                        logging.error(ex)
                        testbody = ''
                    cmmsg = CollectMsg(sequence, key=CollectMsg.STATE, body=testbody)
                    cmmsg['platform'] = cmstate.platform()
                    cmmsg['partName'] = cmstate.partName()
                    cmmsg.send(cmd)
                    continue

                if request == CollectMsg.STARTPLAT:
                    # Assign partition name
                    try:
                        prop = json.loads(msg[4])
                    except Exception as ex:
                        logging.error(ex)
                        prop = {}
                    try:
                        cmstate._partName = prop['partName']
                    except KeyError:
                        logging.error("STARTPLAT message: No partName property")
                        cmstate._partName = 'Unassigned'
                    logging.debug("Partition name: %s" % cmstate._partName)

                    # Send PLAT broadcast
                    logging.debug("Sending PLAT broadcast")
                    cmmsg = CollectMsg(sequence, key=CollectMsg.PLAT)
                    cmmsg.send(publisher)

                    # Send PLATSTARTED reply to client
                    logging.debug("Sending PLATSTARTED reply")
                    cmd.send(identity, zmq.SNDMORE)
                    cmmsg = CollectMsg(sequence, key=CollectMsg.PLATSTARTED)
                    cmmsg.send(cmd)
                    continue

                if request == CollectMsg.STARTALLOC:
                    # Send ALLOC individually
                    logging.debug("Sending ALLOC individually")
                    cmmsg = CollectMsg(sequence, key=CollectMsg.ALLOC)
                    for key in cmstate.entries.keys():
                        cmd.send(key, zmq.SNDMORE)
                        cmmsg.send(cmd)
                        logging.debug("...sent ALLOC")

                    # Send ALLOCSTARTED reply to client
                    logging.debug("Sending ALLOCSTARTED reply")
                    cmd.send(identity, zmq.SNDMORE)
                    cmmsg = CollectMsg(sequence, key=CollectMsg.ALLOCSTARTED)
                    cmmsg.send(cmd)
                    continue

                if request == CollectMsg.STARTCONNECT:
                    # Send CONNECT individually
                    logging.debug("Sending CONNECT individually")

                    # start composing JSON message
                    pybody = {'msgType': 'connect', 'msgVer' : 1}
                    pybody['platform'] = cmstate.platform()
                    pybody['procs'] = {}

                    # add ports for control level
                    pybody['procs']['control'] = {}
                    for key in cmstate.entries.keys():
                        try:
                            level = cmstate.entries[key]['level']
                        except KeyError:
                            pass
                        else:
                            if level == 0:
                                # copy some entries
                                try:
                                    pybody['procs']['control']['name'] = cmstate.entries[key]['name']
                                    pybody['procs']['control']['ports'] = cmstate.entries[key]['ports']
                                except Exception as ex:
                                    logging.error("Failed to copy control entries: %s" % ex)
                                    pass
                                else:
                                    logging.debug("Copied control entries")
                                # done
                                break

                    jsonbody = json.dumps(pybody)
                    cmmsg = CollectMsg(sequence, key=CollectMsg.CONNECT, body=jsonbody)

                    for key in cmstate.entries.keys():
                        cmd.send(key, zmq.SNDMORE)
                        cmmsg.send(cmd)
                        logging.debug("...sent CONNECT")

#                   cmd.send(key, zmq.SNDMORE)
#                   cmmsg.send(cmd)

                    # Send CONNECTSTARTED reply to client
                    logging.debug("Sending CONNECTSTARTED reply")
                    cmd.send(identity, zmq.SNDMORE)
                    cmmsg = CollectMsg(sequence, key=CollectMsg.CONNECTSTARTED)
                    cmmsg.send(cmd)
                    continue

                if request == CollectMsg.STARTKILL:
                    # Send KILL broadcast
                    logging.debug("Sending KILL broadcast")
                    cmmsg = CollectMsg(sequence, key=CollectMsg.KILL)
                    cmmsg['platform'] = cmstate.platform()
                    cmmsg.send(publisher)

                    # reset the CM state
                    cmstate.reset()

                    # Send KILLSTARTED reply to client
                    logging.debug("Sending KILLSTARTED reply")
                    cmd.send(identity, zmq.SNDMORE)
                    cmmsg = CollectMsg(sequence, key=CollectMsg.KILLSTARTED)
                    cmmsg['platform'] = cmstate.platform()
                    cmmsg.send(cmd)
                    continue

                elif request == CollectMsg.STARTDIE:
                    # Send DIE broadcast
                    logging.debug("Sending DIE broadcast")
                    cmmsg = CollectMsg(sequence, key=CollectMsg.DIE)
                    cmmsg.send(publisher)

                    # Send DIESTARTED reply to client
                    logging.debug("Sending DIESTARTED reply")
                    cmd.send(identity, zmq.SNDMORE)
                    cmmsg = CollectMsg(sequence, key=CollectMsg.DIESTARTED)
                    cmmsg.send(cmd)
                    continue

                elif request == CollectMsg.HELLO:
                    logging.debug("Loading HELLO properties with JSON")
                    if len(msg) == 5 or len(msg) == 6:
                        try:
                            prop = json.loads(msg[4])
                        except Exception as ex:
                            logging.error(ex)
                            prop = {}
                        logging.debug("HELLO properties = %s" % prop)
                    else:
                        logging.error("Got HELLO msg of len %d, expected 5 or 6" % len(msg))
                        prop = {}

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

                elif request == CollectMsg.PORTS:
                    try:
                        prop = json.loads(msg[4])
                    except:
                        prop = {}

                    if 'ports' in prop:
                        try:
                            cmstate[identity]['ports'] = prop['ports']
                        except:
                            logging.debug("Setting PORTS property failed")
                    else:
                        logging.error("PORTS message: No ports property")
                    continue

                elif request == CollectMsg.PONG:
                    pongCount += 1
                    logging.debug("PONG #%d" % pongCount)
                    if identity in cmstate.keys():
                        try:
                            # update timestamp
                            cmstate.timestamp(identity)
                        except:
                            logging.debug("PONG timestamp failed")
                    continue

                elif request == CollectMsg.STARTDUMP:
                    # Send reply to client
                    logging.debug("Sending DUMPSTARTED reply")
                    cmd.send(identity, zmq.SNDMORE)
                    cmmsg = CollectMsg(sequence, key=CollectMsg.DUMPSTARTED)
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
                    cmmsg = CollectMsg(sequence, key=CollectMsg.HUH)
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

    logging.info('collectMgr exiting')

if __name__ == '__main__':
    main()
