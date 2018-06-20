#!/usr/bin/env python

"""
collectTest - Collection Manager test client

Author: Chris Ford <caf@slac.stanford.edu>
"""

import time
import zmq
from CollectMsg import CollectMsg
import sys
import argparse
import socket
from zmq.utils import jsonapi as json
import pprint

def main():

    # Process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('clientId', type=int, choices=range(0, 10), help='client ID')
    parser.add_argument('-p', type=int, choices=range(0, 8), default=0, help='platform (default 0)')
    parser.add_argument('-C', metavar='CM_HOST', default='localhost', help='Collection Manager host')
    parser.add_argument('-v', action='store_true', help='be verbose')
    parser.add_argument('--unselect', action='store_true', help='avoid allocation')
    args = parser.parse_args()

    clientId = args.clientId
    verbose = args.v
    unselect = args.unselect

    # Prepare our context and sockets
    ctx = zmq.Context()
    collect_cmd = ctx.socket(zmq.DEALER)
    collect_cmd.linger = 0
    collect_cmd.connect("tcp://%s:%d" % (args.C, CollectMsg.router_port(args.p)))
    collect_sub = ctx.socket(zmq.SUB)
    collect_sub.linger = 0
    collect_sub.setsockopt_string(zmq.SUBSCRIBE, '')
    collect_sub.connect("tcp://%s:%d" % (args.C, CollectMsg.pub_port(args.p)))

    poller = zmq.Poller()
    # Listen to both DEALER and SUB sockets
    poller.register(collect_sub, zmq.POLLIN)
    poller.register(collect_cmd, zmq.POLLIN)

    alarm = time.time()+1.
    while True:
        tickless = 1000*max(0, alarm - time.time())
        try:
            items = dict(poller.poll(tickless))
        except:
            break           # Interrupted

        if collect_sub in items:
            cmmsg = CollectMsg.recv(collect_sub)
            if cmmsg.key == CollectMsg.PING:
                if verbose:
                    print( "Received PING, sending PONG")
                collect_cmd.send(CollectMsg.PONG)

            elif cmmsg.key == CollectMsg.PLAT:
                if verbose:
                    print( "Received PLAT, sending HELLO (platform=%d)" % args.p)
                newmsg = CollectMsg(0, key=CollectMsg.HELLO)
                # Create simulated ports entry based on clientId N, platform P
                # {
                #   'platform' : P
                #   'name'     : 'collectTestN'
                #   'group'    : N
                #   'uid'      : N
                #   'level'    : N
                #   'pid'      : 25NNN
                #   'ip'       : '172.NN.NN.NN'
                #   'host'     : '<hostname>'
                #   'ether'    : 'NN:NN:NN:NN:NN:NN'
                # }
                newmsg['platform'] = args.p
                newmsg['name'] = 'collectTest%d' % clientId
                if unselect:
                    newmsg['select'] = 0
                newmsg['group'] = newmsg['uid'] = newmsg['level'] = \
                    clientId
                newmsg['ip'] = \
                    '172' + (3 * ('.%d%d' % (clientId, clientId)))
                newmsg['pid'] = 25000 + (111 * clientId)
                newmsg['host'] = socket.gethostname()
                newmsg['ether'] = \
                    '%d%d' % (clientId, clientId) + \
                               (5 * (':%d%d' % (clientId, clientId)))
                try:
                    newmsg.send(collect_cmd)
                except Exception as ex:
                    print('E: newmsg.send()', ex)

            elif cmmsg.key == CollectMsg.DIE:
                if verbose:
                    print( "Received DIE, exiting")
                break       # Interrupted

            elif cmmsg.key == CollectMsg.KILL:
                if verbose:
                    print( "Received KILL, ignoring")

            elif verbose:
                print( "Client ID %d: Received key=\"%s\" on SUB socket" % (clientId, cmmsg.key))

        if collect_cmd in items:
            cmmsg = CollectMsg.recv(collect_cmd)
            if cmmsg.key == CollectMsg.ALLOC:
                if verbose:
                    print( "Received ALLOC, sending PORTS")
                newmsg = CollectMsg(0, key=CollectMsg.PORTS)
                # Create simulated ports entry based on clientId N
                # {
                #   'port' : [ '<hostname>', 5550<N> ]
                # }
                ports = {}
                ports['port'] = [ socket.gethostname(), 55500 + clientId ]
                newmsg['ports'] = [ ports ]
                newmsg.send(collect_cmd)

            elif cmmsg.key == CollectMsg.CONNECT:
                if verbose:
                    print("Received CONNECT")
                try:
                    pybody = json.loads(cmmsg.body)
                except Exception as ex:
                    print("CONNECT: json.loads() exception: %s" % ex)
                    print("CONNECT: cmmsg.body = <%s>" % cmmsg.body)
                    pybody = {}
                if verbose:
                    print("--------------------------")
                    pprint.pprint(pybody)
                    print("--------------------------")

            elif verbose:
                print( "Client ID %d: Received key=\"%s\" on DEALER socket" % (clientId, cmmsg.key))

    print ("Interrupted")
    sys.exit(0)

if __name__ == '__main__':
    main()
