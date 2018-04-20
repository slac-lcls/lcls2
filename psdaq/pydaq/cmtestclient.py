#!/usr/bin/env python

import time
import zmq
from CMMsg import CMMsg
import sys
import pickle
import argparse

def main():

    # Process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('clientId', type=int, choices=range(0, 10), help='client ID')
    parser.add_argument('-p', type=int, choices=range(0, 8), default=0, help='platform (default 0)')
    parser.add_argument('-C', metavar='CM_HOST', default='localhost', help='Collection Manager host')
    parser.add_argument('-v', action='store_true', help='be verbose')
    args = parser.parse_args()

    clientId = args.clientId
    verbose = args.v

    # Prepare our context and sockets
    ctx = zmq.Context()
    cmd = ctx.socket(zmq.DEALER)
    cmd.linger = 0
    cmd.connect("tcp://%s:%d" % (args.C, CMMsg.router_port(args.p)))
    subscriber = ctx.socket(zmq.SUB)
    subscriber.linger = 0
    subscriber.setsockopt_string(zmq.SUBSCRIBE, '')
    subscriber.connect("tcp://%s:%d" % (args.C, CMMsg.pub_port(args.p)))

    poller = zmq.Poller()
    poller.register(subscriber, zmq.POLLIN)

    alarm = time.time()+1.
    while True:
        tickless = 1000*max(0, alarm - time.time())
        try:
            items = dict(poller.poll(tickless))
        except:
            break           # Interrupted

        if subscriber in items:
            cmmsg = CMMsg.recv(subscriber)
            if cmmsg.key == CMMsg.PING:
                if verbose:
                    print( "Received PING, sending PONG")
                cmd.send(CMMsg.PONG)

            elif cmmsg.key == CMMsg.PH1:
                if verbose:
                    print( "Received PH1, sending HELLO (platform=%d)" % args.p)
                newmsg = CMMsg(key=CMMsg.HELLO)
                # Create simulated ports entry based on clientId N, platform P
                # {
                #   'platform' : P
                #   'group'    : N
                #   'uid'      : N
                #   'level'    : N
                #   'pid'      : 25NNN
                #   'ip'       : '172.NN.NN.NN'
                #   'ether'    : 'NN:NN:NN:NN:NN:NN'
                # }
                newmsg['platform'] = args.p
                newmsg['group'] = clientId
                newmsg['uid'] = clientId
                newmsg['level'] = clientId
                newmsg['pid'] = 25000 + (111 * clientId)
                newmsg['ip'] = '172' + (3 * ('.%d%d' % (clientId, clientId)))
                newmsg['ether'] = ('%d%d' % (clientId, clientId)) + (5 * (':%d%d' % (clientId, clientId)))
                newmsg.send(cmd)

            elif cmmsg.key == CMMsg.PH2:
                if verbose:
                    print( "Received PH2, sending PORTS")
                newmsg = CMMsg(key=CMMsg.PORTS)
                # Create simulated ports entry based on clientId N
                # {
                #   'name' : 'portN',
                #   'endpoint' : 'tcp://localhost:NNNN'
                # }
                ports = {}
                ports['name'] = 'port%d' % clientId
                ports['endpoint'] = 'tcp://localhost:' + 4 * str(clientId)
                newmsg['ports'] = pickle.dumps(ports)
                newmsg.send(cmd)

            elif cmmsg.key == CMMsg.DIE:
                if verbose:
                    print( "Received DIE, exiting")
                break       # Interrupted

            elif cmmsg.key == CMMsg.KILL:
                if verbose:
                    print( "Received KILL, ignoring")

            else:
                if verbose:
                    print( "Received key=\"%s\"" % cmmsg.key)

    print ("Interrupted")
    sys.exit(0)

if __name__ == '__main__':
    main()
