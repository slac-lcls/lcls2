#!/usr/bin/env python

import time
import zmq
from CMMsg import CMMsg
import sys
import pickle

def usage():
    print("Usage: %s <clientId (0-9)>" % sys.argv[0])

def main():

    clientId = -1   # not set
    if len(sys.argv) == 2:
        try:
            clientId = int(sys.argv[1])
        except:
            pass
    if clientId < 0 or clientId > 9:
        usage()
        sys.exit(1)

    # Prepare our context and sockets
    ctx = zmq.Context()
    cmd = ctx.socket(zmq.DEALER)
    cmd.linger = 0
    cmd.connect("tcp://%s:5556" % CMMsg.host())
    subscriber = ctx.socket(zmq.SUB)
    subscriber.linger = 0
    subscriber.setsockopt_string(zmq.SUBSCRIBE, '')
    subscriber.connect("tcp://%s:5557" % CMMsg.host())

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
                print( "I: Received PING, sending PONG")
                cmd.send(CMMsg.PONG)

            elif cmmsg.key == CMMsg.PH1:
                print( "I: Received PH1, sending HELLO")
                newmsg = CMMsg(0, key=CMMsg.HELLO)
                # Create simulated ports entry based on clientId N
                # {
                #   'platform' : 0
                #   'group'    : N
                #   'uid'      : N
                #   'level'    : N
                #   'pid'      : 25NNN
                #   'ip'       : '172.NN.NN.NN'
                #   'ether'    : 'NN:NN:NN:NN:NN:NN'
                # }
                newmsg['platform'] = 0
                newmsg['group'] = clientId
                newmsg['uid'] = clientId
                newmsg['level'] = clientId
                newmsg['pid'] = 25000 + (111 * clientId)
                newmsg['ip'] = '172' + (3 * ('.%d%d' % (clientId, clientId)))
                newmsg['ether'] = ('%d%d' % (clientId, clientId)) + (5 * (':%d%d' % (clientId, clientId)))
                newmsg.send(cmd)

            elif cmmsg.key == CMMsg.PH2:
                print( "I: Received PH2, sending PORTS")
                newmsg = CMMsg(0, key=CMMsg.PORTS)
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
                print( "I: Received DIE, exiting")
                break       # Interrupted

            elif cmmsg.key == CMMsg.KILL:
                print( "I: Received KILL, ignoring")

            else:
                print( "I: Received key=\"%s\"" % cmmsg.key)

    print ("Interrupted")
    sys.exit(0)

if __name__ == '__main__':
    main()
