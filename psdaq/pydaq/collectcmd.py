#!/usr/bin/env python
"""
collectcmd - send a collection command via ZMQ

Author: Chris Ford <caf@slac.stanford.edu>
"""
import time
import zmq
import pickle
import pprint
import argparse
from CMMsg import CMMsg

def main():

    # Define commands
    command_dict = { 'plat': CMMsg.STARTPLAT,
                     'alloc': CMMsg.STARTALLOC,
                     'connect': CMMsg.STARTCONNECT,
                     'dump': CMMsg.STARTDUMP,
                     'die': CMMsg.STARTDIE,
                     'kill': CMMsg.STARTKILL,
                     'getstate': CMMsg.GETSTATE }

    # Process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=command_dict.keys())
    parser.add_argument('-p', type=int, choices=range(0, 8), default=0, help='platform (default 0)')
    parser.add_argument('-C', metavar='COLLECT_HOST', default='localhost', help='collection host (default localhost)')
    parser.add_argument('-P', metavar='PARTITION', default='AMO', help='partition name (default AMO)')
    args = parser.parse_args()

    # Prepare our context and DEALER socket
    ctx = zmq.Context()
    cmd_socket = ctx.socket(zmq.DEALER)
    cmd_socket.linger = 0
    cmd_socket.RCVTIMEO = 5000 # in milliseconds
    cmd_socket.connect("tcp://%s:%d" % (args.C, CMMsg.router_port(args.p)))

    # Compose message
    newmsg = CMMsg(0, key=command_dict[args.command])
    newmsg[b'partName'] = args.P.encode(encoding='UTF-8')
    newmsg[b'platform'] = ('%d' % args.p).encode(encoding='UTF-8')

    # Send message
    newmsg.send(cmd_socket)

    # Receive reply
    try:
        cmmsg = CMMsg.recv(cmd_socket)
    except Exception as ex:
        print(ex)
    else:
        print ("Received \"%s\"" % cmmsg.key.decode())

        if cmmsg.key == CMMsg.STATE:
            props = cmmsg.properties

            # platform
            platform = 0
            try:
                platform = props[b'platform'].decode()
            except KeyError:
                print('E: platform key not found')
            else:
                print ('Platform:', platform)

            # partition name
            partName = '(None)'
            try:
                partName = props[b'partName'].decode()
            except KeyError:
                print('E: partName key not found')
            else:
                print ('Partition name:', partName)

            # nodes
            try:
                nodes = pickle.loads(cmmsg.body)
            except Exception as ex:
                print('E: pickle.loads()', ex)
            else:
                print ('Nodes:')
                pprint.pprint (nodes)

    return

if __name__ == '__main__':
    main()
