#!/usr/bin/env python
"""
collectCmd - send a collection command via ZMQ

Author: Chris Ford <caf@slac.stanford.edu>
"""
import time
import zmq
import zmq.utils.jsonapi as json
import pprint
import argparse
from CollectMsg import CollectMsg

def main():

    # Define commands
    command_dict = { 'plat': CollectMsg.STARTPLAT,
                     'alloc': CollectMsg.STARTALLOC,
                     'connect': CollectMsg.STARTCONNECT,
                     'dump': CollectMsg.STARTDUMP,
                     'die': CollectMsg.STARTDIE,
                     'kill': CollectMsg.STARTKILL,
                     'getstate': CollectMsg.GETSTATE }

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
    cmd_socket.connect("tcp://%s:%d" % (args.C, CollectMsg.router_port(args.p)))

    # Compose message
    newmsg = CollectMsg(key=command_dict[args.command])
#   newmsg['partName'] = args.P
#   newmsg['platform'] = ('%d' % args.p)

    # Send message
    newmsg.send(cmd_socket)

    # Receive reply
    try:
        cmmsg = CollectMsg.recv(cmd_socket)
    except Exception as ex:
        print(ex)
    else:
        print ("Received \"%s\"" % cmmsg.key.decode())

        if cmmsg.key == CollectMsg.STATE:
            # nodes
            try:
                nodes = json.loads(cmmsg.body)
            except Exception as ex:
                print('E: json.loads()', ex)
            else:
                print ('Nodes:')
                pprint.pprint (nodes)

    return

if __name__ == '__main__':
    main()
