#!/usr/bin/env python
"""
collectcmd - send a collection command via ZMQ

Author: Chris Ford <caf@slac.stanford.edu>
"""
import time
import zmq
import argparse
from CMMsg import CMMsg

def main():

    # Define commands
    command_dict = { 'ph1': CMMsg.STARTPH1,
                     'ph2': CMMsg.STARTPH2,
                     'dump': CMMsg.STARTDUMP,
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
    return

if __name__ == '__main__':
    main()
