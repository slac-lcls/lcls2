#!/usr/bin/env python
"""
CM Phase 1 command
"""
import time
import zmq
import sys
import argparse
from CMMsg import CMMsg

def main():

    # Process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('partName', help='Partition name')
    parser.add_argument('-p', type=int, choices=range(0, 8), default=0, help='platform (default 0)')
    parser.add_argument('-C', metavar='CM_HOST', default='localhost', help='Collection Manager host')
    args = parser.parse_args()

    # Compose message
    newmsg = CMMsg(key=CMMsg.STARTPH1)
    newmsg['partName'] = args.partName
    newmsg['platform'] = args.p

    # Prepare our context and DEALER socket
    ctx = zmq.Context()
    cmd = ctx.socket(zmq.DEALER)
    cmd.linger = 0
    cmd.RCVTIMEO = 5000 # in milliseconds
    cmd.connect("tcp://%s:%d" % (args.C, CMMsg.router_port(args.p)))

    # Send message
    newmsg.send(cmd)

    while True:
        try:
            cmmsg = CMMsg.recv(cmd)
        except Exception as ex:
            print(ex)
            raise
            return

        if cmmsg.key == CMMsg.PH1STARTED:
            print ("I: Received PH1STARTED")
            break          # Done
        else:
            print ("W: Received key \"%s\"" % cmmsg.key)
            continue

#   print ("Done")

if __name__ == '__main__':
    main()
