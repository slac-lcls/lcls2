#!/usr/bin/env python
"""
Collection Manager killPlatform command
"""
import time
import zmq
import argparse
from CollectMsg import CollectMsg

def main():

    # Process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=int, choices=range(0, 8), default=0, help='platform (default 0)')
    parser.add_argument('-C', metavar='CM_HOST', default='localhost', help='Collection Manager host')
    args = parser.parse_args()

    # Prepare our context and DEALER socket
    ctx = zmq.Context()
    cmd = ctx.socket(zmq.DEALER)
    cmd.linger = 0
    cmd.RCVTIMEO = 5000 # in milliseconds
    cmd.connect("tcp://%s:%d" % (args.C, CollectMsg.router_port(args.p)))

    # Initiate partition kill
    cmd.send(CollectMsg.STARTKILL)
    while True:
        try:
            cmmsg = CollectMsg.recv(cmd)
        except Exception as ex:
            print(ex)
            return

        if cmmsg.key == CollectMsg.KILLSTARTED:
            print ("I: Received KILLSTARTED")
            break          # Done
        else:
            print ("W: Received key <%s>" % cmmsg.key.decode())
            continue

#   print ("Done")

if __name__ == '__main__':
    main()
