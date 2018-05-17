#!/usr/bin/env python
"""
CM get command
"""
import time
import zmq
import pickle
import pprint
import argparse
from CMMsg import CMMsg

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
    cmd.connect("tcp://%s:%d" % (args.C, CMMsg.router_port(args.p)))

    cmd.send(CMMsg.GETSTATE)
    while True:
        try:
            msg = CMMsg.recv(cmd)
        except Exception as ex:
            print(ex)
            return

        if msg.key == CMMsg.STATE:
            print ("I: Received STATE")

            props = msg.properties

            # platform
            platform = 0
            try:
                platform = props[b'platform'].decode()
            except KeyError:
                print('E: platform key not found')
            print ('Platform:', platform)

            # partition name
            partName = '(None)'
            try:
                partName = props[b'partName'].decode()
            except KeyError:
                print('E: partName key not found')
            print ('Partition name:', partName)

            # nodes
            nodes = pickle.loads(msg.body)
            print ('Nodes:')
            pprint.pprint (nodes)
            break          # Done
        else:
            print ("W: Received key \"%s\"" % msg.key)
            continue

    print ("Done")

if __name__ == '__main__':
    main()
