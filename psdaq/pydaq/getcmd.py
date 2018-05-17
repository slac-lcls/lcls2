#!/usr/bin/env python
"""
CM get command
"""
import time
import zmq
import pickle
import pprint
import argparse
from kvmsg import decode_properties
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
            msg = cmd.recv_multipart()
        except Exception as ex:
            print(ex)
            return

        request = msg[0]
        if request == CMMsg.STATE:
            print ("I: Received STATE")
            if len(msg) == 5:
                # msg[0]: key
                # msg[1]: sequence
                # msg[2]: identity
                # msg[3]: properties
                # msg[4]: body
                props = decode_properties(msg[3])

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
                nodes = pickle.loads(msg[4])
                print ('Nodes:')
                pprint.pprint (nodes)
            else:
                print ("E: STATE message len %d, expected 5" % len(msg))
            break          # Done
        else:
            print ("W: Received key \"%s\"" % request)
            continue

    print ("Done")

if __name__ == '__main__':
    main()
