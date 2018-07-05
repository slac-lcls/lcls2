#!/usr/bin/env python
"""
Collection Manager showPlatform command
"""
import sys
import zmq
import zmq.utils.jsonapi as json
import argparse
from CollectMsg import CollectMsg

def main():

    # Process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=int, choices=range(0, 8), default=0, help='platform (default 0)')
    parser.add_argument('-C', metavar='CM_HOST', default='localhost', help='Collection Manager host')
    parser.add_argument('--noheader', action='store_true', help='do not print header')
    args = parser.parse_args()

    # Prepare our context and DEALER socket
    ctx = zmq.Context()
    cmd = ctx.socket(zmq.DEALER)
    cmd.linger = 0
    cmd.RCVTIMEO = 5000 # in milliseconds
    cmd.connect("tcp://%s:%d" % (args.C, CollectMsg.router_port(args.p)))

    cmd.send(CollectMsg.GETSTATE)
    while True:
        try:
            msg = CollectMsg.recv(cmd)
        except Exception as ex:
            print(ex)
            return

        request = msg.key
        if request == CollectMsg.STATE:
            # platform FIXME
            platform = '0'

            # partition name FIXME
            partName = '(None)'

            # nodes
            nodes = []
            try:
                nodes = json.loads(msg.body)
            except Exception as ex:
                print('E: json.loads()', ex)
                nodes = []
            displayList = []

            for level, nodelist in nodes.items():
                for node in nodelist:
                    try:
                        host = node['procInfo']['host']
                    except KeyError:
                        host = '(Unknown)'
                    try:
                        pid = node['procInfo']['pid']
                    except KeyError:
                        pid = 0
                    display = "%s/%s/%-16s" % (level, pid, host)
                    if level == 'control':
                        # show control level first 
                        displayList.insert(0, display)
                    else:
                        displayList.append(display)

            if not args.noheader:
                print("Platform | Partition      |    Node")
                print("         | id/name        | level/pid/host")
                print("---------+----------------+------------------------------------")
            print("  %3s     %2s/%-12s" % (platform, platform, partName), end='')
            firstLine = True
            for nn in displayList:
                if firstLine:
                    print("  ", nn)
                    firstLine = False
                else:
                    print("                           ", nn)
            if firstLine:
                print()
            break          # Done
        else:
            print ("W: Received key \"%s\"" % request)
            continue

if __name__ == '__main__':
    main()
