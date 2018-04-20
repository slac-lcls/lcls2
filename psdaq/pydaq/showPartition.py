#!/usr/bin/env python
"""
CM showPartition command
"""
import sys
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
    parser.add_argument('--noheader', action='store_true', help='do not print header')
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
            if len(msg) == 4:
                props = pickle.loads(msg[3])

                # platform
                platform = 0
                try:
                    platform = props['platform']
                except:
                    print('E: platform key not found')

                # partition name
                partName = '(None)'
                try:
                    partName = props['partName']
                except KeyError:
                    print('E: partName key not found')

                # nodes
                nodes = []
                try:
                    nodes = pickle.loads(props['nodes'])
                except Exception:
                    print('E: nodes key not found')
                displayList = []
                for nn in nodes:
                    level = nn['level']
                    pid = nn['pid']
                    ip = nn['ip']
                    portDisplay = ""
                    if 'ports' in nn:
                        try:
                            ports = pickle.loads(nn['ports'])
                        except:
                            print ("E: pickle.loads()")
                            ports = []
                        if len(ports) > 0:
                            portDisplay = pprint.pformat(ports)
                    display = "%d/%05d/%-16s  %s" % (level, pid, ip, portDisplay)
                    displayList.append(display)

                if not args.noheader:
                    print("Platform | Partition  |    Node                 | Ports")
                    print("         | id/name    |  level/ pid /    ip     |")
                    print("---------+------------+-------------------------+----------")
                print("  %03d      %02d/%7s" % (platform, platform, partName), end='')
                firstLine = True
                for nn in sorted(displayList):
                    if firstLine:
                        print("  ", nn)
                        firstLine = False
                    else:
                        print("                       ", nn)
                if firstLine:
                    print()
            else:
                print ("E: STATE message len %d, expected 4" % len(msg))
            break          # Done
        else:
            print ("W: Received key \"%s\"" % request)
            continue

if __name__ == '__main__':
    main()
