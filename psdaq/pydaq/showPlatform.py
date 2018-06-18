#!/usr/bin/env python
"""
CM showPartition command
"""
import sys
import zmq
import zmq.utils.jsonapi as json
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
            msg = CMMsg.recv(cmd)
        except Exception as ex:
            print(ex)
            return

        request = msg.key
        if request == CMMsg.STATE:
            props = msg.properties

            # platform
            platform = '0'
            try:
                platform = props['platform']
            except KeyError:
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
                nodes = json.loads(msg.body)
            except Exception as ex:
                print('E: json.loads()', ex)
                nodes = []
            displayList = []
            for nn in nodes:
                try:
                    level = nn['level']
                except KeyError:
                    print('E: level key not found')
                    level = 0
                try:
                    pid = nn['pid']
                except KeyError:
                    print('E: pid key not found')
                    pid = 0
                try:
                    ip = nn['ip']
                except KeyError:
                    print('E: ip key not found')
                    pid = '0.0.0.0'
                try:
                    portDisplay = "%s" % nn['ports']
                except KeyError:
                    portDisplay = ""
                display = "%s/%s/%-16s  %s" % (level, pid, ip, portDisplay)
                displayList.append(display)

            if not args.noheader:
                print("Platform | Partition      |    Node                 | Ports")
                print("         | id/name        |  level/ pid /    ip     |")
                print("---------+----------------+-------------------------+----------")
            print("  %3s     %2s/%-12s" % (platform, platform, partName), end='')
            firstLine = True
            for nn in sorted(displayList):
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
