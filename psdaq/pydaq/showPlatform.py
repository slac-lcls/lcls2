#!/usr/bin/env python
"""
Collection Manager showPlatform command
"""
import sys
import zmq
import zmq.utils.jsonapi as json
import pprint
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
                    host = nn['host']
                except KeyError:
                    print('E: host key not found')
                    host = '(unknown)'
                try:
                    name = nn['name']
                except KeyError:
                    print('E: name key not found')
                    name = '(unknown)'
                # remove hostname suffix if present
                if host.count('.') > 0:
                    host = host.split('.')[0]
                display = "%s/%s/%s/%-16s" % (level, name, pid, host)
                displayList.append(display)

            if not args.noheader:
                print("Platform | Partition      |    Node")
                print("         | id/name        | level/name/pid/host")
                print("---------+----------------+------------------------------------")
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
