#!/usr/bin/env python
"""
showPlatform command
"""
import os
import time
import zmq
from collection2 import rep_port, create_msg
import pprint
import argparse

# Process arguments
parser = argparse.ArgumentParser()
parser.add_argument('-p', type=int, choices=range(0, 8), default=0, help='platform (default 0)')
args = parser.parse_args()
platform = args.p

partName = '(None)'
try:
    context = zmq.Context(1)
    req = context.socket(zmq.REQ)
    req.linger = 0
    req.RCVTIMEO = 5000 # in milliseconds
    req.connect('tcp://localhost:%d' % rep_port(platform))
    msg = create_msg('getstate')
    req.send_json(msg)
    reply = req.recv_json()
except Exception as ex:
    print('Exception: %s' % ex)
except KeyboardInterrupt:
    pass
else:
    key = 'error'
    try:
        key = reply['header']['key']
    except KeyError:
        pass

    if (key == 'error'):
        print('Error')
    else:
        displayList = []
        try:
            for level in reply['body']:
                for k, v in reply['body'][level].items():
                    host = v['proc_info']['host']
                    pid = v['proc_info']['pid']
                    display = "%s/%s/%-16s" % (level, pid, host)
                    if level == 'control':
                        # show control level first
                        displayList.insert(0, display)
                    else:
                        displayList.append(display)
        except KeyError as ex:
            print('Error: failed to parse reply: %s' % ex)
            pprint.pprint(reply)
        else:
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
