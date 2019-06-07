#!/usr/bin/env python

"""
Asynchronous Error Test
"""

import time
import socket
import zmq
from psdaq.control.control import back_pull_port, error_msg
import argparse

class Client:
    def __init__(self, platform, collectHost, alias, errMsg):

        # configure zmq sockets
        self.context = zmq.Context(1)
        self.push = self.context.socket(zmq.PUSH)
        self.push.connect('tcp://%s:%d' % (collectHost, back_pull_port(platform)))

        # send error message
        self.push.send_json(error_msg("%s: %s" % (alias, errMsg)))

def main():

    try:
        # process arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('-p', type=int, choices=range(0, 8), default=0, help='platform (default 0)')
        parser.add_argument('-C', metavar='COLLECT_HOST', default='localhost', help='collection host (default localhost)')
        parser.add_argument('-m', metavar='ERR_MSG', default='SHAZAM!', help='error message (default SHAZAM!)')
        parser.add_argument('-u', metavar='ALIAS', required=True, help='unique ID')
        args = parser.parse_args()

        # start client
        client = Client(args.p, args.C, args.u, args.m)

    except KeyboardInterrupt:
        print('KeyboardInterrupt')

if __name__ == '__main__':
    main()
