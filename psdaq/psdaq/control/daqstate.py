#!/usr/bin/env python
"""
daqstate command
"""
import os
import sys
import time
import zmq
from psdaq.control.collection import rep_port, create_msg
from psdaq.control.collection import DaqControl
import argparse

def main():

    # process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=int, choices=range(0, 8), default=0, help='platform (default 0)')
    parser.add_argument('-C', metavar='COLLECT_HOST', default='localhost', help='collection host')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--state', choices=DaqControl.states)
    group.add_argument('--transition', choices=DaqControl.triggers)
    args = parser.parse_args()

    control = DaqControl(host=args.C, platform=args.p, timeout=10000)

    # initialize zmq socket
    try:
        context = zmq.Context(1)
        req = context.socket(zmq.REQ)
        req.linger = 0
        req.RCVTIMEO = 10000 # in milliseconds
        req.connect('tcp://%s:%d' % (args.C, rep_port(args.p)))
    except Exception as ex:
        print('Exception: %s' % ex)
        sys.exit(1)

    if args.state:
        # change the state
        rv = control.setstate(args.state)
        if rv is not None:
            print('Error: %s' % rv)

    elif args.transition:
        # transition request
        try:
            msg = create_msg(args.transition)
            req.send_json(msg)
            reply = req.recv_json()
        except Exception as ex:
            print('Exception: %s' % ex)
        except KeyboardInterrupt:
            pass
        else:
            try:
                key = reply['header']['key']
            except KeyError:
                key = ''

            errorMessage = None
            try:
                errorMessage = reply['body']['error']
            except Exception as ex:
                pass

            if errorMessage is not None:
                print('Error: %s' % errorMessage)

    # print current state
    print(control.getstate())

if __name__ == '__main__':
    main()
